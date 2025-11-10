# -*- coding: utf-8 -*-
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmseg.models import build_head as build_seg_head
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.ops import resize
from mmcv.runner import get_dist_info, auto_fp16
from mmcv.parallel import DataContainer
from mmdet3d.models.utils import (
    FisheyeLUTCache,
    build_fisheye_lut,
    prepare_calibrations,
    make_lut_cache_key,
)

import copy


@DETECTORS.register_module()
class FastBEV(BaseDetector):
    def __init__(
        self,
        backbone,
        neck,
        neck_fuse,
        neck_3d,
        bbox_head,
        seg_head,
        n_voxels,
        voxel_size,
        bbox_head_2d=None,
        train_cfg=None,
        test_cfg=None,
        train_cfg_2d=None,
        test_cfg_2d=None,
        pretrained=None,
        init_cfg=None,
        extrinsic_noise=0,
        seq_detach=False,
        multi_scale_id=None,
        multi_scale_3d_scaler=None,
        with_cp=False,
        backproject='inplace',
        style='v4',
        fisheye_lut=None,
        num_views=6,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}', 
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)
        
        # style
        # v1: fastbev wo/ ms
        # v2: fastbev + img ms
        # v3: fastbev + bev ms
        # v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg)
            bbox_head.update(test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.bbox_head.voxel_size = voxel_size
        else:
            self.bbox_head = None

        if seg_head is not None:
            self.seg_head = build_seg_head(seg_head)
        else:
            self.seg_head = None

        if bbox_head_2d is not None:
            bbox_head_2d.update(train_cfg=train_cfg_2d)
            bbox_head_2d.update(test_cfg=test_cfg_2d)
            self.bbox_head_2d = build_head(bbox_head_2d)
        else:
            self.bbox_head_2d = None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        # detach adj feature
        self.seq_detach = seq_detach
        self.backproject = backproject
        # checkpoint
        self.with_cp = with_cp
        self.num_views = num_views
        assert self.num_views > 0, 'num_views 必须为正整数'

        # --- 鱼眼相机相关设置 -------------------------------------------------------
        self.fisheye_lut_cfg = fisheye_lut or {}
        # camera_model 决定是否启用鱼眼 LUT 分支
        self.camera_model = self.fisheye_lut_cfg.get('camera_model', 'perspective')
        assert self.camera_model in ['perspective', 'fisheye'], \
            f"Unsupported camera_model `{self.camera_model}`"
        if self.camera_model == 'fisheye':
            # 初始化 LUT 缓存配置，可指定磁盘缓存目录等
            cache_dir = self.fisheye_lut_cfg.get('cache_dir')
            save_to_disk = self.fisheye_lut_cfg.get('save_to_disk', True)
            self.fisheye_lut_cache = FisheyeLUTCache(
                cache_dir=cache_dir,
                save_to_disk=save_to_disk,
            )
            self.fisheye_distortion_key = self.fisheye_lut_cfg.get('distortion_key', 'distortion')
            self.fisheye_model_key = self.fisheye_lut_cfg.get('model_key', 'models')
            self.fisheye_intrinsic_key = self.fisheye_lut_cfg.get('intrinsic_key', 'intrinsic')
            # 可通过 force_rebuild 强制每次重算 LUT，方便调试新标定
            self.fisheye_force_rebuild = self.fisheye_lut_cfg.get('force_rebuild', False)
            # 多相机聚合方式，默认取平均，可改为 'max' 以增强边界
            self.fisheye_fusion_mode = self.fisheye_lut_cfg.get('fusion_mode', 'mean')
        else:
            self.fisheye_lut_cache = None
            self.fisheye_distortion_key = None
            self.fisheye_model_key = None
            self.fisheye_intrinsic_key = None
            self.fisheye_force_rebuild = False
            self.fisheye_fusion_mode = 'mean'

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.tensor(img_meta["lidar2img"]["intrinsic"][:3, :3])
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["lidar2img"]["extrinsic"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        img = img.reshape(
            [-1] + list(img.shape)[2:]
        )  # [1, 6, 3, 928, 1600] -> [6, 3, 928, 1600]
        x = self.backbone(
            img
        )  # [6, 256, 232, 400]; [6, 512, 116, 200]; [6, 1024, 58, 100]; [6, 2048, 29, 50]

        # use for vovnet
        if isinstance(x, dict):
            tmp = []
            for k in x.keys():
                tmp.append(x[k])
            x = tmp

        # fuse features
        def _inner_forward(x):
            out = self.neck(x)
            return out  # [6, 64, 232, 400]; [6, 64, 116, 200]; [6, 64, 58, 100]; [6, 64, 29, 50])

        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        features_2d = None
        if self.bbox_head_2d:
            features_2d = mlvl_feats

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i], 
                            size=mlvl_feats[msid].size()[2:], 
                            mode="bilinear", 
                            align_corners=False)
                        fuse_feats.append(resized_feat)
                
                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_
        # v3 bev ms
        if isinstance(self.n_voxels, list) and len(mlvl_feats) < len(self.n_voxels):
            pad_feats = len(self.n_voxels) - len(mlvl_feats)
            for _ in range(pad_feats):
                mlvl_feats.append(mlvl_feats[0])

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):  
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, self.num_views, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["lidar2img"]["extrinsic"] = img_meta["lidar2img"]["extrinsic"][
                        seq_id * self.num_views:(seq_id + 1) * self.num_views
                    ]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][
                            seq_id * self.num_views:(seq_id + 1) * self.num_views
                        ]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    if self.style in ['v1', 'v2']:
                        # wo/ bev ms
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        # v3/v4 bev ms
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["lidar2img"]["origin"]),
                    ).to(feat_i.device)

                    if self.camera_model == 'fisheye':
                        # 鱼眼模式：优先查找预计算 LUT，并将特征直接映射回 BEV
                        pixel_indices, valid_mask = self._get_fisheye_lut(
                            img_meta=img_meta,
                            points=points,
                            stride=stride_i,
                            height=height,
                            width=width,
                            voxel_size=torch.tensor(voxel_size, dtype=points.dtype, device=points.device),
                        )
                        volume = backproject_with_lut(
                            feat_i[:, :, :height, :width],
                            tuple(int(v) for v in n_voxels),
                            pixel_indices,
                            valid_mask,
                            fusion_mode=self.fisheye_fusion_mode,
                        )
                    else:
                        # 透视模式维持原始投影流程
                        projection = self._compute_projection(
                            img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                        if self.backproject == 'inplace':
                            volume = backproject_inplace(
                                feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                        else:
                            volume, valid = backproject_vanilla(
                                feat_i[:, :, :height, :width], points, projection)
                            volume = volume.sum(dim=0)
                            valid = valid.sum(dim=0)
                            volume = volume / valid
                            valid = valid > 0
                            volume[:, ~valid[0]] = 0.0
                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])
    
            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])
        
        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        else:
            # bev ms: multi-scale bev map (different x/y/z)
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z*c).permute(0, 3, 1, 2)
                
                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:  
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        x = mlvl_volumes
        def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            out = self.neck_3d(x)
            return out
            
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x, None, features_2d

    def _get_fisheye_lut(
        self,
        img_meta,
        points,
        stride,
        height,
        width,
        voxel_size,
    ):
        """依据当前样本标定信息获取（或创建）鱼眼 LUT，用于回填 BEV 体素。"""

        assert self.fisheye_lut_cache is not None, "Fish-eye LUT cache is not initialised."

        lidar2img = img_meta["lidar2img"]
        # 读取鱼眼相机参数（支持每帧多相机独立标定）
        intrinsic_raw = lidar2img.get(self.fisheye_intrinsic_key)
        if intrinsic_raw is None:
            raise KeyError("Fish-eye intrinsics not found in img_meta['lidar2img'].")
        # 兼容 3x3 或 4x4 齐次内参矩阵，只保留前 3x3 投影部分
        intrinsic = torch.as_tensor(intrinsic_raw)
        if intrinsic.ndim == 2 and intrinsic.shape[0] == 4:
            intrinsic = intrinsic[:3, :3]
        elif intrinsic.ndim == 3 and intrinsic.shape[1] == 4:
            intrinsic = intrinsic[:, :3, :3]

        # 外参通常为多个相机的 4x4 齐次矩阵
        extrinsics = [torch.as_tensor(ex) for ex in lidar2img["extrinsic"]]

        distortion_raw = None
        if self.fisheye_distortion_key is not None:
            distortion_raw = lidar2img.get(self.fisheye_distortion_key)
        distortion = None
        if distortion_raw is not None:
            # 畸变参数可能按相机存储成列表
            distortion = [torch.as_tensor(d) for d in distortion_raw]

        model_list = None
        if self.fisheye_model_key is not None and self.fisheye_model_key in lidar2img:
            # 如果数据集中每个相机使用不同鱼眼模型，可在此读取
            model_list = lidar2img[self.fisheye_model_key]

        calibrations = prepare_calibrations(
            intrinsic=intrinsic,
            extrinsics=extrinsics,
            distortion=distortion,
            model_per_cam=model_list,
        )

        origin_tensor = torch.as_tensor(
            lidar2img["origin"],
            dtype=points.dtype,
            device=points.device,
        )
        voxel_size_tensor = voxel_size.to(points.device)

        # 使用标定、体素参数与下采样步长生成缓存 key，保证 LUT 可复用
        cache_key = make_lut_cache_key(
            calibrations=calibrations,
            stride=stride,
            voxel_size=voxel_size_tensor,
            origin=origin_tensor,
        )

        if self.fisheye_force_rebuild:
            self.fisheye_lut_cache.memory_cache.pop(cache_key, None)
            if self.fisheye_lut_cache.cache_dir is not None:
                disk_path = self.fisheye_lut_cache.cache_dir / f"{cache_key}.pt"
                if disk_path.exists():
                    disk_path.unlink()

        def _builder():
            # 首次缓存未命中时触发构建逻辑
            with torch.no_grad():
                lut, valid = build_fisheye_lut(
                    points=points,
                    cameras=calibrations,
                    height=height,
                    width=width,
                    stride=stride,
                )
            return lut.cpu(), valid.cpu()

        # 尝试从缓存中读取 LUT，若不存在则调用 _builder 重新生成
        lut, valid = self.fisheye_lut_cache.get_lut(cache_key, _builder)
        if lut.dim() == 2 and lut.shape[1] == 2:
            raise RuntimeError(
                "检测到旧版鱼眼 LUT 格式，请删除缓存或设置 fisheye_lut.force_rebuild=True 重新生成。"
            )
        return lut.to(points.device), valid.to(points.device)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            if kwargs["export_2d"]:
                return self.onnx_export_2d(img, img_metas)
            elif kwargs["export_3d"]:
                return self.onnx_export_3d(img, img_metas)
            else:
                raise NotImplementedError

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(
        self, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bev_seg=None, **kwargs
    ):
        feature_bev, valids, features_2d = self.extract_feat(img, img_metas, "train")
        """
        feature_bev: [(1, 256, 100, 100)]
        valids: (1, 1, 200, 200, 12)
        features_2d: [[6, 64, 232, 400], [6, 64, 116, 200], [6, 64, 58, 100], [6, 64, 29, 50]]
        """
        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            loss_det = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses.update(loss_det)

        if self.seg_head is not None and gt_bev_seg is not None:
            x_bev = self.seg_head(feature_bev)

            if isinstance(gt_bev_seg, (list, tuple)):
                gt_list = []
                for seg in gt_bev_seg:
                    if not torch.is_tensor(seg):
                        seg = torch.as_tensor(seg, device=x_bev.device)
                    gt_list.append(seg.long().to(x_bev.device))
                gt_bev = torch.stack(gt_list, dim=0)
            else:
                gt_bev = gt_bev_seg.long().to(x_bev.device)

            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        if self.bbox_head_2d is not None and features_2d is not None:
            gt_bboxes_batch = kwargs.get("gt_bboxes", [])
            gt_labels_batch = kwargs.get("gt_labels", [])
            mv_bboxes_batch = []
            mv_labels_batch = []
            if "mv_bboxes" in kwargs:
                mv_raw = kwargs["mv_bboxes"]
                mv_bboxes_batch = mv_raw.data if isinstance(mv_raw, DataContainer) else mv_raw
            if "mv_labels" in kwargs:
                mv_raw = kwargs["mv_labels"]
                mv_labels_batch = mv_raw.data if isinstance(mv_raw, DataContainer) else mv_raw
            # fall back to mv_bboxes stored in img_metas when gt_bboxes is empty
            if len(gt_bboxes_batch) == 0:
                gt_bboxes_batch = [None] * len(img_metas)
                gt_labels_batch = [None] * len(img_metas)

            def _get_view_value(meta, key, idx, default=None):
                value = meta.get(key, default)
                if isinstance(value, (list, tuple)):
                    if idx < len(value):
                        return value[idx]
                    return value[0]
                return value

            base_device = None
            if isinstance(feature_bev, torch.Tensor):
                base_device = feature_bev.device
            elif isinstance(feature_bev, (list, tuple)) and feature_bev and isinstance(feature_bev[0], torch.Tensor):
                base_device = feature_bev[0].device
            if base_device is None and isinstance(features_2d, (list, tuple)) and features_2d and isinstance(features_2d[0], torch.Tensor):
                base_device = features_2d[0].device
            if base_device is None:
                base_device = torch.device('cpu')

            def _has_valid_targets(value):
                if value is None:
                    return False
                if isinstance(value, DataContainer):
                    value = value.data
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if _has_valid_targets(item):
                            return True
                    return False
                if torch.is_tensor(value):
                    return value.numel() > 0
                try:
                    return len(value) > 0
                except Exception:
                    return False

            def _reshape_tensor(tensor, is_bbox, device):
                tensor = tensor.to(device)
                if is_bbox:
                    tensor = tensor.to(dtype=torch.float32)
                    if tensor.numel() == 0:
                        return tensor.new_zeros((0, 4))
                    tensor = tensor.reshape(-1)
                    if tensor.numel() % 4 != 0:
                        raise ValueError(f"[Fast-BEV] Invalid bbox tensor with "
                                         f"numel={tensor.numel()}")
                    tensor = tensor.reshape(-1, 4)
                else:
                    tensor = tensor.to(dtype=torch.long)
                    tensor = tensor.reshape(-1)
                return tensor

            def _convert_to_view_list(value, is_bbox, device):
                if value is None:
                    return None
                if isinstance(value, DataContainer):
                    value = value.data
                if isinstance(value, (list, tuple)):
                    tensors = []
                    for item in value:
                        if item is None:
                            if is_bbox:
                                tensors.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                            else:
                                tensors.append(torch.zeros((0,), dtype=torch.long, device=device))
                            continue
                        if torch.is_tensor(item):
                            tensors.append(_reshape_tensor(item, is_bbox, device))
                        else:
                            dtype = torch.float32 if is_bbox else torch.long
                            tensor = torch.as_tensor(item, dtype=dtype, device=device)
                            tensors.append(_reshape_tensor(tensor, is_bbox, device))
                    return tensors
                if torch.is_tensor(value):
                    tensor = _reshape_tensor(value, is_bbox, device)
                else:
                    dtype = torch.float32 if is_bbox else torch.long
                    tensor = torch.as_tensor(value, dtype=dtype, device=device)
                    tensor = _reshape_tensor(tensor, is_bbox, device)
                return [tensor]

            def _prepare_targets(primary, fallback, num_views, is_bbox, device):
                source = primary if _has_valid_targets(primary) else None
                if source is None and _has_valid_targets(fallback):
                    source = fallback
                if source is None:
                    return None
                tensors = _convert_to_view_list(source, is_bbox, device)
                if tensors is None:
                    return None
                if len(tensors) < num_views:
                    pad_shape = (0, 4) if is_bbox else (0,)
                    pad = torch.zeros(pad_shape,
                                      dtype=torch.float32 if is_bbox else torch.long,
                                      device=device)
                    tensors = list(tensors) + [pad.clone() for _ in range(num_views - len(tensors))]
                elif len(tensors) > num_views:
                    tensors = tensors[:num_views]
                return tensors

            loss_2d_accum = {}
            processed = 0
            for sample_idx, meta in enumerate(img_metas):
                if sample_idx < len(gt_bboxes_batch):
                    gt_bboxes = gt_bboxes_batch[sample_idx]
                    gt_labels = gt_labels_batch[sample_idx]
                else:
                    gt_bboxes = None
                    gt_labels = None

                sample_feats = []
                for lvl_feat in features_2d:
                    start = sample_idx * self.num_views
                    end = (sample_idx + 1) * self.num_views
                    sample_feats.append(lvl_feat[start:end])
                device = sample_feats[0].device if sample_feats else base_device
                mv_bboxes_src = meta.get("mv_bboxes")
                if (not _has_valid_targets(mv_bboxes_src)) and mv_bboxes_batch:
                    mv_bboxes_src = mv_bboxes_batch[sample_idx] if sample_idx < len(mv_bboxes_batch) else None
                if (not _has_valid_targets(mv_bboxes_src)) and "ann_info" in meta:
                    mv_bboxes_src = meta["ann_info"].get("mv_bboxes")

                mv_labels_src = meta.get("mv_labels")
                if (not _has_valid_targets(mv_labels_src)) and mv_labels_batch:
                    mv_labels_src = mv_labels_batch[sample_idx] if sample_idx < len(mv_labels_batch) else None
                if (not _has_valid_targets(mv_labels_src)) and "ann_info" in meta:
                    mv_labels_src = meta["ann_info"].get("mv_labels")

                num_views = len(meta["img_info"])
                gt_bboxes_list = _prepare_targets(
                    gt_bboxes, mv_bboxes_src, num_views, True, device)
                gt_labels_list = _prepare_targets(
                    gt_labels, mv_labels_src, num_views, False, device)

                if gt_bboxes_list is None or gt_labels_list is None:
                    continue

                # ensure each view has targets tensor
                if len(gt_bboxes_list) != len(gt_labels_list):
                    continue

                if len(gt_bboxes_list) == 0 or len(gt_labels_list) == 0:
                    continue

                img_metas_2d = []
                img_info_list = meta["img_info"]
                for view_idx, info in enumerate(img_info_list):
                    pad_shape = _get_view_value(meta, "pad_shape", view_idx, meta.get("img_shape"))
                    if pad_shape is None:
                        pad_shape = meta.get("ori_shape")
                    img_shape = _get_view_value(meta, "img_shape", view_idx, meta.get("ori_shape"))
                    ori_shape = _get_view_value(meta, "ori_shape", view_idx, img_shape)
                    if isinstance(pad_shape, list):
                        pad_shape = tuple(pad_shape)
                    if isinstance(img_shape, list):
                        img_shape = tuple(img_shape)
                    if isinstance(ori_shape, list):
                        ori_shape = tuple(ori_shape)
                    scale_factor = _get_view_value(meta, "scale_factor", view_idx, 1.0)
                    tmp_dict = dict(
                        filename=info["filename"],
                        ori_filename=info["filename"].split("/")[-1],
                        ori_shape=ori_shape,
                        img_shape=img_shape,
                        pad_shape=pad_shape,
                        scale_factor=scale_factor,
                        flip=False,
                        flip_direction=None,
                    )
                    img_metas_2d.append(tmp_dict)

                loss_2d = self.bbox_head_2d.forward_train(
                    sample_feats, img_metas_2d, gt_bboxes_list, gt_labels_list
                )
                for k, v in loss_2d.items():
                    if k in loss_2d_accum:
                        loss_2d_accum[k] = loss_2d_accum[k] + v
                    else:
                        loss_2d_accum[k] = v
                processed += 1

            if processed > 0:
                for k, v in loss_2d_accum.items():
                    mean_loss = v / processed
                    if k in losses:
                        losses[k] = losses[k] + mean_loss
                    else:
                        losses[k] = mean_loss

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if not self.test_cfg.get('use_tta', False):
            return self.simple_test(img, img_metas)
        return self.aug_test(img, img_metas)

    def onnx_export_2d(self, img, img_metas):
        """
        input: 6, 3, 544, 960
        output: 6, 64, 136, 240
        """
        x = self.backbone(img)
        c1, c2, c3, c4 = self.neck(x)
        c2 = resize(
            c2, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c3 = resize(
            c3, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        c4 = resize(
            c4, size=c1.size()[2:], mode="bilinear", align_corners=False
        )  # [6, 64, 232, 400]
        x = torch.cat([c1, c2, c3, c4], dim=1)
        x = self.neck_fuse(x)

        if bool(os.getenv("DEPLOY", False)):
            x = x.permute(0, 2, 3, 1)
            return x

        return x

    def onnx_export_3d(self, x, _):
        # x: [6, 200, 100, 3, 256]
        # if bool(os.getenv("DEPLOY_DEBUG", False)):
        #     x = x.sum(dim=0, keepdim=True)
        #     return [x]
        if self.style == "v1":
            x = x.sum(dim=0, keepdim=True)  # [1, 200, 100, 3, 256]
            x = self.neck_3d(x)  # [[1, 256, 100, 50], ]
        elif self.style == "v2":
            x = self.neck_3d(x)  # [6, 256, 100, 50]
            x = [x[0].sum(dim=0, keepdim=True)]  # [1, 256, 100, 50]
        elif self.style == "v3":
            x = self.neck_3d(x)  # [1, 256, 100, 50]
        else:
            raise NotImplementedError

        if self.bbox_head is not None:
            cls_score, bbox_pred, dir_cls_preds = self.bbox_head(x)
            cls_score = [item.sigmoid() for item in cls_score]

        if os.getenv("DEPLOY", False):
            if dir_cls_preds is None:
                x = [cls_score, bbox_pred]
            else:
                x = [cls_score, bbox_pred, dir_cls_preds]
            return x

        return x

    def simple_test(self, img, img_metas):
        bbox_results = []
        feature_bev, _, features_2d = self.extract_feat(img, img_metas, "test")
        if self.bbox_head is not None:
            x = self.bbox_head(feature_bev)
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas, valid=None)
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]
        else:
            bbox_results = [dict() for _ in range(len(img_metas))]

        # BEV semantic seg
        if self.seg_head is not None:
            x_bev = self.seg_head(feature_bev)
            for idx, bev_res in enumerate(x_bev):
                bbox_results[idx]['bev_seg'] = bev_res

        if self.bbox_head_2d is not None and features_2d is not None:
            mv_results = self._simple_test_multiview_2d(features_2d, img_metas)
            for sample_idx, mv_res in enumerate(mv_results):
                bbox_results[sample_idx].update(mv_res)

        return bbox_results

    def _simple_test_multiview_2d(self, features_2d, img_metas):
        mv_results = []
        num_samples = len(img_metas)

        def _get_view_value(meta, key, idx, default=None):
            value = meta.get(key, default)
            if isinstance(value, (list, tuple)):
                if idx < len(value):
                    return value[idx]
                return value[0]
            return value

        for sample_idx in range(num_samples):
            sample_feats = []
            for lvl_feat in features_2d:
                start = sample_idx * self.num_views
                end = (sample_idx + 1) * self.num_views
                sample_feats.append(lvl_feat[start:end])
            if not sample_feats or sample_feats[0].numel() == 0:
                mv_results.append(dict(
                    mv_bboxes=[np.zeros((0, 4), dtype=np.float32) for _ in range(self.num_views)],
                    mv_scores=[np.zeros((0,), dtype=np.float32) for _ in range(self.num_views)],
                    mv_labels=[np.zeros((0,), dtype=np.int64) for _ in range(self.num_views)],
                ))
                continue

            img_metas_2d = []
            img_info_list = img_metas[sample_idx].get("img_info", [])
            for view_idx in range(self.num_views):
                info = img_info_list[view_idx] if view_idx < len(img_info_list) else {}
                pad_shape = _get_view_value(img_metas[sample_idx], "pad_shape", view_idx, img_metas[sample_idx].get("img_shape"))
                if pad_shape is None:
                    pad_shape = img_metas[sample_idx].get("ori_shape")
                img_shape = _get_view_value(img_metas[sample_idx], "img_shape", view_idx, img_metas[sample_idx].get("ori_shape"))
                ori_shape = _get_view_value(img_metas[sample_idx], "ori_shape", view_idx, img_shape)
                if isinstance(pad_shape, list):
                    pad_shape = tuple(pad_shape)
                if isinstance(img_shape, list):
                    img_shape = tuple(img_shape)
                if isinstance(ori_shape, list):
                    ori_shape = tuple(ori_shape)
                scale_factor = _get_view_value(img_metas[sample_idx], "scale_factor", view_idx, 1.0)
                tmp_dict = dict(
                    filename=info.get("filename", ""),
                    ori_filename=info.get("filename", "").split("/")[-1] if info.get("filename") else "",
                    ori_shape=ori_shape,
                    img_shape=img_shape,
                    pad_shape=pad_shape,
                    scale_factor=scale_factor,
                    flip=False,
                    flip_direction=None,
                )
                img_metas_2d.append(tmp_dict)

            det_results = self.bbox_head_2d.simple_test_bboxes(
                sample_feats, img_metas_2d, rescale=False)

            mv_bboxes = []
            mv_scores = []
            mv_labels = []
            for det_bbox, det_label in det_results:
                if det_bbox.numel() == 0:
                    mv_bboxes.append(np.zeros((0, 4), dtype=np.float32))
                    mv_scores.append(np.zeros((0,), dtype=np.float32))
                    mv_labels.append(np.zeros((0,), dtype=np.int64))
                    continue
                det_bbox_np = det_bbox.detach().cpu().numpy()
                det_label_np = det_label.detach().cpu().numpy()
                mv_bboxes.append(det_bbox_np[:, :4])
                mv_scores.append(det_bbox_np[:, 4])
                mv_labels.append(det_label_np.astype(np.int64))

            mv_results.append(dict(
                mv_bboxes=mv_bboxes,
                mv_scores=mv_scores,
                mv_labels=mv_labels,
            ))
        return mv_results

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):

            img_metas[0]['img_shape'] = img_shape_copy[24*tta_id:24*(tta_id+1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24*tta_id:24*(tta_id+1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24*tta_id:24*(tta_id+1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


@torch.no_grad()
def backproject_with_lut(features, n_voxels, pixel_indices, valid_mask, fusion_mode='mean'):
    '''
    功能：利用预先计算好的鱼眼 LUT，将多相机特征快速回投影到 BEV 体素。
    输入:
        features: [nv, c, h, w]
        n_voxels: (vx, vy, vz)
        pixel_indices: [nv, vx*vy*vz] -> y * width + x
        valid_mask: [nv, vx*vy*vz] -> bool
        fusion_mode: 聚合方式，支持 'mean'、'max'
    输出:
        volume: [c, vx, vy, vz]
    '''

    n_images, n_channels, height, width = features.shape
    total_voxels = int(n_voxels[0] * n_voxels[1] * n_voxels[2])
    assert pixel_indices.shape == (n_images, total_voxels), \
        "LUT 与相机/体素数量不匹配"
    assert valid_mask.shape == (n_images, total_voxels), \
        "valid_mask 与相机/体素数量不匹配"

    pixel_indices = pixel_indices.to(features.device)
    valid_mask = valid_mask.to(features.device)

    # 展平特征以便快速索引采样
    features_flat = features.reshape(n_images, n_channels, height * width)

    if fusion_mode not in ['mean', 'max']:
        raise ValueError(f"不支持的融合方式 {fusion_mode}")

    volume = torch.zeros(
        (n_channels, total_voxels),
        device=features.device,
        dtype=features.dtype,
    )

    if fusion_mode == 'mean':
        # 平均池化：记录累积值与有效相机数量
        counts = torch.zeros(
            total_voxels, device=features.device, dtype=features.dtype
        )
    elif fusion_mode == 'max':
        # 最大池化：以极小值初始化，后续逐相机取最大
        volume.fill_(-torch.finfo(features.dtype).max)

    for cam_id in range(n_images):
        cam_mask = valid_mask[cam_id]
        if not cam_mask.any():
            continue
        dst_indices = torch.nonzero(cam_mask, as_tuple=False).squeeze(1)
        src_indices = pixel_indices[cam_id, cam_mask].to(torch.long)
        sampled = features_flat[cam_id, :, src_indices]

        if fusion_mode == 'mean':
            volume[:, dst_indices] += sampled
            counts[dst_indices] += 1
        else:
            # 最大池化：逐像素比较，保留强响应
            volume[:, dst_indices] = torch.maximum(
                volume[:, dst_indices], sampled
            )

    if fusion_mode == 'mean':
        valid = counts > 0
        if valid.any():
            volume[:, valid] = volume[:, valid] / counts[valid]
        volume[:, ~valid] = 0
    else:
        # max 模式下，将未被任何相机覆盖的体素置零
        cover = valid_mask.any(dim=0)
        volume[:, ~cover] = 0

    return volume.view(n_channels, *n_voxels)


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume
