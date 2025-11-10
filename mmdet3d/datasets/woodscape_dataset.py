# -*- coding: utf-8 -*-
import copy
import warnings
from typing import Dict, List, Sequence, Union

import mmcv
import numpy as np
import torch
from mmdet.datasets import DATASETS

from .nuscenes_monocular_dataset import NuScenesMultiViewDataset
from mmdet3d.core.bbox import bbox_overlaps_nearest_3d


@DATASETS.register_module()
class WoodScapeMultiViewDataset(NuScenesMultiViewDataset):
    """WoodScape 多视角（鱼眼）数据集适配。

    该数据集假设使用 WoodScape 官方提供的 `woodscape_infos_*.pkl`，
    仅包含 4 个鱼眼相机。若需要其他相机组合，可通过 `camera_types`
    参数自定义。
    """

    DEFAULT_CAMERAS = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
    ]

    CLASSES = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')

    def __init__(self,
                 camera_types: List[str] = None,
                 fill_identity_lidar: bool = True,
                 with_box2d: bool = False,
                 **kwargs):
        self.camera_types = camera_types or copy.deepcopy(self.DEFAULT_CAMERAS)
        self.fill_identity_lidar = fill_identity_lidar
        self.with_box2d = with_box2d
        if 'with_box2d' in kwargs:
            kwargs.pop('with_box2d')
        kwargs.setdefault('use_valid_flag', True)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # 数据加载与字段补齐
    # ------------------------------------------------------------------ #
    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        infos = data['infos']
        metadata = data.get('metadata', {})
        self.metadata = metadata
        self.version = metadata.get('version', 'woodscape-v1')

        processed = []
        for raw_info in infos:
            info = copy.deepcopy(raw_info)
            info['cams'] = self._process_cams(info.get('cams', {}))
            info['lidar_path'] = info.get('lidar_path', '')
            info['sweeps'] = info.get('sweeps', [])
            info['lidar2ego_translation'] = info.get('lidar2ego_translation', [0.0, 0.0, 0.0])
            info['lidar2ego_rotation'] = info.get('lidar2ego_rotation', [0.0, 0.0, 0.0, 1.0])
            info['ego2global_translation'] = info.get('ego2global_translation', [0.0, 0.0, 0.0])
            info['ego2global_rotation'] = info.get('ego2global_rotation', [0.0, 0.0, 0.0, 1.0])
            info['prev'] = info.get('prev', None)
            info['next'] = info.get('next', None)
            info['velo'] = info.get('velo', np.zeros(2, dtype=np.float32))

            info['gt_boxes'] = self._to_float_array(info.get('gt_boxes', []), shape=(-1, 9))
            info['gt_velocity'] = self._to_float_array(info.get('gt_velocity', []), shape=(-1, 2))
            info['gt_names'] = np.asarray(info.get('gt_names', []), dtype=object)
            info['num_lidar_pts'] = self._to_int_array(info.get('num_lidar_pts', []))
            info['num_radar_pts'] = self._to_int_array(info.get('num_radar_pts', []))
            valid_flag = np.asarray(info.get('valid_flag', []), dtype=bool)
            if valid_flag.size == 0:
                valid_flag = np.ones(info['gt_boxes'].shape[0], dtype=bool)
            info['valid_flag'] = valid_flag

            # ------------------------------------------------------------------ #
            # 将可选的 BEV / 语义 标注路径整合进 info，便于 pipeline 加载。
            # 配置产出的 info 建议在 ann_info 中附加 bev_seg_path / motion_seg_path 等键。
            # ------------------------------------------------------------------ #
            ann_info = raw_info.get('ann_info', {})
            if isinstance(ann_info, dict):
                bev_seg = ann_info.get('gt_bev_seg') or ann_info.get('bev_seg_path')
                if bev_seg is not None:
                    info.setdefault('ann_info', {})['gt_bev_seg'] = bev_seg
                bev_class = ann_info.get('bev_seg_classes')
                if bev_class is not None:
                    info.setdefault('ann_info', {})['bev_seg_classes'] = bev_class
                bbox2d = ann_info.get('bboxes')
                labels2d = ann_info.get('labels')
                if bbox2d is not None and labels2d is not None:
                    info.setdefault('ann_info', {})['mv_bboxes'] = bbox2d
                    info['ann_info']['mv_labels'] = labels2d
                    flat_boxes = [
                        np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                        for boxes in bbox2d
                    ]
                    flat_labels = [
                        np.asarray(lbls, dtype=np.int64).reshape(-1)
                        for lbls in labels2d
                    ]
                    if flat_boxes:
                        bboxes_concat = np.concatenate(flat_boxes, axis=0)
                        labels_concat = np.concatenate(flat_labels, axis=0)
                    else:
                        bboxes_concat = np.zeros((0, 4), dtype=np.float32)
                        labels_concat = np.zeros((0,), dtype=np.int64)
                    info['ann_info']['bboxes'] = bboxes_concat
                    info['ann_info']['labels'] = labels_concat
                motion_path = ann_info.get('motion_seg_path')
                if motion_path is not None:
                    info.setdefault('ann_info', {})['motion_seg_path'] = motion_path

            processed.append(info)

        return processed

    def get_data_info(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'\[Fast-BEV] Expected 6 cameras, got [0-9]+\. '
                        r'Continue with adaptive visualization\.',
                category=UserWarning)
            info = super().get_data_info(index)

        ann = info.setdefault('ann_info', {})
        src_ann = self.data_infos[index].get('ann_info', {})
        mv_bboxes = copy.deepcopy(src_ann.get('mv_bboxes'))
        mv_labels = copy.deepcopy(src_ann.get('mv_labels'))
        if mv_bboxes is not None and mv_labels is not None:
            ann['mv_bboxes'] = mv_bboxes
            ann['mv_labels'] = mv_labels
            info['mv_bboxes'] = copy.deepcopy(mv_bboxes)
            info['mv_labels'] = copy.deepcopy(mv_labels)

        # 拷贝关键 3D/BEV 标注，保证 test_mode 下也能加载
        full_ann = self.get_ann_info(index)
        if isinstance(full_ann, dict):
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_seg', 'bev_seg_classes']:
                if key in full_ann:
                    ann[key] = copy.deepcopy(full_ann[key])

        lidar2img = info.get('lidar2img')
        if isinstance(lidar2img, dict):
            extrinsics = lidar2img.get('extrinsic', [])
            lidar2img_aug = lidar2img.get('lidar2img_aug', [])
            lidar2img_extra = lidar2img.get('lidar2img_extra', [])
        else:
            extrinsics = lidar2img
            lidar2img_aug = info.get('lidar2img_aug', [])
            lidar2img_extra = info.get('lidar2img_extra', [])

        intrinsics = []
        distortions = []
        models = []
        for cam_name in self.camera_types:
            cam_meta = self.data_infos[index]['cams'][cam_name]
            intrinsics.append(self._to_float_array(
                cam_meta.get('cam_intrinsic', np.eye(3, dtype=np.float32)),
                shape=(3, 3)))
            distortions.append(self._to_float_array(
                cam_meta.get('cam_distortion', np.zeros(4)), shape=(-1,)))
            models.append(cam_meta.get('cam_model', 'polynomial'))
        intr_stack = np.stack(intrinsics, axis=0)
        distortion_list = [d.astype(np.float32) for d in distortions]
        distortion_primary = distortion_list[0] if distortion_list else np.zeros(4, dtype=np.float32)

        extrinsics = [self._to_float_array(x, shape=(4, 4)) for x in extrinsics]

        info['lidar2img'] = dict(
            extrinsic=extrinsics,
            lidar2img_aug=lidar2img_aug,
            lidar2img_extra=lidar2img_extra,
            intrinsic=intr_stack,
            intrinsics=intr_stack,
            distortion=distortion_primary,
            distortions=distortion_list,
            models=models,
            origin=np.zeros(3, dtype=np.float32),
        )

        return info

    # ------------------------------------------------------------------ #
    # 私有工具函数
    # ------------------------------------------------------------------ #
    def _process_cams(self, cams: Dict[str, Dict]) -> Dict[str, Dict]:
        out = {}
        for cam_name, cam_info in cams.items():
            if cam_name not in self.camera_types:
                continue
            cam = copy.deepcopy(cam_info)
            if self.fill_identity_lidar:
                cam.setdefault('sensor2lidar_rotation', np.eye(3, dtype=np.float32))
                cam.setdefault('sensor2lidar_translation', np.zeros(3, dtype=np.float32))
            cam['sensor2lidar_rotation'] = self._to_float_array(
                cam['sensor2lidar_rotation'], shape=(3, 3))
            cam['sensor2lidar_translation'] = self._to_float_array(
                cam['sensor2lidar_translation'], shape=(3,))
            cam['sensor2ego_translation'] = self._to_float_array(
                cam.get('sensor2ego_translation', [0.0, 0.0, 0.0]), shape=(3,))
            cam['sensor2ego_rotation'] = self._to_float_array(
                cam.get('sensor2ego_rotation', [0.0, 0.0, 0.0, 1.0]), shape=(4,))
            cam['ego2global_translation'] = self._to_float_array(
                cam.get('ego2global_translation', [0.0, 0.0, 0.0]), shape=(3,))
            cam['ego2global_rotation'] = self._to_float_array(
                cam.get('ego2global_rotation', [0.0, 0.0, 0.0, 1.0]), shape=(4,))
            cam['cam_intrinsic'] = self._to_float_array(
                cam.get('cam_intrinsic', np.eye(3, dtype=np.float32)), shape=(3, 3))
            cam['cam_distortion'] = self._to_float_array(
                cam.get('cam_distortion', np.zeros(4)), shape=(-1,))
            cam['cam_model'] = cam.get('cam_model', 'polynomial')

            annos = cam.get('annos', {})
            if isinstance(annos, dict):
                converted = {}
                for key, val in annos.items():
                    if key == 'bbox':
                        converted[key] = np.asarray(val, dtype=np.float32).reshape(-1, 4)
                    elif key in ('category_id', 'instance_id'):
                        converted[key] = np.asarray(val, dtype=np.int64)
                    elif key == 'category_name':
                        converted[key] = np.asarray(val, dtype=object)
                    else:
                        converted[key] = np.asarray(val)
                cam['annos'] = converted
            else:
                cam['annos'] = dict()
            out[cam_name] = cam
        return out

    @staticmethod
    def _to_float_array(value, shape=None):
        arr = np.asarray(value, dtype=np.float32)
        if shape is not None:
            if arr.size == 0:
                target_shape = tuple(0 if dim == -1 else dim for dim in shape)
                arr = np.zeros(target_shape, dtype=np.float32)
            else:
                arr = arr.reshape(shape)
        return arr

    @staticmethod
    def _to_int_array(value):
        arr = np.asarray(value, dtype=np.int64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    # ------------------------------------------------------------------ #
    # 评估阶段：基于 IoU=0.5 计算简单 mAP
    # ------------------------------------------------------------------ #
    def evaluate(self,
                 results,
                 metric: Union[str, Sequence[str]] = None,
                 logger=None,
                 iou_thr: float = 0.5,
                 score_thr: float = 0.0,
                 eval_2d: bool = False,
                 eval_3d: bool = True,
                 eval_bev: bool = False,
                 **kwargs):
        """Evaluate detection results with a simple IoU-based mAP.

        Args:
            results (list[dict]): 推理输出，要求包含 ``boxes_3d``、
                ``scores_3d``、``labels_3d``。
            metric (str | Sequence[str], optional): 当前仅支持 ``'mAP'``。
            logger: 记录器。
            iou_thr (float): IoU 阈值，默认 0.5。
            score_thr (float): 过滤预测框的得分阈值，默认 0（不过滤）。
            eval_2d (bool): 是否同时计算 2D 检测指标。要求 results 中提供
                ``mv_bboxes``（每视角框列表）和 ``mv_scores``。

        Returns:
            dict: 包含各类别 AP 以及 mean AP。
        """
        if metric is None:
            metric = ('mAP', )
        if isinstance(metric, str):
            metric = (metric, )
        allowed_metrics = {'mAP'}
        for m in metric:
            if m not in allowed_metrics:
                raise KeyError(f'不支持的评价指标: {m}')

        assert len(results) == len(self), \
            f'结果数量 {len(results)} 与数据集大小 {len(self)} 不一致'

        eval_results = {}
        if eval_3d:
            if metric is None:
                metric = ('mAP', )
            if isinstance(metric, str):
                metric = (metric, )
            allowed_metrics = {'mAP'}
            for m in metric:
                if m not in allowed_metrics:
                    raise KeyError(f'不支持的评价指标: {m}')

            num_classes = len(self.CLASSES)
            gt_counter = [0] * num_classes
            per_cls_scores = [[] for _ in range(num_classes)]
            per_cls_tps = [[] for _ in range(num_classes)]
            per_cls_fps = [[] for _ in range(num_classes)]

            def _align_dims(b1, b2):
                if b1.size(-1) == b2.size(-1):
                    return b1, b2
                shared = min(b1.size(-1), b2.size(-1))
                assert shared >= 7, \
                    f'共享维度 {shared} < 7，无法计算 IoU'
                return b1[..., :shared], b2[..., :shared]

            for idx, result in enumerate(results):
                gt_ann = self.get_ann_info(idx)
                gt_boxes = gt_ann['gt_bboxes_3d'].tensor.cpu()
                gt_labels = torch.from_numpy(gt_ann['gt_labels_3d']).long()

                pred_boxes = result['boxes_3d'].convert_to(self.box_mode_3d).tensor.cpu()
                pred_scores = result['scores_3d'].detach().cpu()
                pred_labels = result['labels_3d'].detach().cpu().long()

                if score_thr > 0:
                    keep_mask = pred_scores >= score_thr
                    pred_boxes = pred_boxes[keep_mask]
                    pred_scores = pred_scores[keep_mask]
                    pred_labels = pred_labels[keep_mask]

                for cls_id in range(num_classes):
                    gt_mask = (gt_labels == cls_id)
                    gt_cls_boxes = gt_boxes[gt_mask]
                    gt_counter[cls_id] += gt_cls_boxes.size(0)

                    pred_mask = (pred_labels == cls_id)
                    boxes_cls = pred_boxes[pred_mask]
                    scores_cls = pred_scores[pred_mask]

                    if boxes_cls.size(0) == 0:
                        continue

                    order = torch.argsort(scores_cls, descending=True)
                    boxes_cls = boxes_cls[order]
                    scores_cls = scores_cls[order]

                    per_cls_scores[cls_id].extend(scores_cls.tolist())

                    if gt_cls_boxes.size(0) == 0:
                        per_cls_tps[cls_id].extend([0] * boxes_cls.size(0))
                        per_cls_fps[cls_id].extend([1] * boxes_cls.size(0))
                        continue

                    aligned_gt, aligned_pred = _align_dims(gt_cls_boxes, boxes_cls)
                    ious = bbox_overlaps_nearest_3d(
                        aligned_gt.float(), aligned_pred.float())
                    matched = torch.zeros(aligned_gt.size(0), dtype=torch.bool)
                    for pred_id in range(aligned_pred.size(0)):
                        iou_vec = ious[:, pred_id]
                        max_iou, max_idx = iou_vec.max(dim=0)
                        if max_iou >= iou_thr and not matched[max_idx]:
                            matched[max_idx] = True
                            per_cls_tps[cls_id].append(1)
                            per_cls_fps[cls_id].append(0)
                        else:
                            per_cls_tps[cls_id].append(0)
                            per_cls_fps[cls_id].append(1)

            aps = []
            for cls_id, cls_name in enumerate(self.CLASSES):
                gt_num = gt_counter[cls_id]
                scores = np.asarray(per_cls_scores[cls_id], dtype=np.float32)
                tps = np.asarray(per_cls_tps[cls_id], dtype=np.float32)
                fps = np.asarray(per_cls_fps[cls_id], dtype=np.float32)

                if gt_num == 0:
                    eval_results[f'AP_{cls_name}@{iou_thr}'] = float('nan')
                    continue

                if scores.size == 0:
                    eval_results[f'AP_{cls_name}@{iou_thr}'] = 0.0
                    aps.append(0.0)
                    continue

                order = scores.argsort()[::-1]
                tps = tps[order]
                fps = fps[order]
                cum_tp = np.cumsum(tps)
                cum_fp = np.cumsum(fps)
                recalls = cum_tp / max(gt_num, 1)
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
                ap = self._compute_ap(recalls, precisions)
                eval_results[f'AP_{cls_name}@{iou_thr}'] = ap
                aps.append(ap)

            eval_results[f'mAP@{iou_thr}'] = float(np.mean(aps)) if aps else float('nan')

        # ------------------------------------------------------------------ #
        # 可选：统计多视角 2D 检测指标。这里采用简化版 IoU=0.5 计算流程，
        # 将所有视角的预测拼接后计算 P-R 曲线。若业务需要更精细的 per-camera
        # 统计，可在此基础上补充。
        # ------------------------------------------------------------------ #
        if eval_2d:
            two_d_results = self._evaluate_multiview_bbox(results, score_thr=score_thr)
            eval_results.update(two_d_results)

        if eval_bev:
            bev_results = self._evaluate_bev_seg(results)
            eval_results.update(bev_results)

        if logger is not None:
            if eval_3d:
                logger.info('WoodScape evaluation (IoU {:.2f}): {}'.format(
                    iou_thr, eval_results))
            else:
                logger.info('WoodScape 2D evaluation: {}'.format(eval_results))

        return eval_results

    def get_ann_info(self, index):
        """扩展父类 get_ann_info，补充多任务所需标注。"""
        base_info = super().get_ann_info(index)
        data_info = self.data_infos[index]
        ann_info = copy.deepcopy(data_info.get('ann_info', {}))

        bev_seg = ann_info.get('gt_bev_seg')
        if bev_seg is not None:
            base_info['gt_bev_seg'] = bev_seg

        bev_classes = ann_info.get('bev_seg_classes')
        if bev_classes is not None:
            base_info['bev_seg_classes'] = bev_classes

        mv_bboxes = ann_info.get('mv_bboxes')
        mv_labels = ann_info.get('mv_labels')
        if mv_bboxes is not None and mv_labels is not None:
            base_info['mv_bboxes'] = mv_bboxes
            base_info['mv_labels'] = mv_labels
        bboxes = ann_info.get('bboxes')
        labels = ann_info.get('labels')
        if bboxes is not None and labels is not None:
            base_info['bboxes'] = bboxes
            base_info['labels'] = labels

        motion_seg = ann_info.get('motion_seg_path')
        if motion_seg is not None:
            base_info['motion_seg_path'] = motion_seg

        return base_info

    def _evaluate_multiview_bbox(self,
                                 results: List[dict],
                                 iou_thr: float = 0.5,
                                 score_thr: float = 0.0):
        """计算多视角 2D 检测的 mAP。"""
        all_scores = []
        all_tp = []
        all_fp = []
        total_gt = 0

        for idx, result in enumerate(results):
            gt_ann = self.get_ann_info(idx)
            mv_bboxes_gt = gt_ann.get('mv_bboxes', [])
            mv_labels_gt = gt_ann.get('mv_labels', [])
            mv_bboxes_pred = result.get('mv_bboxes', [])
            mv_scores_pred = result.get('mv_scores', [])
            mv_labels_pred = result.get('mv_labels', [])

            if not mv_bboxes_pred or not mv_bboxes_gt:
                continue

            for cam_idx, (gt_boxes, gt_labels) in enumerate(zip(mv_bboxes_gt, mv_labels_gt)):
                total_gt += gt_boxes.shape[0]
                pred_boxes = mv_bboxes_pred[cam_idx]
                pred_scores = mv_scores_pred[cam_idx]
                pred_labels = mv_labels_pred[cam_idx]

                if pred_boxes.shape[0] == 0:
                    continue

                keep = pred_scores >= score_thr
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]

                if pred_boxes.size == 0:
                    continue

                order = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]
                pred_labels = pred_labels[order]

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    all_scores.append(score)
                    ious = self._bbox_iou_2d(box, gt_boxes)
                    best_iou = ious.max() if ious.size else 0.0
                    if best_iou >= iou_thr:
                        all_tp.append(1)
                        all_fp.append(0)
                    else:
                        all_tp.append(0)
                        all_fp.append(1)

        if not all_scores:
            return {'mAP_2d@0.5': float('nan')}

        scores = np.asarray(all_scores)
        tp = np.asarray(all_tp)
        fp = np.asarray(all_fp)
        order = np.argsort(-scores)
        tp = tp[order]
        fp = fp[order]
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / max(total_gt, 1)
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        ap2d = self._compute_ap(recalls, precisions)
        return {'mAP_2d@0.5': ap2d}

    def _evaluate_bev_seg(self, results: List[dict]):
        """Evaluate BEV segmentation predictions."""
        bev_class_names = []
        for info in self.data_infos:
            ann = info.get('ann_info', {})
            bev_class_names = ann.get('bev_seg_classes')
            if bev_class_names:
                break
        if not bev_class_names:
            bev_class_names = []
        num_classes = len(bev_class_names)
        if num_classes == 0:
            sample_pred = next(
                (res.get('bev_seg') for res in results if res.get('bev_seg') is not None),
                None)
            if sample_pred is not None:
                if torch.is_tensor(sample_pred):
                    shape = sample_pred.shape
                else:
                    shape = np.asarray(sample_pred).shape
                if len(shape) >= 3:
                    num_classes = shape[1] if len(shape) == 4 else shape[0]
        if num_classes == 0:
            return {'mIoU_bev': float('nan')}

        intersection = np.zeros(num_classes, dtype=np.float64)
        union = np.zeros(num_classes, dtype=np.float64)

        for idx, result in enumerate(results):
            pred = result.get('bev_seg')
            if pred is None:
                continue
            if torch.is_tensor(pred):
                pred = pred.detach().cpu().numpy()
            pred_arr = np.asarray(pred)
            if pred_arr.ndim == 4:
                pred_arr = pred_arr.argmax(axis=1)
            elif pred_arr.ndim == 3:
                if pred_arr.shape[0] == num_classes:
                    pred_arr = pred_arr.argmax(axis=0)
            pred_arr = np.squeeze(pred_arr)

            gt_ann = self.get_ann_info(idx)
            gt_mask = self._load_bev_mask(gt_ann.get('gt_bev_seg'))
            if gt_mask is None:
                continue
            if gt_mask.shape != pred_arr.shape:
                continue

            for cls_idx in range(num_classes):
                gt_cls = gt_mask == cls_idx
                pred_cls = pred_arr == cls_idx
                inter = np.logical_and(gt_cls, pred_cls).sum()
                union_cls = np.logical_or(gt_cls, pred_cls).sum()
                intersection[cls_idx] += inter
                union[cls_idx] += union_cls

        eps = 1e-12
        ious = intersection / np.maximum(union, eps)
        bev_results = {}
        for cls_idx in range(num_classes):
            cls_name = bev_class_names[cls_idx] if cls_idx < len(bev_class_names) else f'class_{cls_idx}'
            bev_results[f'IoU_bev_{cls_name}'] = float(ious[cls_idx])
        bev_results['mIoU_bev'] = float(np.mean(ious)) if intersection.sum() > 0 else float('nan')
        return bev_results

    @staticmethod
    def _load_bev_mask(mask):
        if mask is None:
            return None
        if isinstance(mask, str):
            if mask.endswith('.npy'):
                bev = np.load(mask)
            else:
                bev = mmcv.imread(mask, flag='unchanged')
        else:
            bev = np.asarray(mask)
        if bev.ndim > 2:
            bev = bev.squeeze()
        return bev

    @staticmethod
    def _bbox_iou_2d(box, boxes):
        """计算单个预测框与多个 GT 框的 IoU。"""
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        pred_area = (box[2] - box[0]) * (box[3] - box[1])
        gt_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = pred_area + gt_area - inter + 1e-12
        return inter / union

    @staticmethod
    def _compute_ap(recalls, precisions):
        """累积计算 P-R 曲线下的面积。"""
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        if indices.size == 0:
            return 0.0
        return float(np.sum((recalls[indices + 1] - recalls[indices]) *
                            precisions[indices + 1]))
