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
            info['valid_flag'] = np.asarray(info.get('valid_flag', []), dtype=bool)

            processed.append(info)

        return processed

    def get_data_info(self, index):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'\[Fast-BEV] Expected 6 cameras, got [0-9]+\. '
                        r'Continue with adaptive visualization\.',
                category=UserWarning)
            return super().get_data_info(index)

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
            cam['cam_intrinsic'] = self._to_float_array(cam['cam_intrinsic'], shape=(3, 3))
            cam['cam_distortion'] = self._to_float_array(
                cam.get('cam_distortion', np.zeros(4)), shape=(-1,))

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
                 **kwargs):
        """Evaluate detection results with a simple IoU-based mAP.

        Args:
            results (list[dict]): 推理输出，要求包含 ``boxes_3d``、
                ``scores_3d``、``labels_3d``。
            metric (str | Sequence[str], optional): 当前仅支持 ``'mAP'``。
            logger: 记录器。
            iou_thr (float): IoU 阈值，默认 0.5。
            score_thr (float): 过滤预测框的得分阈值，默认 0（不过滤）。

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

        def _compute_ap(recalls, precisions):
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            indices = np.where(recalls[1:] != recalls[:-1])[0]
            if indices.size == 0:
                return 0.0
            return float(
                np.sum(
                    (recalls[indices + 1] - recalls[indices]) *
                    precisions[indices + 1]))

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
                # ious shape: [num_gt, num_pred]
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

        eval_results = {}
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
            ap = _compute_ap(recalls, precisions)
            eval_results[f'AP_{cls_name}@{iou_thr}'] = ap
            aps.append(ap)

        if aps:
            eval_results[f'mAP@{iou_thr}'] = float(np.mean(aps))
        else:
            eval_results[f'mAP@{iou_thr}'] = float('nan')

        if logger is not None:
            logger.info('WoodScape evaluation (IoU {:.2f}): {}'.format(
                iou_thr, eval_results))

        return eval_results
