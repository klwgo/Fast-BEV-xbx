# -*- coding: utf-8 -*-
import numpy as np
import mmcv

from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class PrepareSynWoodscape2DTargets:
    """将 ann_info 中的多视角 2D 框整理为固定长度的 per-view 列表。

    SynWoodScape 的 info 中，`mv_bboxes` / `mv_labels` 以列表存储，
    但其长度、dtype、缺失视角并不稳定。该模块会：
        1. 根据 img_info 长度对齐视角数并补齐缺失项；
        2. 将每个视角的 bbox/label 转成统一 dtype 的 numpy 数组；
        3. 同步更新 `ann_info['bboxes']` / `['labels']`，便于后续
           LoadAnnotations3D 直接读取多视角 2D 框。
    同时保留/补全 BEV mask 字段，避免下游流程缺键。
    """

    def __init__(self, bev_mask_shape=(1024, 1024)):
        self.bev_mask_shape = tuple(bev_mask_shape)

    def __call__(self, results):
        ann = results.get('ann_info', {})
        mv_bboxes = results.get('mv_bboxes', ann.get('mv_bboxes'))
        mv_labels = results.get('mv_labels', ann.get('mv_labels'))

        num_img_infos = len(results.get('img_info', []))
        num_mv_boxes = len(mv_bboxes) if isinstance(mv_bboxes, (list, tuple)) else 0
        num_mv_labels = len(mv_labels) if isinstance(mv_labels, (list, tuple)) else 0
        target_views = max(num_img_infos, num_mv_boxes, num_mv_labels, 1)

        def _empty_boxes():
            return np.zeros((0, 4), dtype=np.float32)

        def _empty_labels():
            return np.zeros((0,), dtype=np.int64)

        if mv_bboxes is None or mv_labels is None:
            per_view_bboxes = [_empty_boxes() for _ in range(target_views)]
            per_view_labels = [_empty_labels() for _ in range(target_views)]
        else:
            per_view_bboxes, per_view_labels = [], []
            for view_idx in range(target_views):
                boxes = mv_bboxes[view_idx] if view_idx < len(mv_bboxes) else None
                labels = mv_labels[view_idx] if view_idx < len(mv_labels) else None

                if boxes is None:
                    boxes = _empty_boxes()
                else:
                    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

                if labels is None:
                    labels = _empty_labels()
                else:
                    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

                per_view_bboxes.append(boxes)
                per_view_labels.append(labels)

        if per_view_bboxes:
            flat_bboxes = np.concatenate(per_view_bboxes, axis=0)
            flat_labels = np.concatenate(per_view_labels, axis=0)
        else:
            flat_bboxes = np.zeros((0, 4), dtype=np.float32)
            flat_labels = np.zeros((0,), dtype=np.int64)
        ann['bboxes'] = flat_bboxes
        ann['labels'] = flat_labels
        ann['mv_bboxes'] = [boxes.copy() for boxes in per_view_bboxes]
        ann['mv_labels'] = [labels.copy() for labels in per_view_labels]

        bev_seg = ann.get('gt_bev_seg')
        if bev_seg is None or bev_seg == '':
            ann['gt_bev_seg'] = np.zeros(self.bev_mask_shape, dtype=np.uint8)
            ann.setdefault('bev_mask_shape', list(self.bev_mask_shape))

        results['ann_info'] = ann
        results['mv_bboxes'] = per_view_bboxes
        results['mv_labels'] = per_view_labels
        if 'pad_shape' not in results:
            img_shape = results.get('img_shape')
            if img_shape is not None:
                results['pad_shape'] = tuple(img_shape)
            else:
                ori_shape = results.get('ori_shape')
                if ori_shape is not None:
                    results['pad_shape'] = tuple(ori_shape)
                else:
                    h = w = results['img'][0].shape[-2:]
                    results['pad_shape'] = (h, w, 3)
        return results


@PIPELINES.register_module()
class EnsureSynWoodscape3DTargets:
    """保证 ann_info 中存在 3D 框 / 标签，即使原始 info 缺失也补空数组."""

    def __init__(self, box_dim=9):
        self.box_dim = box_dim

    def __call__(self, results):
        ann = results.setdefault('ann_info', {})
        gt_boxes = ann.get('gt_bboxes_3d')
        if gt_boxes is None:
            ann['gt_bboxes_3d'] = np.zeros((0, self.box_dim), dtype=np.float32)
        gt_labels = ann.get('gt_labels_3d')
        if gt_labels is None:
            ann['gt_labels_3d'] = np.zeros((0,), dtype=np.int64)
        results['ann_info'] = ann
        return results


@PIPELINES.register_module()
class LoadSynWoodscapeBEVSeg:
    """加载 BEV 语义掩码，允许 ann_info 仅存储路径."""

    def __call__(self, results):
        ann = results.get('ann_info', {})
        bev_value = ann.get('gt_bev_seg')
        if bev_value is None:
            return results
        if isinstance(bev_value, str):
            if bev_value.endswith('.npy'):
                bev_mask = np.load(bev_value)
            else:
                bev_mask = mmcv.imread(bev_value, flag='unchanged')
            if bev_mask.ndim == 2:
                bev_mask = bev_mask[np.newaxis, ...]
            results['gt_bev_seg'] = bev_mask
        else:
            bev_mask = np.asarray(bev_value)
            if bev_mask.ndim == 2:
                bev_mask = bev_mask[np.newaxis, ...]
            results['gt_bev_seg'] = bev_mask
        return results
