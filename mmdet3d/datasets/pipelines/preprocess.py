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


@PIPELINES.register_module()
class GenerateBEVMultitaskTargets:
    """将单通道 BEV 语义 ID 映射为多任务训练所需的标签.

    该模块会依据 `bev_seg_classes` 中的类别名称, 自动生成：
    1) `gt_drivable_mask`: 可行驶区域三类标签 (0-不可行驶, 1-可行驶, 2-不确定).
    2) `gt_marking_mask`: 细分道路标线类别 (以 background=0 开始的多类标签).
    3) `gt_marking_boundary`: 标线边界的二值图 (用于辅助监督).
    4) `gt_slot_masks`: 停车位相关二值图, 第 0 通道为车位线, 第 1 通道为端点/入口.

    以上标签完全由现有 `gt_bev_seg` 推导, 不额外依赖原始 PNG,
    便于同时支持 WoodScape 与 SynWoodScape。
    """

    def __init__(self,
                 drivable_positive=None,
                 drivable_ambiguous=None,
                 marking_classes=None,
                 slot_line_classes=None,
                 slot_endpoint_classes=None,
                 slot_endpoint_max_neighbors=2,
                 ignore_index=255,
                 drivable_class_names=('non_drivable', 'drivable', 'uncertain'),
                 slot_channel_names=('slot_line', 'slot_endpoint'),
                 obstacle_classes=None,
                 point_cloud_range=(-50.0, -50.0, -5.0, 50.0, 50.0, 3.0),
                 with_slot=True,
                 with_obstacle=False,
                 with_occlusion=False,
                 occlusion_from_drivable=True):
        # ----------- 可行驶区域配置 -----------
        self.drivable_positive = drivable_positive or [
            'road_surface', 'free_space', 'lane_marking', 'parking_line'
        ]
        self.drivable_ambiguous = drivable_ambiguous or [
            'other_ground_marking', 'zebra_crossing'
        ]
        # ----------- 标线与停车位配置 -----------
        self.marking_classes = marking_classes or [
            'lane_marking', 'parking_line', 'other_ground_marking',
            'zebra_crossing'
        ]
        self.slot_line_classes = slot_line_classes or ['parking_line']
        self.slot_endpoint_classes = slot_endpoint_classes or ['zebra_crossing']
        self.slot_endpoint_max_neighbors = slot_endpoint_max_neighbors
        self.ignore_index = ignore_index
        self.drivable_class_names = list(drivable_class_names or (
            'non_drivable', 'drivable', 'uncertain'))
        if len(self.drivable_class_names) != 3:
            raise ValueError('drivable_class_names 必须包含 3 个条目')
        self.slot_channel_names = list(slot_channel_names or (
            'slot_line', 'slot_endpoint'))
        self.slot_channels = len(self.slot_channel_names)
        if self.slot_channels == 0 and with_slot:
            raise ValueError('slot_channel_names 至少包含一个通道')
        self.obstacle_class_names = list(obstacle_classes or [])
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        if self.point_cloud_range.shape[0] != 6:
            raise ValueError('point_cloud_range 需为长度 6 的序列')
        self.with_slot = with_slot
        self.with_obstacle = with_obstacle and len(self.obstacle_class_names) > 0
        self.with_occlusion = with_occlusion
        self.occlusion_from_drivable = occlusion_from_drivable
        self.num_obstacle_classes = len(self.obstacle_class_names) + 1 if self.with_obstacle else 0
        self.enable_obstacle = self.with_obstacle
        self.enable_occlusion = self.with_occlusion

    def __call__(self, results):
        bev_mask = results.get('gt_bev_seg')
        if bev_mask is None:
            return results

        ann = results.get('ann_info', {})
        class_names = ann.get('bev_seg_classes', [])

        targets = self.build_targets(results, bev_mask, class_names)
        results.update(targets)
        return results

    def build_targets(self, results, bev_mask, class_meta):
        """直接返回由 BEV 语义掩码推导的多任务标签."""
        if isinstance(class_meta, dict):
            class_to_id = class_meta
        else:
            class_to_id = {name: idx for idx, name in enumerate(class_meta or [])}
        label_map = self._squeeze_bev_mask(bev_mask)
        drivable = self._build_drivable_map(label_map, class_to_id)
        marking, boundary = self._build_marking_map(label_map, class_to_id)
        targets = dict(
            gt_drivable_mask=drivable,
            gt_marking_mask=marking,
            gt_marking_boundary=boundary,
        )
        if self.with_slot:
            slot_masks = self._build_slot_map(label_map, class_to_id)
            targets['gt_slot_masks'] = slot_masks
        if self.with_obstacle:
            ann_info = results.get('ann_info', {})
            bev_shape = self._infer_bev_shape(ann_info, label_map.shape)
            obstacle_mask = self._build_obstacle_map(ann_info, bev_shape)
            targets['gt_obstacle_mask'] = obstacle_mask
        if self.with_occlusion:
            ann_info = results.get('ann_info', {})
            occlusion_mask = self._build_occlusion_map(
                ann_info, drivable, label_map.shape)
            targets['gt_occlusion_mask'] = occlusion_mask
        return targets

    @staticmethod
    def _squeeze_bev_mask(bev_mask):
        """将任意形状的 BEV 掩码压缩成 [H, W] 的类别 ID."""
        arr = np.asarray(bev_mask)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[0] <= 32 and arr.max() <= 1:
                # (C, H, W) one-hot -> argmax
                arr = arr.argmax(axis=0)
            elif arr.shape[-1] <= 32 and arr.max() <= 1:
                # (H, W, C) one-hot
                arr = arr.argmax(axis=-1)
            else:
                # 默认取第 0 通道
                arr = arr[0]
        return arr.astype(np.int64, copy=False)

    @staticmethod
    def _collect_ids(class_to_id, names):
        ids = []
        for name in names:
            idx = class_to_id.get(name)
            if idx is not None:
                ids.append(idx)
        return ids

    def _build_drivable_map(self, label_map, class_to_id):
        mask = np.zeros_like(label_map, dtype=np.int64)
        pos_ids = self._collect_ids(class_to_id, self.drivable_positive)
        amb_ids = self._collect_ids(class_to_id, self.drivable_ambiguous)

        if pos_ids:
            mask[np.isin(label_map, pos_ids)] = 1
        if amb_ids:
            mask[np.isin(label_map, amb_ids)] = 2
        return mask.astype(np.int64, copy=False)

    def _build_marking_map(self, label_map, class_to_id):
        marking = np.zeros_like(label_map, dtype=np.int64)
        for cls_idx, name in enumerate(self.marking_classes, start=1):
            ids = self._collect_ids(class_to_id, [name])
            if not ids:
                continue
            marking[np.isin(label_map, ids)] = cls_idx

        binary = (marking > 0).astype(np.uint8)
        boundary = np.zeros_like(binary, dtype=np.uint8)

        # 4 邻域差分提取边缘，避免引入 OpenCV 依赖。
        boundary[:-1, :] |= binary[:-1, :] != binary[1:, :]
        boundary[1:, :] |= binary[1:, :] != binary[:-1, :]
        boundary[:, :-1] |= binary[:, :-1] != binary[:, 1:]
        boundary[:, 1:] |= binary[:, 1:] != binary[:, :-1]
        boundary = (boundary > 0) & (binary > 0)

        return marking.astype(np.int64, copy=False), boundary.astype(np.uint8, copy=False)

    def _build_slot_map(self, label_map, class_to_id):
        line_ids = self._collect_ids(class_to_id, self.slot_line_classes)
        endpoint_ids = self._collect_ids(class_to_id, self.slot_endpoint_classes)

        line_mask = np.zeros_like(label_map, dtype=np.uint8)
        if line_ids:
            line_mask[np.isin(label_map, line_ids)] = 1

        endpoint_mask = np.zeros_like(line_mask, dtype=np.uint8)
        if line_mask.any():
            endpoint_mask = self._extract_endpoints(line_mask)
        if endpoint_ids:
            endpoint_mask = np.logical_or(
                endpoint_mask, np.isin(label_map, endpoint_ids)).astype(np.uint8)

        slot = np.zeros((max(self.slot_channels, 1),) + line_mask.shape, dtype=np.uint8)
        if self.slot_channels > 0:
            slot[0] = line_mask
        if self.slot_channels > 1:
            slot[1] = endpoint_mask
        return slot

    def _build_obstacle_map(self, ann_info, spatial_shape):
        boxes = ann_info.get('gt_bboxes_3d')
        labels = ann_info.get('gt_labels_3d')
        if boxes is None or labels is None or len(boxes) == 0:
            return np.zeros(spatial_shape, dtype=np.int64)
        mask = np.zeros(spatial_shape, dtype=np.int64)
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        height, width = spatial_shape
        x_scale = width / max(x_max - x_min, 1e-6)
        y_scale = height / max(y_max - y_min, 1e-6)
        for box, label in zip(boxes, labels):
            cls_id = int(label) + 1
            if cls_id <= 0 or cls_id >= self.num_obstacle_classes:
                cls_id = min(max(cls_id, 1), self.num_obstacle_classes - 1)
            cx, cy = float(box[0]), float(box[1])
            dx = abs(float(box[3])) if len(box) > 3 else 0.0
            dy = abs(float(box[4])) if len(box) > 4 else 0.0
            x1 = max(min(cx - dx / 2, x_max), x_min)
            x2 = max(min(cx + dx / 2, x_max), x_min)
            y1 = max(min(cy - dy / 2, y_max), y_min)
            y2 = max(min(cy + dy / 2, y_max), y_min)
            if x2 <= x1 or y2 <= y1:
                continue
            c0 = int(np.floor((x1 - x_min) * x_scale))
            c1 = int(np.ceil((x2 - x_min) * x_scale))
            r0 = int(np.floor((y1 - y_min) * y_scale))
            r1 = int(np.ceil((y2 - y_min) * y_scale))
            c0 = np.clip(c0, 0, width - 1)
            c1 = np.clip(c1, 1, width)
            r0 = np.clip(r0, 0, height - 1)
            r1 = np.clip(r1, 1, height)
            if c0 >= c1 or r0 >= r1:
                continue
            mask[r0:r1, c0:c1] = cls_id
        return mask

    def _build_occlusion_map(self, ann_info, drivable_mask, spatial_shape):
        occ_value = ann_info.get('occlusion_mask')
        if occ_value is not None:
            occ = np.asarray(occ_value)
            if occ.shape != spatial_shape:
                occ = mmcv.imresize(
                    occ.astype(np.uint8),
                    (spatial_shape[1], spatial_shape[0]),
                    interpolation='nearest')
            return occ.astype(np.float32)
        if self.occlusion_from_drivable:
            return (drivable_mask == 2).astype(np.float32)
        return np.zeros(spatial_shape, dtype=np.float32)

    @staticmethod
    def _infer_bev_shape(ann_info, fallback_shape):
        bev_shape = ann_info.get('bev_mask_shape')
        if isinstance(bev_shape, (list, tuple)) and len(bev_shape) >= 2:
            return int(bev_shape[0]), int(bev_shape[1])
        return fallback_shape

    def _extract_endpoints(self, line_mask):
        """通过邻域统计寻找端点/入口."""
        padded = np.pad(line_mask, 1, mode='constant', constant_values=0)
        neighbor = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:] +
            padded[:-2, :-2] + padded[:-2, 2:] +
            padded[2:, :-2] + padded[2:, 2:]
        )
        endpoints = (line_mask > 0) & (neighbor <= self.slot_endpoint_max_neighbors)
        return endpoints.astype(np.uint8, copy=False)
