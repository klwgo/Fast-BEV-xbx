#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspect per-frame GT boxes after the full mmdet3d pipeline to ensure they lie inside
`point_cloud_range` / anchors. Useful for debugging 3D loss == 0 issues.

Usage example:
    python tools/debug_woodscape_samples.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val --max-samples 50
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset
from mmcv.parallel import DataContainer

# 确保 WoodScapeMultiViewDataset 等自定义数据集完成注册
import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check WoodScape/SynWoodScape dataset samples after pipeline')
    parser.add_argument('config', help='Path to config file (e.g. fastbev_synwoodscape_pretrain.py)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                        help='Which split inside cfg.data to inspect')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Max number of samples to iterate (default: 100)')
    parser.add_argument('--show-first', action='store_true',
                        help='Print detailed stats for the first sample')
    parser.add_argument('--dump-outliers', type=int, default=5,
                        help='打印最多 N 个越界样本索引以及坐标')
    parser.add_argument('--check-2d', action='store_true',
                        help='额外统计 2D 标注数量，用于排查 bbox_head_2d')
    return parser.parse_args()


def _select_dataset_cfg(cfg, split):
    data_cfg = cfg.data[split]
    if isinstance(data_cfg, (list, tuple)):
        raise ValueError(f'{split} 数据配置是多 dataset 组合，暂不支持自动展开')
    data_cfg = data_cfg.copy()
    # mmcv build_dataset 里会读取 pipeline，保持原样即可
    return data_cfg


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset_cfg = _select_dataset_cfg(cfg, args.split)
    dataset = build_dataset(dataset_cfg)

    # Anchor 尺寸信息：用于评估 GT size 与 anchor 的匹配程度
    anchor_sizes = None
    try:
        anchor_cfg = cfg.model.get('bbox_head', {}).get('anchor_generator', {})
        sizes_cfg = anchor_cfg.get('sizes')
        if sizes_cfg:
            anchor_sizes = [np.asarray(size, dtype=np.float32) for size in sizes_cfg]
    except Exception:
        anchor_sizes = None
    if anchor_sizes:
        print(f'Anchor 尺寸（来自 config）：{anchor_sizes}')

    pc_range = np.array(cfg.get('point_cloud_range', [-math.inf, -math.inf, -math.inf,
                                                     math.inf, math.inf, math.inf]),
                        dtype=np.float32)
    mins = []
    maxs = []
    total_boxes = 0
    samples_with_boxes = 0
    inside = 0
    outside = 0
    outlier_records = []
    total_2d_boxes = 0
    samples_with_2d = 0
    per_class_sizes = defaultdict(list)
    anchor_ious = []

    max_iter = args.max_samples if args.max_samples > 0 else len(dataset)
    for idx in range(min(len(dataset), max_iter)):
        data = dataset[idx]
        boxes = data['gt_bboxes_3d'].data
        tensor = boxes.tensor.cpu().numpy()
        if tensor.shape[0] == 0:
            continue
        samples_with_boxes += 1
        centers = tensor[:, :3]
        sizes = tensor[:, 3:6]  # 假设顺序为 (dx, dy, dz)
        mins.append(centers.min(axis=0))
        maxs.append(centers.max(axis=0))
        mask_inside = (
            (centers[:, 0] >= pc_range[0]) & (centers[:, 0] <= pc_range[3]) &
            (centers[:, 1] >= pc_range[1]) & (centers[:, 1] <= pc_range[4]) &
            (centers[:, 2] >= pc_range[2]) & (centers[:, 2] <= pc_range[5])
        )
        inside += mask_inside.sum()
        outside_boxes = (~mask_inside).sum()
        outside += outside_boxes
        total_boxes += centers.shape[0]

        labels = data['gt_labels_3d'].data.cpu().numpy()
        for label, size in zip(labels, sizes):
            per_class_sizes[int(label)].append(size)

        if anchor_sizes is not None:
            # 统计每个 GT box 与 anchor 的最佳尺寸 IoU
            for size in sizes:
                ious = []
                for anchor_size in anchor_sizes:
                    ious.append(_size_iou(size, anchor_size))
                anchor_ious.append(max(ious))

        if outside_boxes > 0 and len(outlier_records) < args.dump_outliers:
            outlier_centers = centers[~mask_inside]
            outlier_records.append((idx, outlier_centers))

        if args.check_2d:
            gt_bboxes_2d = data.get('gt_bboxes')
            if isinstance(gt_bboxes_2d, DataContainer):
                gt_bboxes_2d = gt_bboxes_2d.data
            if gt_bboxes_2d is not None and len(gt_bboxes_2d) > 0:
                per_sample = 0
                for view in gt_bboxes_2d:
                    arr = view
                    if hasattr(arr, 'tensor'):
                        arr = arr.tensor
                    if hasattr(arr, 'cpu'):
                        arr = arr.cpu()
                    if hasattr(arr, 'numpy'):
                        arr = arr.numpy()
                    per_sample += arr.shape[0]
                total_2d_boxes += per_sample
                samples_with_2d += 1
                if args.show_first and samples_with_boxes == 1:
                    print(f'  2D 框数量: {per_sample} (视角 {len(gt_bboxes_2d)})')

        if args.show_first and samples_with_boxes == 1:
            print(f'样本 {idx}: 共有 {tensor.shape[0]} 个 3D 框')
            print('  中心坐标范围 (x/y/z): '
                  f'{centers[:,0].min():.2f}~{centers[:,0].max():.2f} / '
                  f'{centers[:,1].min():.2f}~{centers[:,1].max():.2f} / '
                  f'{centers[:,2].min():.2f}~{centers[:,2].max():.2f}')

    if not mins:
        print('❗ 未在前 {} 个样本中找到任何 3D 框'.format(max_iter))
        return

    global_min = np.stack(mins, axis=0).min(axis=0)
    global_max = np.stack(maxs, axis=0).max(axis=0)
    print(f'遍历样本数: {min(len(dataset), max_iter)} / {len(dataset)}')
    print(f'含 3D 框样本数: {samples_with_boxes}')
    print(f'3D 框总数: {total_boxes}')
    print('全局中心坐标范围 (x/y/z): '
          f'{global_min[0]:.2f}~{global_max[0]:.2f} / '
          f'{global_min[1]:.2f}~{global_max[1]:.2f} / '
          f'{global_min[2]:.2f}~{global_max[2]:.2f}')
    print('point_cloud_range: '
          f'[{pc_range[0]:.2f}, {pc_range[1]:.2f}, {pc_range[2]:.2f}, '
          f'{pc_range[3]:.2f}, {pc_range[4]:.2f}, {pc_range[5]:.2f}]')
    print(f'位于范围内部的框: {inside} ({inside / max(total_boxes, 1):.2%})')
    print(f'越界框: {outside} ({outside / max(total_boxes, 1):.2%})')
    if args.check_2d:
        print(f'含 2D 框样本数: {samples_with_2d}')
        print(f'2D 框总数: {total_2d_boxes}')

    if outlier_records:
        print('越界样本示例:')
        for idx, centers in outlier_records:
            print(f'  - 样本 {idx} 越界 {centers.shape[0]} 个：{centers}')

    if per_class_sizes:
        print('\n按类别统计 (w, l, h)：')
        class_names = getattr(dataset, 'CLASSES', None)
        for cls_id, sizes in per_class_sizes.items():
            arr = np.stack(sizes, axis=0)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            print(f'  {name}: mean={mean.round(3)}, std={std.round(3)}, count={len(sizes)}')

    if anchor_ious:
        anchor_ious = np.asarray(anchor_ious)
        print('\nAnchor 尺寸匹配统计：')
        print(f'  平均最佳 IoU: {anchor_ious.mean():.3f}')
        for thr in (0.3, 0.4, 0.5):
            ratio = (anchor_ious >= thr).mean()
            print(f'  IoU >= {thr:.1f} 的比例: {ratio:.2%}')


def _size_iou(size_a, size_b):
    """估算两个 3D 框尺寸之间的 IoU（忽略旋转、仅比较体积交并比）。"""
    size_a = np.asarray(size_a)
    size_b = np.asarray(size_b)
    inter = np.minimum(size_a, size_b).prod()
    union = np.maximum(size_a, size_b).prod()
    return inter / (union + 1e-6)


if __name__ == '__main__':
    main()
