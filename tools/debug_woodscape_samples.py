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

    max_iter = args.max_samples if args.max_samples > 0 else len(dataset)
    for idx in range(min(len(dataset), max_iter)):
        data = dataset[idx]
        boxes = data['gt_bboxes_3d'].data
        tensor = boxes.tensor.cpu().numpy()
        if tensor.shape[0] == 0:
            continue
        samples_with_boxes += 1
        centers = tensor[:, :3]
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


if __name__ == '__main__':
    main()
