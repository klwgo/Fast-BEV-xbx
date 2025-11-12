#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 Syn/WoodScape 样本中 3D 框中心在 BEV 栅格上的投影位置，
用于快速确认世界坐标 <-> BEV mask 之间是否存在整体偏移。

示例：
    python tools/check_bev_alignment.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val --max-samples 100 --dump-outliers 5
"""

import argparse
import math
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser('Check BEV/3D alignment statistics')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=50,
                        help='最多统计多少个样本（默认 50）')
    parser.add_argument('--dump-outliers', type=int, default=5,
                        help='打印最多 N 个越界示例')
    parser.add_argument('--require-bev', action='store_true',
                        help='若某样本缺少 BEV mask，则直接报错')
    return parser.parse_args()


def world_to_bev(points_xy, pc_range, bev_shape):
    h, w = bev_shape[:2]
    x_min, y_min, _, x_max, y_max, _ = pc_range
    denom_x = max(x_max - x_min, 1e-6)
    denom_y = max(y_max - y_min, 1e-6)
    px = (points_xy[:, 0] - x_min) * (w / denom_x)
    py = h - (points_xy[:, 1] - y_min) * (h / denom_y)
    return px, py


def clip_and_cast(px, py, w, h):
    px = np.clip(px.round().astype(int), 0, w - 1)
    py = np.clip(py.round().astype(int), 0, h - 1)
    return px, py


def summarize_stats(total_boxes, inside_pixels, outside_boxes):
    ratio = inside_pixels / max(total_boxes, 1)
    print(f'3D 框总数: {total_boxes}')
    print(f'落在 BEV 栅格内的坐标: {inside_pixels} ({ratio:.2%})')
    print(f'越界坐标: {outside_boxes} ({outside_boxes / max(total_boxes, 1):.2%})')


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    data_cfg = cfg.data[args.split].copy()
    data_cfg['test_mode'] = False
    dataset = build_dataset(data_cfg)

    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)

    total_boxes = 0
    inside_pixels = 0
    outside_boxes = 0
    outliers = []

    max_samples = args.max_samples if args.max_samples > 0 else math.inf
    for idx in range(min(len(dataset), max_samples)):
        ann = dataset.get_ann_info(idx)
        bev = ann.get('gt_bev_seg')
        if bev is None:
            bev_path = ann.get('bev_seg_path')
            if bev_path is not None:
                bev = bev_path
        if bev is None:
            if args.require_bev:
                raise RuntimeError(f'样本 {idx} 缺少 gt_bev_seg')
            continue
        if isinstance(bev, str):
            bev = np.load(bev) if bev.endswith('.npy') else mmcv.imread(bev, flag='unchanged')
        bev = np.asarray(bev).squeeze()
        h, w = bev.shape[:2]

        boxes = ann['gt_bboxes_3d'].tensor.cpu().numpy()
        centers = boxes[:, :2]
        if centers.size == 0:
            continue

        px, py = world_to_bev(centers, pc_range, (h, w))
        mask_inside = (
            (px >= 0) & (px < w) &
            (py >= 0) & (py < h)
        )
        inside_pixels += mask_inside.sum()
        outside_boxes += (~mask_inside).sum()
        total_boxes += centers.shape[0]

        if (~mask_inside).any() and len(outliers) < args.dump_outliers:
            px_c, py_c = clip_and_cast(px[~mask_inside], py[~mask_inside], w, h)
            outliers.append((idx, centers[~mask_inside], px_c, py_c))

    if total_boxes == 0:
        print('未找到任何 3D 框；请确认 ann_file 是否指向正确的 Syn/WoodScape pkl。')
        return

    summarize_stats(total_boxes, inside_pixels, outside_boxes)

    if outliers:
        print('\n越界示例（格式：样本 idx / 原始 XY / 映射像素坐标）:')
        for item in outliers:
            idx, centers_xy, px_c, py_c = item
            print(f'  - 样本 {idx}:')
            for xy, px_val, py_val in zip(centers_xy, px_c, py_c):
                print(f'      xy=({xy[0]:.2f}, {xy[1]:.2f}) -> clip(px={px_val}, py={py_val})')


if __name__ == '__main__':
    main()

