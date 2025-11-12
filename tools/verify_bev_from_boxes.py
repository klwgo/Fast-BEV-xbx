#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 3D 框 footprint 重新 rasterize BEV mask，与训练用的 gt_bev_seg 对比，
用于验证 GT 是否有效。

示例：
    python tools/verify_bev_from_boxes.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val --index 0 --out work_dirs/bev_verify
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser('Verify BEV GT by rasterizing 3D boxes')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--out', type=Path, default=Path('work_dirs/bev_verify'),
                        help='输出目录')
    parser.add_argument('--show', action='store_true', help='是否弹出窗口')
    return parser.parse_args()


def ensure_mask(value):
    if isinstance(value, str):
        if value.endswith('.npy'):
            return np.load(value)
        return plt.imread(value)
    mask = np.asarray(value)
    if mask.ndim > 2:
        if mask.shape[0] <= 6:
            mask = mask.max(axis=0)
        else:
            mask = mask[..., 0]
    return mask


def get_bottom_corners(boxes):
    if hasattr(boxes, 'bottom_corners'):
        pts = boxes.bottom_corners
    elif hasattr(boxes, 'corners'):
        pts = boxes.corners[:, :4, :]
    else:
        raise ValueError('gt_bboxes_3d 不支持 corners 接口')
    pts = pts[:, :, :2]
    return pts.cpu().numpy()


def world_to_pixel(xy, pc_range, bev_shape):
    h, w = bev_shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    scale_x = w / max(x_max - x_min, 1e-6)
    scale_y = h / max(y_max - y_min, 1e-6)
    px = (xy[:, 0] - x_min) * scale_x
    py = h - (xy[:, 1] - y_min) * scale_y
    return px, py


def rasterize_polygons(polygons, pc_range, bev_shape):
    h, w = bev_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        px, py = world_to_pixel(poly, pc_range, bev_shape)
        poly_pix = np.stack([px, py], axis=1)
        x_min = max(int(np.floor(poly_pix[:, 0].min())), 0)
        x_max = min(int(np.ceil(poly_pix[:, 0].max())) + 1, w)
        y_min = max(int(np.floor(poly_pix[:, 1].min())), 0)
        y_max = min(int(np.ceil(poly_pix[:, 1].max())) + 1, h)
        if x_min >= x_max or y_min >= y_max:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(x_min, x_max),
            np.arange(y_min, y_max))
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        path = MplPath(poly_pix)
        inside = path.contains_points(points)
        inside = inside.reshape((y_max - y_min), (x_max - x_min))
        mask[y_min:y_max, x_min:x_max][inside] = 1
    return mask


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    data_cfg = cfg.data[args.split].copy()
    data_cfg['test_mode'] = False
    dataset = build_dataset(data_cfg)
    idx = max(0, min(len(dataset) - 1, args.index))
    ann = dataset.get_ann_info(idx)
    boxes = ann['gt_bboxes_3d']
    bev = ann.get('gt_bev_seg')
    if bev is None:
        raise RuntimeError(f'样本 {idx} 缺少 gt_bev_seg')
    bev_mask = ensure_mask(bev).squeeze()
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)

    corners = get_bottom_corners(boxes)
    polygons = corners
    raster = rasterize_polygons(polygons, pc_range, bev_mask.shape[:2])
    total_boxes = polygons.shape[0]
    x_min, y_min, _, x_max, y_max, _ = pc_range
    centers = ann['gt_bboxes_3d'].tensor[:, :2].cpu().numpy()
    inside_mask = (
        (centers[:, 0] >= x_min) & (centers[:, 0] <= x_max) &
        (centers[:, 1] >= y_min) & (centers[:, 1] <= y_max)
    )
    print(f'[Info] Sample {idx}: boxes={total_boxes}, inside_range={inside_mask.mean():.2%}')

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    extent = [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]

    axes[0].imshow(bev_mask, origin='lower', extent=extent, cmap='tab20')
    axes[0].set_title('gt_bev_seg')
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])

    axes[1].imshow(raster, origin='lower', extent=extent, cmap='gray')
    axes[1].set_title('Rasterized 3D footprint')
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])

    axes[2].imshow(bev_mask, origin='lower', extent=extent, cmap='gray', alpha=0.5)
    for poly in polygons:
        axes[2].plot(poly[:, 0], poly[:, 1], 'r-', linewidth=0.8)
    axes[2].set_title('gt + 3D footprints')
    axes[2].set_xlim(extent[0], extent[1])
    axes[2].set_ylim(extent[2], extent[3])

    for ax in axes:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

    fig.tight_layout()
    out_file = args.out / f'bev_verify_{idx:05d}.png'
    fig.savefig(out_file, dpi=150)
    print(f'[Info] 保存对比图到 {out_file}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
