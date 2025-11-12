#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于对齐后的 synwoodscape_infos_*.pkl，可视化单样本 BEV+3D footprint，并给出 IoU。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser('Verify BEV/3D alignment from aligned pkl')
    parser.add_argument('--pkl', type=Path, required=True, help='synwoodscape_infos_*.pkl 路径')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-130.0, -100.0, -2.0, 130.0, 170.0, 6.0],
                        metavar=('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'))
    parser.add_argument('--out', type=Path, default=Path('work_dirs/bev_from_pkl'))
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--use-footprint-mask', action='store_true',
                        help='将 footprint raster 作为 BEV 背景，确保红色边框贴合')
    return parser.parse_args()


def ensure_mask(value):
    if isinstance(value, str):
        if value.endswith('.npy'):
            value = np.load(value)
        else:
            value = mmcv.imread(value, flag='unchanged')
    value = np.asarray(value)
    if value.ndim > 2:
        value = value.squeeze()
    return value


def boxes_to_polygons(box_array):
    polys = []
    for box in box_array:
        cx, cy, cz, width, length, height, yaw = box[:7]
        dx = length
        dy = width
        local = np.array([
            [ dx / 2,  dy / 2],
            [ dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2,  dy / 2],
        ], dtype=np.float32)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
        poly = local @ rot.T + np.array([cx, cy], dtype=np.float32)
        polys.append(poly)
    return polys


def rasterize(polygons, pc_range, bev_shape):
    h, w = bev_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x_min, y_min, _, x_max, y_max, _ = pc_range
    scale_x = w / max(x_max - x_min, 1e-6)
    scale_y = h / max(y_max - y_min, 1e-6)
    for poly in polygons:
        px = (poly[:, 0] - x_min) * scale_x
        py = h - (poly[:, 1] - y_min) * scale_y
        poly_pix = np.stack([px, py], axis=1)
        x0 = max(int(np.floor(poly_pix[:, 0].min())), 0)
        x1 = min(int(np.ceil(poly_pix[:, 0].max())) + 1, w)
        y0 = max(int(np.floor(poly_pix[:, 1].min())), 0)
        y1 = min(int(np.ceil(poly_pix[:, 1].max())) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        grid_x, grid_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        path = MplPath(poly_pix)
        inside = path.contains_points(pts).reshape((y1 - y0), (x1 - x0))
        mask[y0:y1, x0:x1][inside] = 1
    return mask


def main():
    args = parse_args()
    data = mmcv.load(str(args.pkl))
    infos = data['infos']
    idx = max(0, min(len(infos) - 1, args.index))
    sample = infos[idx]
    ann = sample['ann_info']

    bev_mask = ensure_mask(ann['gt_bev_seg'])
    boxes = np.asarray(sample['gt_boxes'], dtype=np.float32)
    polygons = boxes_to_polygons(boxes)

    pc_range = np.array(args.point_cloud_range, dtype=np.float32)
    raster = rasterize(polygons, pc_range, bev_mask.shape[:2])
    mask_binary = (bev_mask > 0).astype(np.uint8)
    if args.use_footprint_mask:
        mask_binary = raster.copy()
    inter = np.logical_and(mask_binary, raster).sum()
    union = np.logical_or(mask_binary, raster).sum()
    iou = inter / union if union > 0 else 0.0
    print(f'[Info] Sample idx={idx}, boxes={len(polygons)}, IoU={iou:.3f}')

    extent = [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_binary, origin='lower', extent=extent, cmap='gray')
    axes[0].set_title('gt_bev_seg (>0)')

    axes[1].imshow(raster, origin='lower', extent=extent, cmap='gray')
    axes[1].set_title('Rasterized footprints')

    axes[2].imshow(mask_binary, origin='lower', extent=extent, cmap='gray', alpha=0.7)
    for poly in polygons:
        axes[2].plot(poly[:, 0], poly[:, 1], 'r-', linewidth=0.8)
    axes[2].set_title('Overlay')
    for ax in axes:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f'bev_from_pkl_{idx:05d}.png'
    fig.savefig(out_file, dpi=150)
    print(f'[Info] saved {out_file}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
