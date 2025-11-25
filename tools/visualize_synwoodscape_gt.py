#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 Syn/WoodScape 数据集单帧 GT（BEV 语义 + 3D 框中心）。

示例：
    python tools/visualize_synwoodscape_gt.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val --num-samples 5 \
        --start-index 0 --out-dir work_dirs/gt_vis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import sitecustomize  # noqa: F401
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SynWoodScape GT (BEV seg + 3D centers)')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        default='val', help='data split')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='可视化多少个样本')
    parser.add_argument('--start-index', type=int, default=0,
                        help='从哪个样本 index 开始')
    parser.add_argument('--out-dir', default='work_dirs/gt_vis',
                        help='结果保存目录')
    parser.add_argument('--cam-name', default='CAM_FRONT',
                        help='需要可视化 2D 框的相机名称')
    return parser.parse_args()


def load_dataset(cfg_path, split):
    cfg = Config.fromfile(cfg_path)
    data_cfg = cfg.data[split].copy()
    data_cfg['test_mode'] = False
    dataset = build_dataset(data_cfg)
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)
    return dataset, pc_range


def ensure_numpy_mask(mask):
    if mask is None:
        return None
    if isinstance(mask, str):
        if mask.endswith('.npy'):
            mask = np.load(mask)
        else:
            mask = mmcv.imread(mask, flag='unchanged')
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    return mask


def get_bev_footprints(boxes):
    """从 3D 框对象中提取底面四边形，返回 [N, 4, 2]"""
    corners = None
    if hasattr(boxes, 'corners'):
        corners = boxes.corners
    elif hasattr(boxes, 'bottom_corners'):
        corners = boxes.bottom_corners
    if corners is None:
        return None
    if hasattr(corners, 'cpu'):
        corners = corners.cpu().numpy()
    corners = np.asarray(corners)
    if corners.ndim != 3 or corners.shape[1] < 4:
        return None
    # LiDARInstance3DBoxes.corners 返回顺序默认 0~3 为底面
    return corners[:, [0, 1, 2, 3], :2]


def visualize_sample(dataset, pc_range, idx, cam_name, out_dir):
    data_info = dataset.data_infos[idx]
    ann = dataset.get_ann_info(idx)

    bev = ann.get('gt_bev_seg')
    bev = ensure_numpy_mask(bev)
    if bev is None:
        bev = np.zeros((1, 1))

    boxes_obj = ann['gt_bboxes_3d']
    boxes_np = boxes_obj.tensor.cpu().numpy()
    centers = boxes_np[:, :2]
    footprints = get_bev_footprints(boxes_obj)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x_min, y_min, _, x_max, y_max, _ = pc_range
    extent = [x_min, x_max, y_min, y_max]

    axes[0].imshow(bev, cmap='tab20', origin='lower', extent=extent, aspect='auto')
    axes[0].set_title('GT BEV 语义', fontproperties='SimHei')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(bev, cmap='gray', alpha=0.3, origin='lower', extent=extent, aspect='auto')
    axes[1].set_title('BEV + 3D 框 footprint', fontproperties='SimHei')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset.CLASSES)))
    if footprints is not None:
        for footprint, label in zip(footprints, ann['gt_labels_3d']):
            color = colors[int(label) % len(colors)]
            poly = patches.Polygon(
                footprint,
                closed=True,
                linewidth=1,
                edgecolor=color,
                facecolor='none')
            axes[1].add_patch(poly)
    else:
        # fallback：使用长宽 + yaw 绘制矩形
        for box, label in zip(boxes_np, ann['gt_labels_3d']):
            cx, cy = box[0], box[1]
            dx, dy = box[3], box[4]
            yaw = box[6] if box.shape[0] > 6 else 0.0
            color = colors[int(label) % len(colors)]
            rect = patches.Rectangle((cx - dx / 2, cy - dy / 2),
                                     dx, dy,
                                     linewidth=1,
                                     edgecolor=color,
                                     facecolor='none')
            rotation = transforms.Affine2D().rotate_deg_around(
                cx, cy, np.degrees(yaw)) + axes[1].transData
            rect.set_transform(rotation)
            axes[1].add_patch(rect)

    axes[2].scatter(centers[:, 0], centers[:, 1], c='g', s=10)
    axes[2].set_title('GT 3D 框 XY 中心', fontproperties='SimHei')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)
    axes[2].grid(True, linestyle='--', alpha=0.3)

    mv_bboxes = ann.get('mv_bboxes')
    mv_labels = ann.get('mv_labels')
    cam_img = None
    if mv_bboxes and cam_name in data_info['cams']:
        cam_meta = data_info['cams'][cam_name]
        img_path = cam_meta['data_path']
        cam_img = mmcv.imread(img_path)
        axes_img = axes[0].inset_axes([0.65, 0.05, 0.32, 0.32])
        axes_img.imshow(cam_img[:, :, ::-1])
        axes_img.set_title(f'{cam_name} 2D 框', fontsize=8, fontproperties='SimHei')
        axes_img.axis('off')
        cam_idx = dataset.camera_types.index(cam_name)
        if cam_idx < len(mv_bboxes):
            boxes_2d = mv_bboxes[cam_idx]
            labels_2d = mv_labels[cam_idx]
            for box2d, label2d in zip(boxes_2d, labels_2d):
                x1, y1, x2, y2 = box2d
                rect2d = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=1,
                                           edgecolor='lime',
                                           facecolor='none')
                axes_img.add_patch(rect2d)
        else:
            axes_img.text(0.5, 0.5, '无 2D 框', ha='center', va='center',
                          color='red', transform=axes_img.transAxes)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(out_dir) / f'gt_sample_{idx:05d}.png'
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    print(f'[GT Vis] saved sample {idx} -> {fname}')


def main():
    args = parse_args()
    dataset, pc_range = load_dataset(args.config, args.split)
    total = min(args.num_samples, len(dataset) - args.start_index)
    for offset in range(total):
        idx = args.start_index + offset
        visualize_sample(dataset, pc_range, idx, args.cam_name, args.out_dir)


if __name__ == '__main__':
    main()
