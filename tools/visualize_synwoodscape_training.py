#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化/统计 SynWoodScape 训练样本（BEV 语义 + 2D 标注）。

示例：
    python tools/visualize_synwoodscape_training.py \
        configs/woodscape/fastbev_synwoodscape_multitask.py \
        --split train --num-samples 4 --start-index 0 \
        --cam-name CAM_FRONT --out-dir work_dirs/train_sample_vis
"""

import argparse
from pathlib import Path
from collections import Counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SynWoodScape training samples (BEV + 2D)')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        default='train', help='数据 split')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='输出多少个样本截图')
    parser.add_argument('--start-index', type=int, default=0,
                        help='从哪一个样本开始')
    parser.add_argument('--cam-name', default='CAM_FRONT',
                        help='可视化 2D 框的相机视角')
    parser.add_argument('--out-dir', default='work_dirs/train_sample_vis',
                        help='可视化输出目录')
    return parser.parse_args()


def build_split_dataset(cfg_path, split):
    cfg = Config.fromfile(cfg_path)
    if 'data' not in cfg or split not in cfg.data:
        raise KeyError(f'配置 {cfg_path} 中缺少 data.{split}')
    data_cfg = cfg.data[split].copy()
    data_cfg['test_mode'] = False
    dataset = build_dataset(data_cfg)
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    bev_classes = None
    for info in base_dataset.data_infos:
        ann = info.get('ann_info', {}) or {}
        bev_classes = ann.get('bev_seg_classes')
        if bev_classes:
            break
    if not bev_classes:
        bev_classes = list(cfg.get('bev_seg_classes', []))
    return dataset, base_dataset, tuple(bev_classes)


def load_bev_mask(bev_spec, data_root=None):
    if bev_spec is None:
        return None
    if isinstance(bev_spec, str):
        bev_path = Path(bev_spec)
        candidates = []
        if bev_path.is_absolute():
            candidates.append(bev_path)
        else:
            cwd = Path.cwd()
            candidates.append((cwd / bev_path).resolve())
            if data_root:
                data_root_path = Path(data_root)
                if not data_root_path.is_absolute():
                    data_root_path = (cwd / data_root_path).resolve()
                candidates.append((data_root_path / bev_path).resolve())
                candidates.append((data_root_path.parent / bev_path).resolve())
            # 兼容 info 中只写了 synwoodscape_bev_masks/xxx.npy 的情况
            candidates.append((cwd / 'data' / bev_path).resolve())
        unique_candidates = []
        seen = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(cand)
        resolved_path = None
        for cand in unique_candidates:
            if cand.exists():
                resolved_path = cand
                break
        if resolved_path is None:
            resolved_path = unique_candidates[0]
        if resolved_path.suffix == '.npy':
            mask = np.load(str(resolved_path))
        else:
            mask = mmcv.imread(str(resolved_path), flag='unchanged')
    else:
        mask = np.asarray(bev_spec)
    if mask.ndim > 2:
        mask = mask.squeeze()
    return mask


def summarize_dataset(dataset, base_dataset, bev_classes):
    total = len(base_dataset.data_infos)
    stats_2d = Counter()
    stats_samples = {'with_bev': 0, 'with_2d': 0}
    for info in base_dataset.data_infos:
        ann = info.get('ann_info', {}) or {}
        mv_bboxes = ann.get('mv_bboxes')
        mv_labels = ann.get('mv_labels')
        if mv_bboxes and mv_labels:
            total_boxes = sum(np.asarray(b).reshape(-1, 4).shape[0]
                              for b in mv_bboxes)
            if total_boxes > 0:
                stats_samples['with_2d'] += 1
                stats_2d['boxes'] += total_boxes
        if ann.get('gt_bev_seg') or ann.get('bev_seg_path'):
            stats_samples['with_bev'] += 1
    print('=========== SynWoodScape Dataset Summary ===========')
    print(f'Total samples: {total}')
    print(f'With BEV mask : {stats_samples["with_bev"]}/{total} '
          f'({stats_samples["with_bev"]/max(total,1):.1%})')
    print(f'With 2D boxes : {stats_samples["with_2d"]}/{total} '
          f'({stats_samples["with_2d"]/max(total,1):.1%})')
    if stats_2d['boxes']:
        avg_boxes = stats_2d['boxes'] / max(stats_samples['with_2d'], 1)
        print(f'Avg 2D boxes per annotated sample: {avg_boxes:.1f}')
    print(f'BEV classes: {", ".join(bev_classes) if bev_classes else "N/A"}')
    print('====================================================')


def build_bev_colormap(bev_classes):
    if not bev_classes:
        return None, None, tuple()
    base_cmap = plt.get_cmap('tab20', len(bev_classes))
    colors = base_cmap(np.arange(len(bev_classes)))
    cmap = ListedColormap(colors)
    boundaries = np.arange(len(bev_classes) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)
    legend_handles = tuple(
        patches.Patch(facecolor=colors[i], edgecolor='none', label=cls_name)
        for i, cls_name in enumerate(bev_classes)
    )
    return cmap, norm, legend_handles


def visualize_sample(dataset, base_dataset, idx, cam_name, bev_classes,
                     out_dir, bev_cmap=None, bev_norm=None, legend_handles=()):
    info = base_dataset.data_infos[idx]
    ann = base_dataset.get_ann_info(idx)
    bev_mask = load_bev_mask(ann.get('gt_bev_seg'), getattr(base_dataset, 'data_root', None))
    if bev_mask is None:
        bev_mask = np.zeros((1, 1))
    else:
        bev_mask = np.asarray(bev_mask)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(bev_mask, cmap=bev_cmap or 'tab20', norm=bev_norm)
    axes[0].set_title('BEV 语义', fontproperties='SimHei')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    if legend_handles:
        axes[0].legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            borderaxespad=0.0,
            ncol=2,
            fontsize=8,
            frameon=False)

    inset = axes[1]
    inset.set_title(f'{cam_name} 2D 框', fontproperties='SimHei')
    inset.axis('off')
    cam_idx = None
    camera_types = getattr(base_dataset, 'camera_types', getattr(dataset, 'camera_types', None))
    if camera_types and cam_name in camera_types:
        cam_idx = camera_types.index(cam_name)

    cam_meta = info.get('cams', {}).get(cam_name)
    if cam_meta and cam_meta.get('data_path'):
        img = mmcv.imread(cam_meta['data_path'])
        inset.imshow(img[:, :, ::-1])
    else:
        inset.text(0.5, 0.5, '缺少图像', ha='center', va='center',
                   color='red', fontsize=12, transform=inset.transAxes)
        img = None

    mv_bboxes = ann.get('mv_bboxes')
    mv_labels = ann.get('mv_labels')
    if img is not None and mv_bboxes and mv_labels and cam_idx is not None and cam_idx < len(mv_bboxes):
        boxes_2d = np.asarray(mv_bboxes[cam_idx]).reshape(-1, 4)
        for box in boxes_2d:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor='lime', facecolor='none')
            inset.add_patch(rect)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f'train_sample_{idx:05d}.png'
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f'[Train Sample] saved idx={idx} -> {fname}')


def main():
    args = parse_args()
    dataset, base_dataset, bev_classes = build_split_dataset(args.config, args.split)
    summarize_dataset(dataset, base_dataset, bev_classes)
    cmap, norm, legend_handles = build_bev_colormap(bev_classes)
    total = min(args.num_samples, len(base_dataset) - args.start_index)
    for offset in range(total):
        idx = args.start_index + offset
        visualize_sample(dataset, base_dataset, idx, args.cam_name, bev_classes,
                         args.out_dir, bev_cmap=cmap, bev_norm=norm,
                         legend_handles=legend_handles)


if __name__ == '__main__':
    main()
