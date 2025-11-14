#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 SynWoodScape BEV 语义 GT 与预测结果，方便排查通道/标签映射问题。

使用流程（需先通过 tools/test.py --format-only 导出预测 pkl）：

1. 生成预测结果：
   python tools/test.py \
       configs/woodscape/fastbev_synwoodscape_multitask.py \
       work_dirs/syn_headABD_balanced/latest.pth \
       --eval bev_seg \
       --format-only \
       --out work_dirs/syn_headABD_balanced/bev_preds.pkl

2. 可视化对比（指定样本范围或数量）：
   python tools/visualize_synwoodscape_bev_compare.py \
       configs/woodscape/fastbev_synwoodscape_multitask.py \
       work_dirs/syn_headABD_balanced/bev_preds.pkl \
       --split val --num-samples 6 --start-index 0 \
       --out-dir work_dirs/bev_compare_vis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SynWoodScape BEV GT vs. predictions')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('predictions', help='tools/test.py --format-only 生成的 pkl')
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        default='val', help='可视化数据集 split')
    parser.add_argument('--num-samples', type=int, default=6,
                        help='可视化多少个样本')
    parser.add_argument('--start-index', type=int, default=0,
                        help='从哪个样本 index 开始')
    parser.add_argument('--indices', type=int, nargs='*',
                        help='显式指定若干样本索引，优先于 num-samples/start-index')
    parser.add_argument('--out-dir', default='work_dirs/bev_compare_vis',
                        help='保存目录')
    parser.add_argument('--dpi', type=int, default=140, help='图像 DPI')
    return parser.parse_args()


def build_dataset_from_cfg(cfg_path, split):
    cfg = Config.fromfile(cfg_path)
    data_cfg = cfg.data[split].copy()
    data_cfg['test_mode'] = False
    dataset = build_dataset(data_cfg)
    class_names = getattr(dataset, 'bev_seg_classes', None) or \
        cfg.get('bev_seg_classes') or \
        ('road_surface', 'lane_marking', 'sidewalk', 'vegetation', 'ground', 'background')
    return dataset, tuple(class_names)


def load_predictions(pred_path):
    preds = mmcv.load(pred_path)
    if not isinstance(preds, (list, tuple)):
        raise ValueError(f'预测文件 {pred_path} 不是 list/tuple，无法解析')
    return preds


def ensure_mask_array(mask):
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 0:
        return None
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] <= 6:  # (C,H,W)
            mask = mask.argmax(axis=0)
        elif mask.shape[-1] <= 6:
            mask = mask.argmax(axis=-1)
        else:
            mask = mask.squeeze()
    if mask.ndim == 1:
        side = int(np.sqrt(mask.size))
        if side * side == mask.size:
            mask = mask.reshape(side, side)
        else:
            raise ValueError(f'无法推断 mask 形状, ndim=1, size={mask.size}')
    if mask.ndim != 2:
        raise ValueError(f'期望 2D mask, 但得到形状 {mask.shape}')
    return mask


def decode_prediction(entry, num_classes):
    bev_pred = entry.get('bev_seg')
    if bev_pred is None:
        return None
    bev_pred = np.asarray(bev_pred)
    if bev_pred.ndim == 4:
        bev_pred = bev_pred[0]
    if bev_pred.ndim == 3:
        if bev_pred.shape[0] == num_classes:
            bev_mask = bev_pred.argmax(axis=0)
        elif bev_pred.shape[-1] == num_classes:
            bev_mask = bev_pred.argmax(axis=-1)
        else:
            # 已经是 mask
            bev_mask = bev_pred.squeeze()
    elif bev_pred.ndim == 2:
        bev_mask = bev_pred
    else:
        raise ValueError(f'无法解析预测维度 {bev_pred.shape}')
    return bev_mask.astype(np.int16)


def plot_hist(ax, values, class_names, title, color_fn):
    xs = np.arange(len(class_names))
    bars = ax.bar(xs, values, color=[color_fn(i) for i in xs])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontproperties='SimHei')
    ax.set_ylabel('占比 (%)', fontproperties='SimHei')
    ax.set_title(title, fontproperties='SimHei')
    ax.margins(y=0.15)


def load_gt_mask(dataset, ann):
    bev_value = ann.get('gt_bev_seg') or ann.get('bev_seg_path')
    if bev_value is None:
        return None
    if hasattr(dataset, '_load_bev_mask'):
        try:
            return dataset._load_bev_mask(bev_value)
        except Exception:
            pass
    return ensure_mask_array(bev_value)


def visualize_sample(idx, dataset, preds, class_names, out_dir, dpi=140):
    ann = dataset.get_ann_info(idx)
    gt_mask = load_gt_mask(dataset, ann)
    pred_mask = decode_prediction(preds[idx], len(class_names))
    if gt_mask is None:
        print(f'[Skip] 样本 {idx} 缺少 gt_bev_seg，跳过')
        return False
    if pred_mask is None:
        print(f'[Skip] 预测文件中样本 {idx} 缺少 bev_seg，跳过')
        return False

    if pred_mask.shape != gt_mask.shape:
        pred_mask = mmcv.imrescale(
            pred_mask.astype(np.float32),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation='nearest',
            backend='cv2').astype(np.int16)

    diff_mask = (pred_mask != gt_mask).astype(np.uint8)

    cmap = plt.cm.get_cmap('tab20', len(class_names))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    im0 = axes[0].imshow(gt_mask, cmap=cmap, vmin=0, vmax=len(class_names) - 1)
    axes[0].set_title(f'GT BEV 语义 (idx={idx})', fontproperties='SimHei')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=len(class_names) - 1)
    axes[1].set_title('预测 BEV 语义', fontproperties='SimHei')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(diff_mask, cmap='Reds')
    axes[2].set_title('错误区域 (红色)', fontproperties='SimHei')
    axes[2].axis('off')

    gt_counts = [(gt_mask == i).sum() for i in range(len(class_names))]
    pred_counts = [(pred_mask == i).sum() for i in range(len(class_names))]
    total_pixels = gt_mask.size
    gt_pct = [cnt / total_pixels * 100 for cnt in gt_counts]
    pred_pct = [cnt / total_pixels * 100 for cnt in pred_counts]

    plot_hist(axes[3], gt_pct, class_names, 'GT vs Pred 占比',
              lambda i: cmap(i))
    for i, (g, p) in enumerate(zip(gt_pct, pred_pct)):
        axes[3].text(i, max(g, p) + 0.5,
                     f'Pred: {p:.2f}%', rotation=90,
                     ha='center', va='bottom', fontsize=7, color='tab:red')

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out_path = Path(out_dir) / f'bev_compare_{idx:05d}.png'
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'[BEV Compare] saved sample {idx} -> {out_path}')
    return True


def main():
    args = parse_args()
    dataset, class_names = build_dataset_from_cfg(args.config, args.split)
    preds = load_predictions(args.predictions)

    total = len(dataset)
    if len(preds) != total:
        print(f'[Warn] 预测数量 {len(preds)} 与数据集 {total} 不一致，将按最小值截断')
        total = min(total, len(preds))
    indices = args.indices
    if indices is None or len(indices) == 0:
        max_count = min(args.num_samples, total - args.start_index)
        indices = list(range(args.start_index, args.start_index + max_count))
    saved = 0
    for idx in indices:
        if idx >= total:
            print(f'[Skip] idx {idx} 超出可视范围')
            continue
        ok = visualize_sample(idx, dataset, preds, class_names, args.out_dir, dpi=args.dpi)
        if ok:
            saved += 1
    if saved == 0:
        print('[Warn] 没有任何样本成功可视化，请确认预测/GT 是否齐全')


if __name__ == '__main__':
    main()
