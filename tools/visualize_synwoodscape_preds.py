#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize SynWoodScape predictions (3D & BEV) for sanity checking.

示例：
    python tools/visualize_synwoodscape_preds.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        work_dirs/synwoodscape_pretrain_anchor_ext_bs1_iou045/latest.pth \
        --split val --num-samples 5 --score-thr 0.05 \
        --out-dir work_dirs/debug_vis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmdet.datasets import build_dataset
from mmdet3d.models import build_model

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize BEV/3D predictions for SynWoodScape')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型 checkpoint 路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val',
                        help='使用哪个 data split 可视化')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='可视化样本数量')
    parser.add_argument('--start-index', type=int, default=0,
                        help='起始样本索引')
    parser.add_argument('--score-thr', type=float, default=0.05,
                        help='3D 框可视化阈值')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--out-dir', default='work_dirs/debug_vis',
                        help='结果保存目录')
    return parser.parse_args()


def build_model_from_cfg(cfg, checkpoint, device):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def prepare_dataset(cfg, split):
    dataset_cfg = cfg.data[split]
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg['test_mode'] = True
    dataset = build_dataset(dataset_cfg)
    return dataset


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


def plot_sample(idx, result, gt_ann, score_thr, out_dir):
    bev_pred = result.get('bev_seg')
    if bev_pred is None:
        bev_pred = np.zeros((1, 1))
    bev_pred = np.asarray(bev_pred)
    if bev_pred.ndim == 4:
        bev_pred = bev_pred[0]
    if bev_pred.ndim == 3:
        bev_pred_mask = bev_pred.argmax(axis=0)
    else:
        bev_pred_mask = bev_pred

    bev_gt = ensure_numpy_mask(gt_ann.get('gt_bev_seg'))
    if bev_gt is None:
        bev_gt = np.zeros_like(bev_pred_mask)

    boxes_3d = result['boxes_3d'].tensor.cpu().numpy()
    scores_3d = result['scores_3d'].cpu().numpy()
    labels_3d = result['labels_3d'].cpu().numpy()

    mask_keep = scores_3d >= score_thr
    boxes_3d = boxes_3d[mask_keep]
    scores_3d = scores_3d[mask_keep]
    labels_3d = labels_3d[mask_keep]

    gt_boxes_3d = gt_ann['gt_bboxes_3d'].tensor.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(bev_gt, cmap='tab20')
    axes[0].set_title('GT BEV 语义（真实标签）', fontproperties='SimHei')
    axes[1].imshow(bev_pred_mask, cmap='tab20')
    axes[1].set_title('Pred BEV 语义（模型预测）', fontproperties='SimHei')

    axes[2].set_title(f'3D 框中心分布 (score>{score_thr})', fontproperties='SimHei')
if gt_boxes_3d.shape[0] > 0:
    axes[2].scatter(gt_boxes_3d[:, 0], gt_boxes_3d[:, 1],
                    s=10, c='g', label='GT 框中心')
if boxes_3d.shape[0] > 0:
    axes[2].scatter(boxes_3d[:, 0], boxes_3d[:, 1],
                    s=10, c='r', label='预测框中心')
    axes[2].legend(prop={'family': 'SimHei'})
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / f'sample_{idx:05d}.png', dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    device = torch.device(args.device)
    model = build_model_from_cfg(cfg, args.checkpoint, device)
    dataset = prepare_dataset(cfg, args.split)

    total = min(args.num_samples, len(dataset) - args.start_index)
    for offset in range(total):
        idx = args.start_index + offset
        data = dataset[idx]
        data = collate([data], samples_per_gpu=1)
        if device.type == 'cuda':
            target_gpu = device.index if device.index is not None else torch.cuda.current_device()
            data = scatter(data, [target_gpu])[0]
        else:
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        result = result[0]
        gt_ann = dataset.get_ann_info(idx)
        plot_sample(idx, result, gt_ann, args.score_thr, args.out_dir)
        print(f'[Vis] saved sample {idx} to {args.out_dir}')


if __name__ == '__main__':
    main()
