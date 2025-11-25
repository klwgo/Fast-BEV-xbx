#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在数据增广（缩放/裁剪等）之后检查 BEV 多任务标签的占比，定位是否被裁剪/缩放成全背景。

用法示例：
    python tools/inspect_bev_targets_after_aug.py \
        --config configs/woodscape/fastbev_syn_headabc.py \
        --split train --limit 50
"""

import argparse
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmdet3d.datasets import build_dataset


def _to_numpy(x):
    if isinstance(x, DataContainer):
        x = x.data
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    return x


def analyze_sample(data, keys):
    stats = {}
    for key in keys:
        mask = _to_numpy(data.get(key))
        if mask is None:
            stats[key] = None
            continue
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[0] > 1:
            # 多通道 one-hot，转为 argmax
            mask = mask.argmax(axis=0)
        total = mask.size
        ratios = {}
        for cls_id, cnt in enumerate(np.bincount(mask.reshape(-1))):
            ratios[int(cls_id)] = cnt / total if total > 0 else 0.0
        stats[key] = dict(shape=mask.shape, ratios=ratios, nonzero=int(mask.any()))
    return stats


def main():
    parser = argparse.ArgumentParser(description='Inspect BEV targets after aug')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--limit', type=int, default=20, help='max samples to inspect')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.data.train if args.split == 'train' else cfg.data.val
    dataset = build_dataset(dataset_cfg)

    keys = ['gt_drivable_mask', 'gt_marking_mask', 'gt_obstacle_mask']
    counters = {k: [] for k in keys}

    for idx in range(min(args.limit, len(dataset))):
        data = dataset[idx]
        sample_stats = analyze_sample(data, keys)
        print(f'\nSample {idx}:')
        for k, v in sample_stats.items():
            if v is None:
                print(f'  {k}: missing')
                continue
            ratios_str = ', '.join([f'{cls}:{ratio:.6f}' for cls, ratio in sorted(v["ratios"].items())])
            print(f'  {k}: shape={v["shape"]}, nonzero={v["nonzero"]}, ratios[{ratios_str}]')
            counters[k].append(v)

    print('\nSummary:')
    for k in keys:
        vals = counters[k]
        if not vals:
            print(f'  {k}: no data')
            continue
        nonzero = sum(v['nonzero'] for v in vals)
        print(f'  {k}: nonzero {nonzero}/{len(vals)}')


if __name__ == '__main__':
    main()
