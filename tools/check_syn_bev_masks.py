#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 SynWoodScape BEV 掩码：尺寸、类别占比、是否为 one-hot 等。
"""

import argparse
from collections import Counter

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='检查 SynWoodScape BEV 掩码尺寸与类别分布')
    parser.add_argument('ann_file', help='synwoodscape_infos_*.pkl')
    parser.add_argument(
        '--samples', type=int, default=20,
        help='随机抽查样本数，默认 20')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='随机种子，默认 0')
    return parser.parse_args()


def summarize_mask(mask_path, class_names):
    """读取单个 BEV 掩码并统计形状、像素占比。"""
    arr = np.load(mask_path)
    if arr.ndim == 3 and arr.shape[-1] == len(class_names):
        arr = arr.argmax(axis=-1)
    unique, counts = np.unique(arr, return_counts=True)
    return arr.shape, dict(zip(unique.tolist(), counts.tolist()))


def main():
    args = parse_args()
    data = mmcv.load(args.ann_file)
    infos = data['infos']
    metadata = data.get('metadata', {})
    class_names = metadata.get('bev_seg', {}).get('class_names', [])

    print(f'总样本 {len(infos)}，BEV 类别: {class_names}')
    if not infos:
        return

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(infos), size=min(args.samples, len(infos)), replace=False)

    for idx in sorted(indices):
        ann = infos[idx].get('ann_info', {})
        mask_path = ann.get('gt_bev_seg')
        if mask_path is None:
            print(f'[{idx}] 无 gt_bev_seg 字段')
            continue
        shape, stats = summarize_mask(mask_path, class_names)
        total = sum(stats.values())
        if total == 0:
            ratios = {k: '0.0%' for k in stats}
        else:
            ratios = {}
            for cls_idx, count in stats.items():
                name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
                ratios[name] = f'{count / total:.3%}'
        print(f'[{idx}] {mask_path} shape={shape}, 类别占比={ratios}')


if __name__ == '__main__':
    main()
