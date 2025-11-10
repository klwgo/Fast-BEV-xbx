#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速统计 SynWoodScape/ WoodScape info.pkl 中三维、二维、BEV 标注情况。

示例：
    python tools/analyze_synwoodscape_info.py data/synwoodscape_infos_val.pkl
"""

import argparse
import collections
from pathlib import Path

import mmcv
import numpy as np


DEFAULT_CLASSES = ['pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle']


def parse_args():
    parser = argparse.ArgumentParser(description='Quick stats for WoodScape/SynWoodScape infos')
    parser.add_argument('ann_file', help='路径指向 *.pkl info 文件')
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='打印空 GT 样本的最多条目数，默认 10')
    parser.add_argument(
        '--show-empty-only',
        action='store_true',
        help='仅打印空 GT 样本索引')
    return parser.parse_args()


def fmt_count_table(counter, total, title):
    if total == 0:
        return f'{title}: 无统计数据'
    lines = [f'{title}: (总数 {total})']
    for name, count in counter.items():
        ratio = 100.0 * count / total if total else 0.0
        lines.append(f'  - {name:<24} {count:>8} ({ratio:5.2f}%)')
    return '\n'.join(lines)


def main():
    args = parse_args()
    ann_path = Path(args.ann_file)
    if not ann_path.is_file():
        raise FileNotFoundError(f'找不到 info 文件: {ann_path}')

    data = mmcv.load(str(ann_path))
    infos = data.get('infos', [])
    metadata = data.get('metadata', {})
    instance_classes = (
        metadata.get('instance_classes')
        or metadata.get('bbox2d_classes')
        or metadata.get('classes')
        or DEFAULT_CLASSES
    )

    total_frames = len(infos)
    per_class_counter = collections.OrderedDict((cls, 0) for cls in instance_classes)
    total_3d_boxes = 0
    empty_gt_indices = []

    total_2d_boxes = 0
    frames_with_2d = 0

    bev_available = 0

    for idx, info in enumerate(infos):
        gt_names = info.get('gt_names', np.array([], dtype=object))
        if isinstance(gt_names, list):
            gt_names = np.asarray(gt_names, dtype=object)

        num_boxes = gt_names.shape[0]
        total_3d_boxes += num_boxes
        if num_boxes == 0:
            empty_gt_indices.append(idx)
        else:
            for name in gt_names:
                if name in per_class_counter:
                    per_class_counter[name] += 1
                else:
                    per_class_counter.setdefault(name, 0)
                    per_class_counter[name] += 1

        ann_info = info.get('ann_info', {})
        mv_labels = ann_info.get('mv_labels') or []
        per_frame_2d = sum(len(labels) for labels in mv_labels)
        total_2d_boxes += per_frame_2d
        if per_frame_2d > 0:
            frames_with_2d += 1

        if ann_info.get('gt_bev_seg') not in (None, '', []):
            bev_available += 1

    if args.show_empty_only:
        print(f'共 {total_frames} 帧，其中 {len(empty_gt_indices)} 帧无 3D GT：')
        print(empty_gt_indices[:args.topk])
        return

    print(f'Info 文件: {ann_path}')
    print(f'数据集版本: {metadata.get("version", "unknown")} | split: {metadata.get("split")}')
    print(f'总样本数: {total_frames}')
    print(f'含 3D 标注的样本: {total_frames - len(empty_gt_indices)} ({(1 - len(empty_gt_indices)/total_frames):.2%})')
    print(f'含 2D 标注的样本: {frames_with_2d} ({frames_with_2d / total_frames:.2%})')
    print(f'含 BEV mask 的样本: {bev_available} ({bev_available / total_frames:.2%})')
    print()
    print(fmt_count_table(per_class_counter, total_3d_boxes, '3D GT 统计'))
    print()
    print(f'二维框总数: {total_2d_boxes}')
    if empty_gt_indices:
        cap = args.topk if args.topk > 0 else len(empty_gt_indices)
        print(f'无 3D GT 样本索引（最多 {cap} 条）: {empty_gt_indices[:cap]}')


if __name__ == '__main__':
    main()
