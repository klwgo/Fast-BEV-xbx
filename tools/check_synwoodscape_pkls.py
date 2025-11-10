#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量检查 SynWoodscape info.pkl 是否包含 2D 标注 (mv_bboxes/mv_labels)。

示例：
    python tools/check_synwoodscape_pkls.py data/synwoodscape_infos_train.pkl data/synwoodscape_infos_val.pkl
"""

import argparse
import mmcv
import numpy as np


def inspect_file(path, num_samples=5):
    data = mmcv.load(path)
    infos = data['infos']
    total = len(infos)
    has_mv = 0
    example_shapes = []
    for idx in range(min(num_samples, total)):
        ann = infos[idx].get('ann_info', {})
        mv = ann.get('mv_bboxes')
        mv_labels = ann.get('mv_labels')
        if mv and mv_labels:
            has_mv += 1
            shapes = [np.asarray(b).reshape(-1, 4).shape for b in mv]
            example_shapes.append(shapes)
        else:
            example_shapes.append('None')
    return total, has_mv, example_shapes


def parse_args():
    parser = argparse.ArgumentParser(description='Check SynWoodScape info for 2D annotations')
    parser.add_argument('files', nargs='+', help='需要检查的 pkl 路径')
    parser.add_argument('--num-samples', type=int, default=5, help='每个文件打印的样本数')
    return parser.parse_args()


def main():
    args = parse_args()
    for path in args.files:
        total, has_mv, shapes = inspect_file(path, args.num_samples)
        print(f'\n=== {path} ===')
        print(f'samples: {total}, samples with mv_bboxes: {has_mv}')
        for idx, shp in enumerate(shapes):
            print(f'  sample {idx}: {shp}')


if __name__ == '__main__':
    main()
