#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查 Fast-BEV SynWoodScape 数据管线中的 2D/3D 监督是否完整。

示例：
    python tools/check_synwoodscape_2d.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split train --num-samples 5
"""

import argparse

from mmcv import Config
from mmdet.datasets import build_dataset
from mmcv.parallel import DataContainer

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect per-sample 2D/3D annotations after pipeline')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        default='train', help='data split to inspect')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='检查样本数，默认 5')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data[args.split])
    max_samples = min(args.num_samples, len(dataset))

    print(f'Inspecting split={args.split}, samples={max_samples}/{len(dataset)}')
    for idx in range(max_samples):
        sample = dataset[idx]
        ann = dataset.data_infos[idx].get('ann_info', {})
        print(f'\n=== Sample {idx} ===')
        # 3D
        gt3d = sample['gt_bboxes_3d'].data.tensor
        print('3D boxes:', gt3d.shape)
        mv_ann = ann.get('mv_bboxes')
        if mv_ann:
            print('ann_info mv_bboxes lens:',
                  [arr.shape for arr in mv_ann[:len(mv_ann)]])
        else:
            print('ann_info mv_bboxes missing or empty')
        # 2D from gt_bboxes container
        gt2d = sample['gt_bboxes']
        if isinstance(gt2d, DataContainer):
            gt2d = gt2d.data
        print('2D container views:', len(gt2d))
        for cam_idx, arr in enumerate(gt2d):
            print(f'  cam{cam_idx}: {arr.shape}')
        # meta mv_* (用于 fastbev fallback)
        meta = sample['img_metas'].data
        mv = meta.get('mv_bboxes')
        if isinstance(mv, DataContainer):
            mv = mv.data
        if mv is not None:
            print('meta mv_bboxes views:', len(mv))
            for cam_idx, arr in enumerate(mv):
                shape = getattr(arr, 'shape', None)
                if shape is None and hasattr(arr, 'numpy'):
                    shape = arr.numpy().shape
                print(f'  cam{cam_idx}: {shape}')
        else:
            print('meta mv_bboxes missing')


if __name__ == '__main__':
    main()
