#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查 Collect3D 之后的 img_metas 中是否保留了 mv_bboxes/mv_labels。

用法：
    python tools/inspect_mv_bboxes_after_collect.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py --index 0 --split train
"""

import argparse

from mmcv import Config
from mmcv.parallel import DataContainer
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def _shape_list(value):
    if value is None:
        return 'None'
    if isinstance(value, DataContainer):
        value = value.data
    shapes = []
    for view in value:
        view_data = view.data if isinstance(view, DataContainer) else view
        shape = getattr(view_data, 'shape', None)
        if shape is None and hasattr(view_data, 'numpy'):
            shape = view_data.numpy().shape
        shapes.append(shape)
    return str(shapes)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect mv_bboxes stored in img_metas after Collect3D')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--split',
                        choices=['train', 'val', 'test'],
                        default='train',
                        help='选择数据划分')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data[args.split])
    sample = dataset[args.index]

    img_metas = sample['img_metas']
    if isinstance(img_metas, DataContainer):
        img_metas = img_metas.data

    mv_bboxes = img_metas.get('mv_bboxes')
    mv_labels = img_metas.get('mv_labels')

    print(f"== sample {args.index} ({args.split}) ==")
    print('token:', img_metas.get('token', 'N/A'))
    print('mv_bboxes shapes:', _shape_list(mv_bboxes))
    print('mv_labels shapes:', _shape_list(mv_labels))

    if mv_bboxes is not None and len(mv_bboxes) > 0:
        first_view = mv_bboxes[0].data if isinstance(
            mv_bboxes[0], DataContainer) else mv_bboxes[0]
        print('first view bbox head (up to 5 rows):')
        print(first_view[:5])


if __name__ == '__main__':
    main()
