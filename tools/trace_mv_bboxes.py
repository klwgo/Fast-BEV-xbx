#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐步追踪 SynWoodScape pipeline 中 mv_bboxes/mv_labels 的变化，定位 2D 数据在哪个阶段丢失。

用法：
    python tools/trace_mv_bboxes.py configs/woodscape/fastbev_synwoodscape_pretrain.py --index 0
"""

import argparse
import numpy as np

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from mmcv.parallel import DataContainer

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def _format_views(data):
    if data is None:
        return 'None'
    if isinstance(data, DataContainer):
        data = data.data
    lens = []
    for view in data:
        shape = getattr(view, 'shape', None)
        if shape is None and hasattr(view, 'numpy'):
            shape = view.numpy().shape
        lens.append(shape)
    return str(lens)


def parse_args():
    parser = argparse.ArgumentParser(description='Trace mv_bboxes through pipeline')
    parser.add_argument('config', help='配置文件')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        default='train', help='选择 data split')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data[args.split])

    raw_ann = mmcv.load(cfg.data[args.split].ann_file)
    raw_info = raw_ann['infos'][args.index]['ann_info']
    raw_shapes = []
    for b in raw_info.get('mv_bboxes', []):
        arr = np.asarray(b, dtype=np.float32).reshape(-1, 4)
        raw_shapes.append(arr.shape)
    print(f'raw info mv_bboxes lens: {raw_shapes}')

    input_dict = dataset.get_data_info(args.index)
    dataset.pre_pipeline(input_dict)
    data = input_dict

    print(f'== trace sample {args.index} ==')
    print('pre_pipeline ann mv shapes:',
          _format_views(data.get('ann_info', {}).get('mv_bboxes')))

    for idx, t in enumerate(dataset.pipeline.transforms):
        data = t(data)
        if data is None:
            print(f'transform {idx} ({t}) 返回 None')
            break
        mv = data.get('mv_bboxes')
        mv_labels = data.get('mv_labels')
        ann_mv = data.get('ann_info', {}).get('mv_bboxes') if 'ann_info' in data else None
        print(f'[{idx}] {t.__class__.__name__}: mv={_format_views(mv)} ann_mv={_format_views(ann_mv)}')

    if data is not None:
        mv_meta = data.get('img_metas')
        if mv_meta is not None:
            meta = mv_meta.data if hasattr(mv_meta, 'data') else mv_meta
            mv_inner = meta.get('mv_bboxes')
            if mv_inner is not None:
                print('img_metas mv_bboxes:', _format_views(
                    mv_inner.data if hasattr(mv_inner, 'data') else mv_inner))


if __name__ == '__main__':
    main()
