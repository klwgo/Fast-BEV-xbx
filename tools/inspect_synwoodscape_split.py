#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查 SynWoodScape 数据划分是否包含 2D / 3D / BEV 监督。

示例：
    python tools/inspect_synwoodscape_split.py \
        --ann-file data/synwoodscape_infos_full.pkl

    python tools/inspect_synwoodscape_split.py \
        --config configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val
"""

import argparse
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config


def _infer_ann_file(cfg_path, split):
    cfg = Config.fromfile(cfg_path)
    if 'data' not in cfg or split not in cfg.data:
        raise KeyError(f'配置 {cfg_path} 中缺少 data.{split}')
    ann_file = cfg.data[split].get('ann_file')
    if ann_file is None:
        raise KeyError(f'config.data.{split}.ann_file 未设置')
    return ann_file


def _load_infos(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'未找到 ann_file: {path}')
    data = mmcv.load(str(path))
    if isinstance(data, dict) and 'infos' in data:
        return data['infos'], data.get('metadata', {})
    raise ValueError(f'{path} 内容无 infos 字段，无法解析')


def summarize_infos(infos):
    total = len(infos)
    stats = dict(has_2d=0, has_3d=0, has_bev=0)
    examples = dict(two_d=None, three_d=None, bev=None)

    for info in infos:
        ann = info.get('ann_info', {})
        mv_bboxes = ann.get('mv_bboxes')
        mv_labels = ann.get('mv_labels')
        if mv_bboxes is not None and mv_labels is not None:
            if any(np.asarray(box).size > 0 for box in mv_bboxes):
                stats['has_2d'] += 1
                if examples['two_d'] is None:
                    examples['two_d'] = {
                        'num_views': len(mv_bboxes),
                        'per_view_shapes': [np.asarray(b).reshape(-1, 4).shape for b in mv_bboxes]
                    }

        gt_boxes = info.get('gt_boxes')
        gt_names = info.get('gt_names')
        if gt_boxes is not None and np.asarray(gt_boxes).size > 0:
            stats['has_3d'] += 1
            if examples['three_d'] is None:
                examples['three_d'] = {
                    'num_boxes': int(np.asarray(gt_boxes).shape[0]),
                    'sample_names': gt_names[:4].tolist() if isinstance(gt_names, np.ndarray) else gt_names
                }

        bev = ann.get('gt_bev_seg') or ann.get('bev_seg_path')
        if bev is not None:
            stats['has_bev'] += 1
            if examples['bev'] is None:
                examples['bev'] = str(bev)[:120]

    summary = {
        'total': total,
        'ratio_2d': stats['has_2d'] / total if total else 0.0,
        'ratio_3d': stats['has_3d'] / total if total else 0.0,
        'ratio_bev': stats['has_bev'] / total if total else 0.0,
        'examples': examples,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description='Inspect SynWoodScape split supervision')
    parser.add_argument('--ann-file', help='指定 infos pkl 路径')
    parser.add_argument('--config', help='可选：从配置里读取 ann_file')
    parser.add_argument('--split', default='val', help='数据 split，配合 --config 使用')
    args = parser.parse_args()

    if not args.ann_file:
        if not args.config:
            parser.error('必须提供 --ann-file 或 (--config 与 --split)')
        ann_file = _infer_ann_file(args.config, args.split)
    else:
        ann_file = args.ann_file

    infos, metadata = _load_infos(ann_file)
    summary = summarize_infos(infos)
    print(f'Loaded {len(infos)} samples from {ann_file}')
    if metadata:
        print(f"Metadata: {metadata}")
    print('2D annotations: {}/{} ({:.1%})'.format(
        summary['ratio_2d'] * len(infos), len(infos), summary['ratio_2d']))
    print('3D annotations: {}/{} ({:.1%})'.format(
        summary['ratio_3d'] * len(infos), len(infos), summary['ratio_3d']))
    print('BEV masks: {}/{} ({:.1%})'.format(
        summary['ratio_bev'] * len(infos), len(infos), summary['ratio_bev']))
    print('Examples:')
    for key, value in summary['examples'].items():
        print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
