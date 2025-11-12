#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 GT 伪装成预测，验证 dataset.evaluate 逻辑是否正常。

示例：
    python tools/dummy_eval_with_gt.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val
"""

import argparse

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval GT as dummy predictions to check metrics')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--max-samples', type=int, default=-1,
                        help='限制评估样本数，-1 为全部')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    data_cfg = cfg.data[args.split].copy()
    data_cfg['test_mode'] = True
    dataset = build_dataset(data_cfg)

    results = []
    total = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    for idx in range(total):
        ann = dataset.get_ann_info(idx)
        gt_boxes = ann['gt_bboxes_3d']
        if not isinstance(gt_boxes, LiDARInstance3DBoxes):
            gt_boxes = LiDARInstance3DBoxes(gt_boxes.tensor)
        res = dict(
            boxes_3d=gt_boxes,
            scores_3d=torch.ones(len(gt_boxes)),
            labels_3d=torch.from_numpy(ann['gt_labels_3d'])
        )
        mv_bboxes = ann.get('mv_bboxes') or []
        mv_labels = ann.get('mv_labels') or []
        mv_scores = [np.ones(len(b)) for b in mv_bboxes]
        res['mv_bboxes'] = mv_bboxes
        res['mv_labels'] = mv_labels
        res['mv_scores'] = mv_scores

        bev = ann.get('gt_bev_seg')
        if bev is not None:
            if isinstance(bev, str):
                bev = np.load(bev) if bev.endswith('.npy') else mmcv.imread(bev, flag='unchanged')
            res['bev_seg'] = bev
        results.append(res)

    eval_kwargs = dict(eval_3d=True, eval_2d=True, eval_bev=True, score_thr=0.0)
    metrics = dataset.evaluate(results, **eval_kwargs)
    print('Dummy evaluation using GT as prediction:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
