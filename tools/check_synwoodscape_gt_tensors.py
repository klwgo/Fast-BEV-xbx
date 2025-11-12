#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印指定样本的 3D/2D/BEV GT 数值，确认坐标、尺寸是否与 BEV 对齐。

示例：
    python tools/check_synwoodscape_gt_tensors.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val --index 0
"""

import argparse
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser('Inspect single sample GT values')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    data_cfg = cfg.data[args.split].copy()
    data_cfg['test_mode'] = False
    dataset = build_dataset(data_cfg)

    idx = args.index
    data_info = dataset.data_infos[idx]
    ann = dataset.get_ann_info(idx)

    print(f'=== Sample {idx} ===')
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)
    print('point_cloud_range:', pc_range)
    print('cams:', list(data_info['cams'].keys()))

    boxes = ann['gt_bboxes_3d'].tensor.cpu().numpy()
    labels = ann['gt_labels_3d']
    print(f'3D boxes count: {boxes.shape[0]}')
    print('First 5 boxes (x, y, z, dx, dy, dz, yaw, ...):')
    print(boxes[:5])
    print('First 5 labels:', labels[:5])
    centers = boxes[:, :2]
    print('XY min/max:', centers.min(axis=0), centers.max(axis=0))

    mv_bboxes = ann.get('mv_bboxes')
    mv_labels = ann.get('mv_labels')
    if mv_bboxes and mv_labels:
        print(f'2D views: {len(mv_bboxes)}')
        first_view = mv_bboxes[0]
        print('First view 2D boxes:', np.asarray(first_view)[:5])

    bev = ann.get('gt_bev_seg')
    if bev is not None:
        if isinstance(bev, str):
            if bev.endswith('.npy'):
                bev_arr = np.load(bev)
            else:
                bev_arr = mmcv.imread(bev, flag='unchanged')
        else:
            bev_arr = np.asarray(bev)
        print('BEV seg shape:', bev_arr.shape)
    else:
        print('No BEV seg.')


if __name__ == '__main__':
    main()
