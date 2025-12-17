#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SynWoodScape info 基础检查。

两种用法：
1) 批量检查是否包含 2D 标注：
   python tools/check_synwoodscape_pkls.py data/synwoodscape_infos_train.pkl data/synwoodscape_infos_val.pkl
2) 针对单个样本做详细校验（相机内参/畸变/BEV 掩码）：
   python tools/check_synwoodscape_pkls.py --ann-file data/synwoodscape_infos_train_small_fixed.pkl --idx 0
"""

import argparse
import os
from pathlib import Path

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


def inspect_detail(path: str, idx: int):
    data = mmcv.load(path)
    infos = data['infos']
    if idx < 0 or idx >= len(infos):
        raise IndexError(f'idx={idx} 超出范围，当前文件共有 {len(infos)} 条。')

    info = infos[idx]
    print(f'=== {path} | sample #{idx} ===')
    print(f"cams: {list(info['cams'].keys())}")
    for cam_name, cam in info['cams'].items():
        intr = np.asarray(cam.get('cam_intrinsic'))
        dist = np.asarray(cam.get('cam_distortion'))
        radial = cam.get('cam_radial_params')
        model = cam.get('cam_model')
        print(
            f'[{cam_name}] model={model}, intr_shape={intr.shape}, dist_shape={dist.shape}, '
            f'radial_keys={list(radial.keys()) if isinstance(radial, dict) else radial}'
        )
        if radial:
            vals = np.array(
                [radial.get(k, 0) for k in ['k1', 'k2', 'k3', 'k4', 'cx_offset', 'cy_offset', 'aspect_ratio']]
            )
            print(f'    radial stats: min={vals.min():.4f}, max={vals.max():.4f}')
        if np.allclose(dist, 0):
            print('    [WARN] 畸变系数全 0')

    ann = info.get('ann_info', {})
    bev_path = ann.get('gt_bev_seg')
    bev_exists = os.path.isfile(bev_path) if bev_path else False
    print(f"gt_bev_seg: {bev_path}, exists={bev_exists}")
    print(f"bev_seg_classes: {ann.get('bev_seg_classes')}")


def parse_args():
    """兼容两种模式：
    1) 批量模式：传入一个或多个文件路径，打印每个文件的 mv_bboxes 概况。
    2) 详细模式：指定 --ann-file 与 --idx，打印单条样本的相机/BEV 信息。
    """
    parser = argparse.ArgumentParser(description='Check SynWoodScape info for annotations / calibration')
    parser.add_argument('files', nargs='*', help='需要检查的 pkl 路径（批量模式）')
    parser.add_argument('--num-samples', type=int, default=5, help='批量模式下每个文件打印的样本数')
    parser.add_argument('--ann-file', help='单文件详细检查路径')
    parser.add_argument('--idx', type=int, default=0, help='详细检查的样本 index')
    return parser.parse_args()


def main():
    args = parse_args()
    # 优先详细模式
    if args.ann_file is not None:
        inspect_detail(args.ann_file, args.idx)
        return

    # 批量模式
    if not args.files:
        raise SystemExit("请提供 pkl 文件路径，或使用 --ann-file 进行单文件检查。")

    for path in args.files:
        total, has_mv, shapes = inspect_file(path, args.num_samples)
        print(f'\n=== {path} ===')
        print(f'samples: {total}, samples with mv_bboxes: {has_mv}')
        for idx, shp in enumerate(shapes):
            print(f'  sample {idx}: {shp}')


if __name__ == '__main__':
    main()
