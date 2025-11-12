#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打印指定 SynWoodScape/WoodScape 样本的关键 GT 字段，核对
2D 框、3D 框、BEV mask 与点云坐标系 / 标定矩阵的一致性。

示例：
    python tools/verify_synwoodscape_alignment.py \
        configs/woodscape/fastbev_synwoodscape_pretrain.py \
        --split val --index 0 --dump-lidar2img
"""

import argparse
import copy
import os
from typing import Any, Dict, Iterable, Tuple

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset

# 确保 WoodScape 数据集完成注册
import mmdet3d.datasets.woodscape_dataset  # noqa: F401


np.set_printoptions(suppress=True, precision=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect single SynWoodScape sample for 2D/3D/BEV alignment')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val',
                        help='cfg.data 中需要解析的 split')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--topk', type=int, default=5,
                        help='最多打印前 K 个 3D 框（默认 5）')
    parser.add_argument('--dump-lidar2img', action='store_true',
                        help='打印 data_info["lidar2img"] 的外参/内参矩阵')
    parser.add_argument('--load-bev', action='store_true',
                        help='若 gt_bev_seg 是路径，是否实际加载并打印 shape')
    return parser.parse_args()


def build_split_dataset(cfg: Config, split: str):
    data_cfg = cfg.data[split]
    if isinstance(data_cfg, (list, tuple)):
        raise ValueError(f'{split} 是由多个 dataset 组成，暂不支持自动拆分')
    data_cfg = copy.deepcopy(data_cfg)
    data_cfg['test_mode'] = False
    return build_dataset(data_cfg)


def to_numpy(value: Any):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, 'tensor'):
        value = value.tensor
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if hasattr(value, 'cpu'):
        try:
            return value.cpu().numpy()
        except Exception:
            pass
    return np.asarray(value)


def load_bev_mask(value: Any, load_from_path: bool) -> Tuple[np.ndarray, str]:
    if value is None:
        return None, 'missing'
    if isinstance(value, str):
        desc = f'path={value}'
        if load_from_path and os.path.exists(value):
            if value.endswith('.npy'):
                arr = np.load(value)
            else:
                arr = mmcv.imread(value, flag='unchanged')
            return arr, desc
        return None, desc + (' (file not found)' if not os.path.exists(value) else '')
    return to_numpy(value), 'inline-array'


def print_lidar2img(info: Dict[str, Any], cam_names: Iterable[str]):
    lidar2img = info.get('lidar2img')
    if lidar2img is None:
        print('lidar2img: <missing>')
        return
    if not isinstance(lidar2img, dict):
        print('lidar2img:', to_numpy(lidar2img))
        return
    extrinsics = lidar2img.get('extrinsic')
    if extrinsics is None:
        extrinsics = lidar2img.get('extrinsics')
    intrinsics = lidar2img.get('intrinsic')
    if intrinsics is None:
        intrinsics = lidar2img.get('intrinsics')
    models = lidar2img.get('models')
    distortions = lidar2img.get('distortions')

    cam_names = list(cam_names)
    def _seq_len(value):
        if value is None:
            return 0
        return len(value)

    if extrinsics is not None and not isinstance(extrinsics, (list, tuple)):
        extrinsics = list(extrinsics)
    if intrinsics is not None and not isinstance(intrinsics, (list, tuple)):
        intrinsics = list(intrinsics)
    if models is not None and not isinstance(models, (list, tuple)):
        models = list(models)
    if distortions is not None and not isinstance(distortions, (list, tuple)):
        distortions = list(distortions)

    num_cams = max(len(cam_names), _seq_len(extrinsics), _seq_len(intrinsics))
    if num_cams == 0:
        num_cams = max(len(models or []), len(distortions or []), 1)
    if len(cam_names) < num_cams:
        cam_names = cam_names + [f'cam_{i}' for i in range(len(cam_names), num_cams)]

    def _pad_list(data):
        if data is None:
            return [None] * num_cams
        if not isinstance(data, list):
            data = list(data)
        if len(data) < num_cams:
            data += [None] * (num_cams - len(data))
        return data

    extr_list = _pad_list(extrinsics)
    intr_list = _pad_list(intrinsics)
    model_list = _pad_list(models)
    dist_list = _pad_list(distortions)

    print('lidar2img matrices:')
    for idx in range(num_cams):
        cam = cam_names[idx]
        ext = extr_list[idx]
        intr = intr_list[idx]
        model = model_list[idx]
        dist = dist_list[idx]
        print(f'  [{idx}] {cam}:')
        if ext is not None:
            print('    extrinsic:\n', np.asarray(ext))
        if intr is not None:
            print('    intrinsic:\n', np.asarray(intr))
        if dist is not None:
            print('    distortion:', np.asarray(dist))
        if model is not None:
            print('    model:', model)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_split_dataset(cfg, args.split)
    idx = args.index
    info = dataset.get_data_info(idx)
    ann = dataset.get_ann_info(idx)

    print(f'=== Sample {idx} ===')
    pc_range_cfg = np.array(cfg.get('point_cloud_range', []), dtype=np.float32)
    if pc_range_cfg.size:
        print('point_cloud_range (from cfg):', pc_range_cfg)
    data_info_range = info.get('point_cloud_range')
    if data_info_range is not None:
        print('point_cloud_range (from info):', data_info_range)
    print('available cameras:', list(info.get('cams', {}).keys()))

    boxes = ann.get('gt_bboxes_3d')
    if boxes is not None and hasattr(boxes, 'tensor'):
        box_tensor = boxes.tensor.detach().cpu().numpy()
        print(f'3D boxes shape: {box_tensor.shape}')
        if box_tensor.shape[0]:
            limit = min(args.topk, box_tensor.shape[0])
            print(f'First {limit} boxes (x, y, z, dx, dy, dz, yaw, ...):')
            print(box_tensor[:limit])
            centers = box_tensor[:, :2]
            print('XY range min/max:',
                  centers.min(axis=0).round(3), centers.max(axis=0).round(3))
        else:
            print('3D boxes: <empty>')
    else:
        print('gt_bboxes_3d missing or invalid')

    labels = ann.get('gt_labels_3d')
    if labels is not None:
        labels_np = to_numpy(labels)
        limit = min(args.topk, labels_np.shape[0]) if labels_np.size else 0
        print(f'gt_labels_3d shape: {labels_np.shape}')
        if limit:
            print('First labels:', labels_np[:limit])
    else:
        print('gt_labels_3d missing')

    mv_bboxes = ann.get('mv_bboxes')
    mv_labels = ann.get('mv_labels')
    if mv_bboxes is not None and mv_labels is not None:
        print(f'Multi-view 2D 标注: {len(mv_bboxes)} views')
        first_view = mv_bboxes[0]
        first_view_np = np.asarray(first_view).reshape(-1, 4)
        limit = min(args.topk, first_view_np.shape[0])
        print(f'  View[0] first {limit} boxes:', first_view_np[:limit])
    else:
        print('mv_bboxes/mv_labels missing')

    bev_value = ann.get('gt_bev_seg') or ann.get('bev_seg_path')
    if bev_value is None:
        print('gt_bev_seg: <missing>')
    else:
        bev_mask, desc = load_bev_mask(bev_value, args.load_bev)
        if bev_mask is not None:
            print(f'gt_bev_seg ({desc}) shape: {bev_mask.shape}, dtype={bev_mask.dtype}')
        else:
            print(f'gt_bev_seg reference ({desc}), shape未加载 (使用 --load-bev 可读取)')

    if args.dump_lidar2img:
        print_lidar2img(info, cam_names=info.get('cams', {}).keys())


if __name__ == '__main__':
    main()
