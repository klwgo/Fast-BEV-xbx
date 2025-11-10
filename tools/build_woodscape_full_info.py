#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 WoodScape 基础 info：
- 读取多视角图像、实例多边形、2D 框路径
- 为后续 convert_woodscape_full.py 所需做准备

说明：
- 该脚本假设实例 JSON 中的 segmentation 为 BEV 平面坐标；
  若实际坐标系不同，需根据数据格式调整转换逻辑。
- 目前生成的 3D 框字段仅保留在 ann_info['gt_polygons'] 内，
  Fast-BEV 3D 检测仍依赖额外转换。可在生成后按需要填充 gt_boxes。
"""
import argparse
import json
import os
from pathlib import Path

import mmcv
import numpy as np

CAMERA_ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
CAM_SUFFIX = {
    'CAM_FRONT': 'FV',
    'CAM_FRONT_LEFT': 'MVL',
    'CAM_FRONT_RIGHT': 'MVR',
    'CAM_BACK': 'RV'
}


def parse_args():
    parser = argparse.ArgumentParser(description='构建 WoodScape 基础 info')
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='WoodScape_ICCV19 根目录')
    parser.add_argument('--rgb-root', type=Path, required=True,
                        help='RGB 图像根目录（例如 woodscape_extracted/rgb_images）')
    parser.add_argument('--output', type=Path, required=True,
                        help='输出 info.pkl 路径')
    parser.add_argument('--split-file', type=Path, default=None,
                        help='可选，列出需要处理的 token 列表（txt，一行一个）')
    parser.add_argument('--split', type=str, default='train',
                        help='写入 metadata 的 split 名称')
    parser.add_argument('--camera-types', nargs='+', default=CAMERA_ORDER,
                        help='相机视角顺序')
    return parser.parse_args()


def load_target_tokens(split_file: Path):
    if split_file is None or not split_file.exists():
        return None
    tokens = []
    with split_file.open() as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)
    return set(tokens)


def build_single_info(token: str,
                      raw_root: Path,
                      rgb_root: Path,
                      camera_types):
    """构造单帧 info."""
    cams = {}
    ann_info = dict(gt_polygons=dict(), mv_bboxes=[], mv_labels=[])

    for cam in camera_types:
        suffix = CAM_SUFFIX.get(cam, cam)
        img_name = f'{token}_{suffix}.png'
        img_path = rgb_root / img_name
        cams[cam] = dict(
            data_path=str(img_path),
            height=720,
            width=1280,
            timestamp=0,
            sensor2ego_rotation=[0, 0, 0, 1],
            sensor2ego_translation=[0, 0, 0],
            ego2global_rotation=[0, 0, 0, 1],
            ego2global_translation=[0, 0, 0],
            sensor2lidar_rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            sensor2lidar_translation=[0, 0, 0],
        )

        bbox_txt = raw_root / 'box_2d_annotations' / 'box_2d_annotations' / f'{token}_{suffix}.txt'
        bboxes = []
        labels = []
        if bbox_txt.exists():
            with bbox_txt.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    cls_name = parts[0]
                    x1, y1, x2, y2 = map(float, parts[2:6])
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(cls_name)
        ann_info['mv_bboxes'].append(np.asarray(bboxes, dtype=np.float32))
        ann_info['mv_labels'].append(np.asarray(labels, dtype=object))

    inst_path = raw_root / 'instance_annotations' / 'instance_annotations' / f'{token}_FV.json'
    if inst_path.exists():
        data = json.loads(inst_path.read_text())
        record = data.get(next(iter(data)))
        polygons = dict()
        for ann in record.get('annotation', []):
            tags = ann.get('tags', [])
            seg = ann.get('segmentation', [])
            if not tags or not seg:
                continue
            cls = tags[0]
            polygons.setdefault(cls, []).append(seg)
        ann_info['gt_polygons'] = polygons

    info = dict(
        token=token,
        scene_name='woodscape_scene',
        frame_id=int(token),
        timestamp=0,
        cams=cams,
        ann_info=ann_info,
        gt_boxes=np.zeros((0, 9), dtype=np.float32),
        gt_names=np.asarray([], dtype=object),
        gt_velocity=np.zeros((0, 2), dtype=np.float32),
        num_lidar_pts=np.zeros((0,), dtype=np.int64),
        num_radar_pts=np.zeros((0,), dtype=np.int64),
        valid_flag=np.zeros((0,), dtype=np.bool_)
    )
    return info


def main():
    args = parse_args()
    target_tokens = load_target_tokens(args.split_file)

    raw_root = args.raw_root
    rgb_root = args.rgb_root

    instance_files = sorted((raw_root / 'instance_annotations' / 'instance_annotations').glob('*.json'))
    if target_tokens is not None:
        instance_files = [p for p in instance_files if p.stem.split('_')[0] in target_tokens]

    infos = []
    for path in mmcv.track_iter_progress(instance_files):
        token = path.stem.split('_')[0]
        info = build_single_info(token, raw_root, rgb_root, args.camera_types)
        infos.append(info)

    class_info_path = raw_root / 'instance_annotations' / 'class_info.json'
    instance_classes = []
    if class_info_path.exists():
        class_meta = mmcv.load(class_info_path)
        instance_classes = class_meta.get('classes', [])

    metadata = dict(
        version='woodscape-base',
        dataset='WoodScape',
        camera_types=list(args.camera_types),
        split=args.split,
        instance_classes=instance_classes,
        bbox2d_classes=['vehicles', 'person', 'bicycle', 'traffic_light', 'traffic_sign'],
    )

    mmcv.dump(dict(metadata=metadata, infos=infos), args.output)
    print(f'写出基础 info 至 {args.output}（样本数 {len(infos)}）')


if __name__ == '__main__':
    main()
