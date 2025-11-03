#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成一个满足 FastBEV/verify_woodscape_info.py 要求的 WoodScape 模拟数据集。
默认输出到 data/woodscape_mock。
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pickle


CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
CLASSES = ['pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle']


def make_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (640, 640), color=color)
    img.save(path)


def create_sample(idx: int, root: Path):
    token = f'woodscape-mock-{idx:06d}'
    timestamp = 1_700_000_000_000_000 + idx * 100_000
    num_boxes = 3 if idx % 2 == 0 else 2

    gt_boxes = np.stack([
        np.array([
            np.random.uniform(-40, 40),   # x
            np.random.uniform(-40, 40),   # y
            np.random.uniform(-1, 2),     # z
            np.random.uniform(1.5, 2.5),  # w
            np.random.uniform(3.0, 5.0),  # l
            np.random.uniform(1.2, 2.0),  # h
            np.random.uniform(-np.pi, np.pi),  # yaw
            np.random.uniform(-1.0, 1.0),      # vx
            np.random.uniform(-1.0, 1.0),      # vy
        ], dtype=np.float32) for _ in range(num_boxes)
    ], axis=0)

    gt_names = np.random.choice(CLASSES, size=(num_boxes,), replace=True)
    gt_velocity = gt_boxes[:, -2:].astype(np.float32)
    num_lidar_pts = np.random.randint(5, 15, size=(num_boxes,), dtype=np.int64)
    num_radar_pts = np.random.randint(0, 5, size=(num_boxes,), dtype=np.int64)
    valid_flag = (num_lidar_pts > 0).astype(bool)

    cams = {}
    for cam_idx, cam_name in enumerate(CAMERAS):
        img_path = root / 'rgb_images' / cam_name / f'{token}_{cam_name}.png'
        make_image(img_path, color=(40 + cam_idx * 40, 60 + idx * 30, 120 + cam_idx * 20))

        num_2d = np.random.randint(1, 4)
        bboxes = []
        labels = []
        for _ in range(num_2d):
            x1 = np.random.uniform(50, 300)
            y1 = np.random.uniform(50, 300)
            w = np.random.uniform(60, 150)
            h = np.random.uniform(60, 150)
            bboxes.append([x1, y1, min(x1 + w, 639), min(y1 + h, 639)])
            labels.append(np.random.choice(CLASSES))

        cams[cam_name] = dict(
            data_path=str(img_path),
            height=640,
            width=640,
            timestamp=timestamp,
            sensor2ego_translation=[float(cam_idx), 0.0, 1.5],
            sensor2ego_rotation=[0.0, 0.0, 0.0, 1.0],
            ego2global_translation=[0.0, 0.0, 0.0],
            ego2global_rotation=[0.0, 0.0, 0.0, 1.0],
            sensor2lidar_translation=np.zeros(3, dtype=np.float32),
            sensor2lidar_rotation=np.eye(3, dtype=np.float32),
            cam_intrinsic=np.array([
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 320.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32),
            cam_distortion=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            annos=dict(
                bbox=np.array(bboxes, dtype=np.float32),
                category_name=np.array(labels, dtype=object),
                category_id=np.array([CLASSES.index(lbl) for lbl in labels], dtype=np.int64),
                truncation=np.zeros(len(bboxes), dtype=np.float32),
                occlusion=np.zeros(len(bboxes), dtype=np.float32),
                instance_id=np.arange(len(bboxes), dtype=np.int64)
            )
        )

    sample = dict(
        token=token,
        scene_name='mock_scene',
        frame_id=idx,
        timestamp=timestamp,
        lidar_path='',
        sweeps=[],
        prev=None,
        next=None,
        lidar2ego_translation=[0.0, 0.0, 0.0],
        lidar2ego_rotation=[0.0, 0.0, 0.0, 1.0],
        ego2global_translation=[0.0, 0.0, 0.0],
        ego2global_rotation=[0.0, 0.0, 0.0, 1.0],
        gt_boxes=gt_boxes,
        gt_names=gt_names.astype(object),
        gt_velocity=gt_velocity,
        num_lidar_pts=num_lidar_pts,
        num_radar_pts=num_radar_pts,
        valid_flag=valid_flag,
        cams=cams,
    )
    return sample


def main(output_root: Path, num_samples: int):
    output_root.mkdir(parents=True, exist_ok=True)
    infos = [create_sample(i, output_root) for i in range(num_samples)]
    metadata = dict(version='woodscape-mock-v2', dataset='WoodScapeMock', camera_types=CAMERAS)
    bundle = dict(metadata=metadata, infos=infos)
    for split in ['train', 'val']:
        with (output_root / f'woodscape_infos_{split}.pkl').open('wb') as f:
            pickle.dump(bundle, f)
    print(f'Generated {num_samples} samples in {output_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 mock WoodScape 数据集')
    parser.add_argument('--output-root', type=Path, default=Path('data/woodscape_mock'))
    parser.add_argument('--num-samples', type=int, default=4)
    args = parser.parse_args()
    main(args.output_root, args.num_samples)
