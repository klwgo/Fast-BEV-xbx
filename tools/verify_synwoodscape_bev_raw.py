#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单样本 BEV/3D GT 验证脚本（完全复刻 build_synwoodscape_info.py 内的处理逻辑）。

功能：
    1. 从 SynWoodScape 原始目录读取指定 token；
    2. 解析车辆姿态，将 3D 框转换到车体系；
    3. 对 BEV PNG 做居中、旋转、翻转、重采样；
    4. 叠加 3D footprint，直观验证障碍物是否对齐。

使用示例：
    python tools/verify_synwoodscape_bev_raw.py \
        --raw-root data/synwoodscape/SynWoodScape_V0.1.1 \
        --token 00000 \
        --point-cloud-range -130 -100 -2 130 170 6 \
        --bev-resolution 0.4 \
        --center-ego --rotate-to-ego \
        --out work_dirs/bev_single_check \
        --show
"""

import argparse
import math
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import mmcv
import numpy as np

EGO_SEMANTIC_ID = 24
CARLA_TO_FASTBEV = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
TRANSFORM_RE = re.compile(
    r'Transform\(Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\), '
    r'Rotation\(pitch=([-\d.]+), yaw=([-\d.]+), roll=([-\d.]+)\)\)'
)


def parse_args():
    parser = argparse.ArgumentParser('单样本验证 SynWoodScape BEV/3D 对齐情况')
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='SynWoodScape 解压目录')
    parser.add_argument('--token', type=str, default=None,
                        help='指定 token，例如 00000；若为空则根据 index 选择')
    parser.add_argument('--index', type=int, default=0,
                        help='未指定 token 时，按排序后的第 index 个样本')
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-130.0, -100.0, -2.0, 130.0, 170.0, 6.0],
                        metavar=('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'))
    parser.add_argument('--bev-resolution', type=float, default=0.4,
                        help='BEV 栅格大小（米/像素），<=0 表示保持原尺寸')
    parser.add_argument('--center-ego', action='store_true',
                        help='将 BEV PNG 通过 ego-vehicle 居中')
    parser.add_argument('--rotate-to-ego', action='store_true',
                        help='根据车辆 yaw 将 BEV PNG 旋转到车体坐标')
    parser.add_argument('--flip-y', action='store_true',
                        help='是否在 raster 前做 Y 轴翻转（对齐训练坐标）')
    parser.add_argument('--out', type=Path, default=Path('work_dirs/bev_single_check'),
                        help='输出目录')
    parser.add_argument('--show', action='store_true', help='是否弹出窗口')
    return parser.parse_args()


def list_tokens(root: Path):
    box_dir = root / 'box_3d_annotations'
    return sorted(p.stem for p in box_dir.glob('*.pkl'))


def euler_to_matrix(roll, pitch, yaw):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1, 0, 0],
                      [0, cr, -sr],
                      [0, sr, cr]], dtype=np.float32)
    rot_y = np.array([[cp, 0, sp],
                      [0, 1, 0],
                      [-sp, 0, cp]], dtype=np.float32)
    rot_z = np.array([[cy, -sy, 0],
                      [sy, cy, 0],
                      [0, 0, 1]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def carla_to_fastbev_rotation(rot_carla):
    return CARLA_TO_FASTBEV @ rot_carla @ CARLA_TO_FASTBEV


def carla_to_fastbev_translation(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(3, 1)
    return (CARLA_TO_FASTBEV @ vec).reshape(3)


def load_vehicle_pose(raw_root: Path, token: str):
    txt_path = raw_root / 'vehicle_data' / 'rgb_images' / f'{token}.txt'
    pose = dict(
        translation=np.zeros(3, dtype=np.float32),
        rot_ego_to_global=np.eye(3, dtype=np.float32),
        yaw_deg=0.0,
    )
    if not txt_path.exists():
        return pose
    match = None
    with txt_path.open() as f:
        for line in f:
            m = TRANSFORM_RE.search(line.strip())
            if m:
                match = m
                break
    if match is None:
        return pose
    loc_x, loc_y, loc_z, pitch, yaw, roll = map(float, match.groups())
    trans_carla = np.array([loc_x, loc_y, loc_z], dtype=np.float32)
    rot_carla = euler_to_matrix(roll, pitch, yaw)
    pose.update(
        translation=carla_to_fastbev_translation(trans_carla),
        rot_ego_to_global=carla_to_fastbev_rotation(rot_carla),
        yaw_deg=yaw  # 直接使用原始文件里给出的 yaw
    )
    return pose


def center_bev_mask(mask):
    coords = np.argwhere(mask == EGO_SEMANTIC_ID)
    if coords.size == 0:
        return mask
    target = np.array(mask.shape[:2], dtype=np.float32) / 2.0
    center = coords.mean(axis=0)
    shift = np.round(target - center).astype(int)
    shifted = np.roll(mask, shift[0], axis=0)
    shifted = np.roll(shifted, shift[1], axis=1)
    shifted[shifted == EGO_SEMANTIC_ID] = 0
    return shifted


def align_bev_mask(mask, target_shape, flip_y):
    aligned = mask
    if flip_y:
        aligned = np.flip(aligned, axis=0)
    if target_shape is not None:
        aligned = mmcv.imresize(
            aligned, (int(target_shape[1]), int(target_shape[0])), interpolation='nearest')
    return aligned


def compute_target_shape(pc_range, resolution):
    if resolution <= 0:
        return None
    span_x = pc_range[3] - pc_range[0]
    span_y = pc_range[4] - pc_range[1]
    if span_x <= 0 or span_y <= 0:
        return None
    width = max(int(round(span_x / resolution)), 1)
    height = max(int(round(span_y / resolution)), 1)
    return (height, width)


def load_box_corners(box_path: Path, pose: dict):
    with box_path.open('rb') as f:
        box_dict = pickle.load(f)
    footprints = []
    for corners in box_dict.values():
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape[-1] > 3:
            corners = corners[..., :3]
        corners_fb = (CARLA_TO_FASTBEV @ corners.T).T
        rot_world2ego = pose['rot_ego_to_global'].T
        corners_ego = (corners_fb - pose['translation'][None, :]) @ rot_world2ego
        footprints.append(corners_ego[:4, :2])
    return footprints


def rasterize(polygons, pc_range, bev_shape):
    h, w = bev_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    x_min, y_min, _, x_max, y_max, _ = pc_range
    scale_x = w / max(x_max - x_min, 1e-6)
    scale_y = h / max(y_max - y_min, 1e-6)
    for poly in polygons:
        px = (poly[:, 0] - x_min) * scale_x
        py = h - (poly[:, 1] - y_min) * scale_y
        poly_pix = np.stack([px, py], axis=1)
        x0 = max(int(np.floor(poly_pix[:, 0].min())), 0)
        x1 = min(int(np.ceil(poly_pix[:, 0].max())) + 1, w)
        y0 = max(int(np.floor(poly_pix[:, 1].min())), 0)
        y1 = min(int(np.ceil(poly_pix[:, 1].max())) + 1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        grid_x, grid_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        path = MplPath(poly_pix)
        inside = path.contains_points(pts).reshape((y1 - y0), (x1 - x0))
        mask[y0:y1, x0:x1][inside] = 1
    return mask


def main():
    args = parse_args()
    tokens = list_tokens(args.raw_root)
    if not tokens:
        raise RuntimeError('未在 raw-root 找到任何 box_3d_annotations/*.pkl')
    token = args.token or tokens[min(max(args.index, 0), len(tokens) - 1)]
    print(f'[Info] 当前验证样本 token={token}')

    pose = load_vehicle_pose(args.raw_root, token)
    pc_range = np.array(args.point_cloud_range, dtype=np.float32)
    target_shape = compute_target_shape(pc_range, args.bev_resolution)

    bev_png = args.raw_root / 'semantic_annotations' / 'gtLabels' / f'{token}_BEV.png'
    bev_mask = mmcv.imread(str(bev_png), flag='unchanged').astype(np.uint8)
    if args.center_ego:
        bev_mask = center_bev_mask(bev_mask)
    if args.rotate_to_ego:
        bev_mask = mmcv.imrotate(bev_mask, angle=-pose['yaw_deg'],
                                 border_value=0, auto_bound=True)
    bev_aligned = align_bev_mask(bev_mask, target_shape, flip_y=args.flip_y)

    footprints = load_box_corners(args.raw_root / 'box_3d_annotations' / f'{token}.pkl', pose)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_min, y_min, _, x_max, y_max, _ = pc_range
    extent = [x_min, x_max, y_min, y_max]

    axes[0].imshow(bev_aligned, origin='lower', extent=extent, cmap='tab20')
    for fp in footprints:
        axes[0].plot(fp[:, 0], fp[:, 1], 'r-', linewidth=0.8)
    axes[0].set_title('原始 BEV PNG 对齐后 + 3D footprint')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    bev_binary = (bev_aligned > 0).astype(np.uint8)
    axes[1].imshow(bev_binary, origin='lower', extent=extent, cmap='gray')
    for fp in footprints:
        axes[1].plot(fp[:, 0], fp[:, 1], 'r-', linewidth=0.8)
    axes[1].set_title('二值化后的 BEV + footprint')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_xlabel('X (m)')

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f'verify_{token}.png'
    fig.savefig(out_file, dpi=150)
    print(f'[Info] 保存结果到 {out_file}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
