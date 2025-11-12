#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从 SynWoodScape 原始目录读取 BEV 语义 PNG 与 3D 框，
将二者叠加在统一坐标系下进行可视化。

示例：
    python tools/visualize_synwoodscape_raw.py \
        --raw-root data/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0 \
        --token 00000 \
        --point-cloud-range -130 -100 -2 130 170 6 \
        --bev-resolution 0.4 \
        --out work_dirs/raw_vis
"""

import argparse
import math
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
import numpy as np


EGO_SEMANTIC_ID = 24
CARLA_TO_FASTBEV = np.diag([1.0, -1.0, 1.0]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser('Visualize SynWoodScape raw BEV + 3D boxes')
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='SynWoodScape 解压根目录（包含 box_3d_annotations/ 等子目录）')
    parser.add_argument('--token', type=str, default=None,
                        help='指定帧 token（例如 00000）。若为空则使用 --index')
    parser.add_argument('--index', type=int, default=0,
                        help='按字典序挑选第 index 个 token（仅在未提供 --token 时生效）')
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-130.0, -100.0, -2.0, 130.0, 170.0, 6.0],
                        metavar=('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'),
                        help='可视化时使用的空间范围 (m)')
    parser.add_argument('--bev-resolution', type=float, default=0.4,
                        help='BEV 图像重采样分辨率（米/像素）；<=0 时保持原尺寸')
    parser.add_argument('--out', type=Path, default=Path('work_dirs/raw_vis'),
                        help='输出目录')
    parser.add_argument('--show', action='store_true',
                        help='是否弹出窗口展示（默认仅保存 PNG）')
    parser.add_argument('--center-ego', action='store_true',
                        help='将 BEV 语义图按照 ego-vehicle 像素居中再叠加')
    parser.add_argument('--rotate-to-ego', action='store_true',
                        help='根据车辆姿态将 BEV 图旋转到车体坐标')
    return parser.parse_args()


def list_tokens(root: Path):
    box_dir = root / 'box_3d_annotations'
    tokens = sorted(p.stem for p in box_dir.glob('*.pkl'))
    return tokens


def euler_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
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


def carla_to_fastbev_rotation(rot_carla: np.ndarray) -> np.ndarray:
    return CARLA_TO_FASTBEV @ rot_carla @ CARLA_TO_FASTBEV


def carla_to_fastbev_translation(vec_carla: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec_carla, dtype=np.float32).reshape(3, 1)
    return (CARLA_TO_FASTBEV @ vec).reshape(3)


TRANSFORM_RE = re.compile(
    r'Transform\(Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\), '
    r'Rotation\(pitch=([-\d.]+), yaw=([-\d.]+), roll=([-\d.]+)\)\)'
)


def load_vehicle_pose(raw_root: Path, token: str):
    txt_path = raw_root / 'vehicle_data' / 'rgb_images' / f'{token}.txt'
    pose = dict(
        translation=np.zeros(3, dtype=np.float32),
        rot_ego_to_global=np.eye(3, dtype=np.float32),
        yaw_deg=0.0,
    )
    if not txt_path.exists():
        return pose
    frame_id = None
    match = None
    with txt_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('frame'):
                try:
                    frame_id = int(line.split('=')[1])
                except (IndexError, ValueError):
                    frame_id = None
            m = TRANSFORM_RE.search(line)
            if m:
                match = m
                break
    if match is None:
        return pose
    loc_x, loc_y, loc_z, pitch, yaw, roll = map(float, match.groups())
    trans_carla = np.array([loc_x, loc_y, loc_z], dtype=np.float32)
    rot_carla = euler_to_matrix(roll, pitch, yaw)
    trans_fastbev = carla_to_fastbev_translation(trans_carla)
    rot_fastbev = carla_to_fastbev_rotation(rot_carla)
    pose.update(
        translation=trans_fastbev,
        rot_ego_to_global=rot_fastbev,
        yaw_deg=float(math.degrees(math.atan2(rot_fastbev[1, 0], rot_fastbev[0, 0])))
    )
    return pose


def corners_to_box(corners: np.ndarray):
    pts = corners[:, :3]
    z_min = pts[:, 2].min()
    z_max = pts[:, 2].max()
    height = float(z_max - z_min)
    bottom_idx = np.argsort(pts[:, 2])[:4]
    foot = pts[bottom_idx, :2]
    centroid = foot.mean(axis=0)
    centered = foot - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis_long = eigvecs[:, order[0]]
    axis_short = eigvecs[:, order[1]]
    proj_long = centered @ axis_long
    proj_short = centered @ axis_short
    length = float(np.ptp(proj_long))
    width = float(np.ptp(proj_short))
    if width > length:
        width, length = length, width
        axis_long, axis_short = axis_short, axis_long
    yaw = float(np.arctan2(axis_long[1], axis_long[0]))
    return centroid, np.array([width, length, height], dtype=np.float32), yaw


def load_boxes(box_path: Path, pose: dict):
    with box_path.open('rb') as f:
        box_dict = pickle.load(f)
    centers = []
    dims = []
    yaws = []
    for corners in box_dict.values():
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape[-1] > 3:
            corners = corners[..., :3]
        corners_fb = (CARLA_TO_FASTBEV @ corners.T).T
        rot_world2ego = pose['rot_ego_to_global'].T
        corners_ego = (corners_fb - pose['translation'][None, :]) @ rot_world2ego
        center, dim, yaw = corners_to_box(corners_ego)
        centers.append(center)
        dims.append(dim)
        yaws.append(yaw)
    if not centers:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.vstack(centers), np.vstack(dims), np.asarray(yaws, dtype=np.float32)


def center_bev_map(sem_map: np.ndarray, ego_label: int = EGO_SEMANTIC_ID) -> np.ndarray:
    if sem_map.ndim > 2:
        sem_map = sem_map.squeeze()
    coords = np.argwhere(sem_map == ego_label)
    if coords.size == 0:
        return sem_map
    target = np.array(sem_map.shape[:2], dtype=np.float32) / 2.0
    center = coords.mean(axis=0)
    shift = np.round(target - center).astype(int)
    shifted = np.roll(sem_map, shift[0], axis=0)
    shifted = np.roll(shifted, shift[1], axis=1)
    shifted[shifted == ego_label] = 0
    return shifted


def align_bev_image(img: np.ndarray, target_shape, flip_y=True):
    aligned = img
    if flip_y:
        aligned = np.flip(aligned, axis=0)
    if target_shape is not None:
        aligned = mmcv.imresize(
            aligned,
            (int(target_shape[1]), int(target_shape[0])),
            interpolation='nearest')
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


def corners_from_box(center, dim, yaw):
    dx, dy = dim[0], dim[1]
    local = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
    ], dtype=np.float32)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]], dtype=np.float32)
    return center[:2] + local @ rot.T


def main():
    args = parse_args()
    raw_root = args.raw_root
    tokens = list_tokens(raw_root)
    if not tokens:
        raise RuntimeError(f'在 {raw_root} 下未找到 box_3d_annotations/*.pkl')
    token = args.token or tokens[min(max(args.index, 0), len(tokens) - 1)]
    print(f'[Info] 使用 token={token}')

    bev_path = raw_root / 'semantic_annotations' / 'gtLabels' / f'{token}_BEV.png'
    box_path = raw_root / 'box_3d_annotations' / f'{token}.pkl'
    if not bev_path.exists():
        raise FileNotFoundError(f'BEV 语义文件不存在: {bev_path}')
    if not box_path.exists():
        raise FileNotFoundError(f'3D 框文件不存在: {box_path}')

    pose = load_vehicle_pose(raw_root, token)
    bev_raw = mmcv.imread(str(bev_path), flag='unchanged')
    if args.center_ego:
        bev_raw = center_bev_map(bev_raw)
    if args.rotate_to_ego:
        bev_raw = mmcv.imrotate(bev_raw, angle=-pose['yaw_deg'], border_value=0, auto_bound=True)
    pc_range = np.array(args.point_cloud_range, dtype=np.float32)
    target_shape = compute_target_shape(pc_range, args.bev_resolution)
    bev_aligned = align_bev_image(bev_raw, target_shape)

    centers, dims, yaws = load_boxes(box_path, pose)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_min, y_min, _, x_max, y_max, _ = pc_range
    extent = [x_min, x_max, y_min, y_max]

    axes[0].imshow(bev_aligned, origin='lower', extent=extent, aspect='auto')
    axes[0].set_title(f'BEV 语义 + 3D footprint ({token})')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')

    for center, dim, yaw in zip(centers, dims, yaws):
        footprint = corners_from_box(center, dim, yaw)
        axes[0].add_patch(patches.Polygon(
            footprint,
            closed=True,
            edgecolor='red',
            facecolor='none',
            linewidth=1.2))

    axes[1].scatter(centers[:, 0], centers[:, 1], s=10, c='g')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('3D 框中心分布')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    args.out.mkdir(parents=True, exist_ok=True)
    # 保存未叠加的 BEV 语义
    fig_only = plt.figure(figsize=(6, 6))
    ax_only = fig_only.add_subplot(111)
    ax_only.imshow(bev_aligned, origin='lower', extent=extent, aspect='auto')
    ax_only.set_title(f'BEV 语义 ({token})')
    ax_only.set_xlim(x_min, x_max)
    ax_only.set_ylim(y_min, y_max)
    ax_only.set_xlabel('X (m)')
    ax_only.set_ylabel('Y (m)')
    fig_only.tight_layout()
    raw_only_file = args.out / f'raw_bev_{token}_semantic.png'
    fig_only.savefig(raw_only_file, dpi=160)
    plt.close(fig_only)

    out_file = args.out / f'raw_bev_{token}_overlay.png'
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    print(f'[Info] 保存 BEV 语义图到 {raw_only_file}')
    print(f'[Info] 保存叠加图到 {out_file}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
