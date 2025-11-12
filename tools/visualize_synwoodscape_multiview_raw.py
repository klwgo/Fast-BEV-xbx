#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 SynWoodScape 原始目录里同时可视化四个相机视图的 2D 框
以及 BEV+3D footprint，帮助核对标注是否有效。

示例：
    python tools/visualize_synwoodscape_multiview_raw.py \
        --raw-root data/synwoodscape/SynWoodScape_V0.1.1 \
        --token 00000 \
        --point-cloud-range -130 -100 -2 130 170 6 \
        --bev-resolution 0.4 \
        --out work_dirs/mv_raw_vis
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

CAM_MAP = {
    'CAM_FRONT': 'FV',
    'CAM_FRONT_LEFT': 'MVL',
    'CAM_FRONT_RIGHT': 'MVR',
    'CAM_BACK': 'RV',
}
EGO_SEMANTIC_ID = 24
CARLA_TO_FASTBEV = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
TRANSFORM_RE = re.compile(
    r'Transform\(Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\), '
    r'Rotation\(pitch=([-\d.]+), yaw=([-\d.]+), roll=([-\d.]+)\)\)'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SynWoodScape raw multi-view 2D boxes + BEV 3D footprint')
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='SynWoodScape 解压根目录')
    parser.add_argument('--token', type=str, default=None,
                        help='指定帧 token（例如 00000）')
    parser.add_argument('--index', type=int, default=0,
                        help='若未提供 token，则按排序后的第 index 个 token')
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[-130.0, -100.0, -2.0, 130.0, 170.0, 6.0],
                        metavar=('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'))
    parser.add_argument('--bev-resolution', type=float, default=0.4,
                        help='BEV 栅格大小（米/像素），<=0 表示保持原尺寸')
    parser.add_argument('--center-ego', action='store_true',
                        help='是否根据 ego-vehicle 像素居中 BEV mask')
    parser.add_argument('--rotate-to-ego', action='store_true',
                        help='是否根据车辆 yaw 将 BEV mask 旋转到车体坐标')
    parser.add_argument('--out', type=Path, default=Path('work_dirs/mv_raw_vis'),
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
    trans_fastbev = carla_to_fastbev_translation(trans_carla)
    rot_fastbev = carla_to_fastbev_rotation(rot_carla)
    pose.update(
        translation=trans_fastbev,
        rot_ego_to_global=rot_fastbev,
        yaw_deg=float(math.degrees(math.atan2(rot_fastbev[1, 0], rot_fastbev[0, 0])))
    )
    return pose


def load_boxes(box_path: Path, pose: dict):
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


def align_bev(mask, target_shape, flip_y=True):
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


def load_box2d(txt_path):
    boxes = []
    labels = []
    if not txt_path.exists():
        return boxes, labels
    with txt_path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 6:
                continue
            cls_name = parts[0]
            x1, y1, x2, y2 = map(float, parts[2:6])
            boxes.append([x1, y1, x2, y2])
            labels.append(cls_name)
    return boxes, labels


def main():
    args = parse_args()
    tokens = list_tokens(args.raw_root)
    if not tokens:
        raise RuntimeError('未在 raw-root 下找到 box_3d_annotations/*.pkl')
    if args.token is not None:
        token = args.token
    else:
        token = tokens[min(max(args.index, 0), len(tokens) - 1)]
    print(f'[Info] 使用 token={token}')

    pose = load_vehicle_pose(args.raw_root, token)
    pc_range = np.array(args.point_cloud_range, dtype=np.float32)
    target_shape = compute_target_shape(pc_range, args.bev_resolution)

    bev_path = args.raw_root / 'semantic_annotations' / 'gtLabels' / f'{token}_BEV.png'
    bev_mask = mmcv.imread(str(bev_path), flag='unchanged').astype(np.uint8)
    if args.center_ego:
        bev_mask = center_bev_mask(bev_mask)
    if args.rotate_to_ego:
        bev_mask = mmcv.imrotate(bev_mask, angle=-pose['yaw_deg'], border_value=0, auto_bound=True)
    bev_aligned = align_bev(bev_mask, target_shape)

    footprints = load_boxes(args.raw_root / 'box_3d_annotations' / f'{token}.pkl', pose)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_mv = axes[0, :3]
    cam_names = list(CAM_MAP.keys())
    for idx, (cam_name, ax) in enumerate(zip(cam_names, axes_mv)):
        suffix = CAM_MAP[cam_name]
        img_path = args.raw_root / 'rgb_images' / f'{token}_{suffix}.png'
        box2d_path = args.raw_root / 'box_2d_annotations' / f'{token}_{suffix}.txt'
        img = mmcv.imread(str(img_path))
        img = mmcv.bgr2rgb(img)
        ax.imshow(img)
        boxes2d, labels2d = load_box2d(box2d_path)
        for box, label in zip(boxes2d, labels2d):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                edgecolor='lime', facecolor='none', linewidth=1)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label, color='yellow', fontsize=6,
                    bbox=dict(facecolor='black', alpha=0.4, linewidth=0))
        ax.set_title(cam_name)
        ax.axis('off')

    # BEV subplot
    ax_bev = axes[1, 0]
    x_min, y_min, _, x_max, y_max, _ = pc_range
    extent = [x_min, x_max, y_min, y_max]
    ax_bev.imshow(bev_aligned, origin='lower', extent=extent, cmap='tab20')
    for footprint in footprints:
        ax_bev.plot(footprint[:, 0], footprint[:, 1], 'r-', linewidth=0.8)
    ax_bev.set_xlabel('X (m)')
    ax_bev.set_ylabel('Y (m)')
    ax_bev.set_xlim(x_min, x_max)
    ax_bev.set_ylim(y_min, y_max)
    ax_bev.set_title('BEV + 3D footprints')

    # Text summary
    ax_summary = axes[1, 1]
    ax_summary.axis('off')
    summary_lines = [
        f'token: {token}',
        f'#3D boxes: {len(footprints)}',
        f'point_cloud_range: {pc_range.tolist()}',
        f'center_ego: {args.center_ego}, rotate_to_ego: {args.rotate_to_ego}',
    ]
    ax_summary.text(0, 1, '\n'.join(summary_lines), va='top', fontsize=12)

    # Empty subplot placeholder
    axes[1, 2].axis('off')

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f'mv_raw_{token}.png'
    fig.savefig(out_file, dpi=150)
    print(f'[Info] 保存结果到 {out_file}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

