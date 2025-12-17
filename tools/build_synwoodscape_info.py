#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 SynWoodScape 数据转换为 Fast-BEV 所需的 info.pkl."""

import argparse
import ast
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import mmcv
import numpy as np

# 相机命名前缀映射（与 WoodScape 配置保持一致）
CAM_MAP = {
    'CAM_FRONT': 'FV',
    'CAM_FRONT_LEFT': 'MVL',
    'CAM_FRONT_RIGHT': 'MVR',
    'CAM_BACK': 'RV',
}

# 预训练/微调共享的三类检测类别（需与 WoodScapeMultiViewDataset.CLASSES 对齐）
DETECTION_CLASSES = ['pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle']
DET_CLASS_TO_ID = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}
# 语义标签到检测类别的映射，无法匹配的类别将被忽略
SEMANTIC_TO_DET = {
    'pedestrian': 'pedestrian',
    'four-wheeler vehicle': 'four-wheeler vehicle',
    'two-wheeler vehicle': 'two-wheeler vehicle',
}
# BEV 分割类别
BEV_CLASS_NAMES = [
    'road_surface',
    'lane_marking',
    'sidewalk',
    'vegetation',
    'ground',
    'background',
]
BEV_CLASS_TO_ID = {name: idx for idx, name in enumerate(BEV_CLASS_NAMES)}
SEMANTIC_TO_BEV = {
    'road': 'road_surface',
    'road_line': 'lane_marking',
    'sidewalk': 'sidewalk',
    'vegetation': 'vegetation',
    'ground': 'ground',
    'terrain': 'ground',
}
# CARLA -> Fast-BEV 坐标系转换矩阵（保持 X 前、Y 左、Z 上）
CARLA_TO_FASTBEV = np.diag([1.0, -1.0, 1.0]).astype(np.float32)

# 语义 ID -> 类别名称（来自 SynWoodScape README）
SEMANTIC_ID_TO_CLASS = {
    0: 'unlabeled',
    1: 'building',
    2: 'fence',
    3: 'other',
    4: 'pedestrian',
    5: 'pole',
    6: 'road_line',
    7: 'road',
    8: 'sidewalk',
    9: 'vegetation',
    10: 'four-wheeler vehicle',
    11: 'wall',
    12: 'traffic_sign',
    13: 'sky',
    14: 'ground',
    15: 'bridge',
    16: 'rail_track',
    17: 'guard_rail',
    18: 'traffic_light',
    19: 'water',
    20: 'terrain',
    21: 'two-wheeler vehicle',
    22: 'static',
    23: 'dynamic',
    24: 'ego-vehicle',
}
EGO_SEMANTIC_ID = 24


def _normalize_quaternion(quat):
    arr = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return arr / norm


def quaternion_wxyz_to_matrix(quat) -> np.ndarray:
    """将四元数 (w, x, y, z) 转为旋转矩阵。"""
    w, x, y, z = _normalize_quaternion(quat)
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def matrix_to_quaternion_wxyz(mat: np.ndarray) -> np.ndarray:
    """旋转矩阵转四元数 (w, x, y, z)。"""
    m = np.asarray(mat, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    quat /= max(np.linalg.norm(quat), 1e-8)
    return quat


def carla_to_fastbev_rotation(rot_carla: np.ndarray) -> np.ndarray:
    """将 CARLA 坐标系下的旋转转换至 Fast-BEV/木渎坐标系。"""
    return CARLA_TO_FASTBEV @ rot_carla @ CARLA_TO_FASTBEV


def carla_to_fastbev_translation(vec_carla: np.ndarray) -> np.ndarray:
    """CARLA 坐标 -> Fast-BEV 坐标的平移变换。"""
    vec = np.asarray(vec_carla, dtype=np.float32)
    return (CARLA_TO_FASTBEV @ vec.reshape(3, 1)).reshape(3)


def convert_bev_semantic(sem_map: np.ndarray) -> np.ndarray:
    """将 BEV 语义标签转换为 Fast-BEV 训练所需的 6 类栅格."""
    mask = np.full(sem_map.shape, BEV_CLASS_TO_ID['background'], dtype=np.uint8)
    for sem_id, sem_name in SEMANTIC_ID_TO_CLASS.items():
        target = SEMANTIC_TO_BEV.get(sem_name)
        if target is None:
            continue
        mask[sem_map == sem_id] = BEV_CLASS_TO_ID[target]
    return mask


def center_bev_map(sem_map: np.ndarray,
                   ego_label: int = EGO_SEMANTIC_ID) -> np.ndarray:
    """通过平移使 ego-vehicle 像素位于图像中心。"""
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


def rotate_bev_map(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """旋转 BEV mask，使车辆前向对齐 Fast-BEV X 轴."""
    if abs(angle_deg) < 1e-3:
        return mask
    rotated = mmcv.imrotate(
        mask,
        angle=angle_deg,
        border_value=0,
        auto_bound=True,
        interpolation='nearest')
    return rotated.astype(mask.dtype, copy=False)


def align_bev_mask(mask: np.ndarray,
                   target_shape: Optional[Tuple[int, int]] = None,
                   flip_y: bool = True) -> np.ndarray:
    """根据配置对齐 BEV mask 的朝向与分辨率."""
    aligned = mask
    if flip_y:
        aligned = np.flip(aligned, axis=0)
    if target_shape is not None and tuple(aligned.shape) != tuple(target_shape):
        aligned = mmcv.imresize(
            aligned,
            (int(target_shape[1]), int(target_shape[0])),
            interpolation='nearest')
    return aligned.astype(np.uint8)


def euler_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """根据 ZYX (yaw-pitch-roll) 顺序构建旋转矩阵，角度单位为度。"""
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
    # 遵循 yaw -> pitch -> roll
    return rot_z @ rot_y @ rot_x


def load_calibrations(calib_dir: Path) -> Dict[str, dict]:
    """读取 calibration_data/*.json 并解析为相机参数字典。"""
    calibs: Dict[str, dict] = {}
    if not calib_dir.exists():
        return calibs
    for json_path in calib_dir.glob('*.json'):
        data = mmcv.load(json_path)
        name = data.get('name') or json_path.stem
        intrinsic_info = data.get('intrinsic', {})
        extrinsic_info = data.get('extrinsic', {})

        width = int(round(intrinsic_info.get('width', 0)))
        height = int(round(intrinsic_info.get('height', 0)))
        aspect = float(intrinsic_info.get('aspect_ratio', 1.0)) or 1.0
        focal = float(intrinsic_info.get('focal_length', intrinsic_info.get('k1', 1.0)))
        if focal <= 0:
            focal = 1.0
        fx = focal
        fy = focal / aspect
        cx = width * 0.5 + float(intrinsic_info.get('cx_offset', 0.0))
        cy = height * 0.5 + float(intrinsic_info.get('cy_offset', 0.0))
        cam_intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        distortion = np.array(
            [float(intrinsic_info.get(f'k{i}', 0.0)) for i in range(1, 5)],
            dtype=np.float32)
        radial_params = dict(
            k1=float(intrinsic_info.get('k1', 0.0)),
            k2=float(intrinsic_info.get('k2', 0.0)),
            k3=float(intrinsic_info.get('k3', 0.0)),
            k4=float(intrinsic_info.get('k4', 0.0)),
            cx_offset=float(intrinsic_info.get('cx_offset', 0.0)),
            cy_offset=float(intrinsic_info.get('cy_offset', 0.0)),
            aspect_ratio=float(intrinsic_info.get('aspect_ratio', 1.0) or 1.0),
            width=float(width),
            height=float(height),
        )
        quat = extrinsic_info.get('quaternion', [1.0, 0.0, 0.0, 0.0])
        trans = extrinsic_info.get(
            'translation used in CARLA (CARLA reference)',
            extrinsic_info.get('translation', [0.0, 0.0, 0.0])
        )
        model = intrinsic_info.get('model', 'radial_poly')

        calibs[name] = dict(
            intrinsic=cam_intrinsic,
            distortion=distortion,
            radial_params=radial_params,
            quaternion=np.asarray(quat, dtype=np.float32),
            translation=np.asarray(trans, dtype=np.float32),
            width=width,
            height=height,
            model=model,
        )
    return calibs


TRANSFORM_RE = re.compile(
    r'Transform\(Location\(x=([-\d.]+), y=([-\d.]+), z=([-\d.]+)\), '
    r'Rotation\(pitch=([-\d.]+), yaw=([-\d.]+), roll=([-\d.]+)\)\)'
)


def load_vehicle_pose(vehicle_root: Path, token: str) -> Dict[str, np.ndarray]:
    """从 vehicle_data 文本中解析车辆位姿，返回 Fast-BEV 坐标系下的外参。"""
    default_ts = int(token) if token.isdigit() else 0
    pose = dict(
        translation=np.zeros(3, dtype=np.float32),
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        timestamp=default_ts,
    )
    txt_path = vehicle_root / 'rgb_images' / f'{token}.txt'
    if not txt_path.exists():
        return pose

    frame_id: Optional[int] = None
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
    if match is None:
        # 未能解析 Transform，返回默认姿态
        if frame_id is not None:
            pose['timestamp'] = frame_id
        return pose

    loc_x, loc_y, loc_z, pitch, yaw, roll = map(float, match.groups())
    trans_carla = np.array([loc_x, loc_y, loc_z], dtype=np.float32)
    rot_carla = euler_to_matrix(roll, pitch, yaw)
    trans_fastbev = carla_to_fastbev_translation(trans_carla)
    rot_fastbev = carla_to_fastbev_rotation(rot_carla)
    quat_fastbev = matrix_to_quaternion_wxyz(rot_fastbev)

    timestamp = frame_id if frame_id is not None else default_ts
    # 将帧号转换为微秒尺度（保持相对顺序即可）
    pose.update(
        translation=trans_fastbev,
        rotation=quat_fastbev,
        timestamp=int(timestamp) * 100_000,
    )
    return pose

def parse_args():
    parser = argparse.ArgumentParser(description='构建 SynWoodScape info')
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='SynWoodScape_V0.1.0 根目录')
    parser.add_argument('--output', type=Path, required=True,
                        help='输出 pkl 路径')
    parser.add_argument('--split', type=str, default='syn',
                        help='metadata 的 split 标识')
    parser.add_argument('--limit', type=int, default=None,
                        help='可选，仅处理前 N 帧用于调试')
    parser.add_argument(
        '--point-cloud-range',
        type=float,
        nargs=6,
        default=[-130.0, -100.0, -2.0, 130.0, 170.0, 6.0],
        metavar=('xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax'),
        help='用于对齐 BEV mask 的物理范围，单位米')
    parser.add_argument('--bev-resolution', type=float, default=0.4,
                        help='BEV 栅格大小（米/像素），<=0 表示保持原分辨率')
    parser.add_argument('--disable-bev-flip', dest='bev_flip_y', action='store_false',
                        help='禁用 BEV mask 的垂直翻转')
    parser.add_argument('--disable-bev-center', dest='bev_center', action='store_false',
                        help='禁用 BEV mask 居中平移')
    parser.add_argument('--disable-bev-rotate', dest='bev_rotate', action='store_false',
                        help='禁用 BEV mask 旋转到车体坐标')
    parser.set_defaults(bev_flip_y=True, bev_center=True, bev_rotate=True)
    return parser.parse_args()


def _compute_bev_shape(point_cloud_range: np.ndarray,
                       resolution: float) -> Optional[Tuple[int, int]]:
    if resolution <= 0:
        return None
    span_x = float(point_cloud_range[3] - point_cloud_range[0])
    span_y = float(point_cloud_range[4] - point_cloud_range[1])
    if span_x <= 0 or span_y <= 0:
        return None
    width = max(int(round(span_x / resolution)), 1)
    height = max(int(round(span_y / resolution)), 1)
    return height, width


def load_distances(path: Path):
    mapping = {}
    if not path.exists():
        return mapping
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj_token, rest = line.split(',', 1)
            obj_id = int(obj_token.strip())
            left = rest.find('[')
            right = rest.find(']', left)
            if left == -1 or right == -1:
                continue
            color_repr = rest[left:right + 1]
            try:
                color_seq = ast.literal_eval(color_repr)
            except (ValueError, SyntaxError):
                color_seq = [int(x) for x in color_repr.strip('[] ').split(',')]
            color = tuple(int(c) for c in color_seq)
            mapping[obj_id] = color
    return mapping


def corners_to_box(corners: np.ndarray):
    """将 8 个顶点转换为 (center, dims, yaw)."""
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
    center = np.array([centroid[0], centroid[1], z_min + height / 2], dtype=np.float32)
    dims = np.array([width, length, height], dtype=np.float32)
    return center, dims, yaw


def majority(arr):
    if arr.size == 0:
        return None
    cnt = Counter(arr.tolist())
    return cnt.most_common(1)[0][0]


def main():
    args = parse_args()
    raw_root = args.raw_root
    pc_range = np.asarray(args.point_cloud_range, dtype=np.float32)
    bev_target_shape = _compute_bev_shape(pc_range, args.bev_resolution)

    calibrations = load_calibrations(raw_root / 'calibration_data')
    vehicle_root = raw_root / 'vehicle_data'

    box3d_dir = raw_root / 'box_3d_annotations'
    box2d_dir = raw_root / 'box_2d_annotations'
    inst_rgb_dir = raw_root / 'instance_annotations' / 'rgbLabels'
    sem_dir = raw_root / 'semantic_annotations' / 'gtLabels'
    motion_dir = raw_root / 'motion_annotations' / 'gtLabels'
    rgb_dir = raw_root / 'rgb_images'
    prev_dir = raw_root / 'previous_images'

    indices = sorted(p.stem for p in box3d_dir.glob('*.pkl'))
    if args.limit:
        indices = indices[:args.limit]

    infos = []
    instance_classes = set()
    instance_counter = Counter()
    bev_mask_dir = args.output.parent / 'synwoodscape_bev_masks'
    bev_mask_dir.mkdir(parents=True, exist_ok=True)
    bev_mask_shape = None

    for token in mmcv.track_iter_progress(indices):
        box3d_path = box3d_dir / f'{token}.pkl'
        dist_path = raw_root / 'distances_traveled' / f'{token}.txt'
        view_cache = {}
        for view in CAM_MAP.values():
            inst_rgb_path = inst_rgb_dir / f'{token}_{view}.png'
            sem_path = sem_dir / f'{token}_{view}.png'
            if inst_rgb_path.exists() and sem_path.exists():
                inst_rgb = mmcv.imread(inst_rgb_path)
                sem_map = mmcv.imread(sem_path, flag='unchanged').astype(np.uint8)
                view_cache[view] = (inst_rgb, sem_map)
        id_color = load_distances(dist_path)
        vehicle_pose = load_vehicle_pose(vehicle_root, token)

        with open(box3d_path, 'rb') as f:
            box_dict = pickle.load(f)

        gt_boxes = []
        gt_names = []

        for obj_id, corners in box_dict.items():
            corners = np.asarray(corners, dtype=np.float32)
            color = id_color.get(obj_id)
            if color is None:
                continue
            class_name = None
            for inst_rgb, sem_map in view_cache.values():
                mask = np.all(inst_rgb == color, axis=-1)
                if not mask.any():
                    continue
                sem_id = majority(sem_map[mask])
                if sem_id is None:
                    continue
                raw_name = SEMANTIC_ID_TO_CLASS.get(sem_id, 'unlabeled')
                class_name = SEMANTIC_TO_DET.get(raw_name)
                if class_name is not None:
                    break
            if class_name is None:
                continue
            center, dims, yaw = corners_to_box(corners)
            width, length, height = dims
            gt_boxes.append([center[0], center[1], center[2],
                             width, length, height, yaw, 0.0, 0.0])
            gt_names.append(class_name)
            instance_counter[class_name] += 1
            instance_classes.add(class_name)

        if not gt_boxes:
            continue

        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_names = np.asarray(gt_names, dtype=object)
        gt_velocity = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)
        zeros = np.zeros((gt_boxes.shape[0],), dtype=np.int64)

        cams = {}
        ann_info = dict(
            mv_bboxes=[],
            mv_labels=[],
            semantic_paths={},
            motion_paths={},
            bbox_class_names=list(DETECTION_CLASSES),
        )

        ego_translation = vehicle_pose['translation']
        ego_rotation = vehicle_pose['rotation']
        ego_translation_list = ego_translation.tolist()
        ego_rotation_list = ego_rotation.tolist()
        frame_timestamp = int(vehicle_pose['timestamp'])

        for cam_name, suffix in CAM_MAP.items():
            img_path = rgb_dir / f'{token}_{suffix}.png'
            prev_path = prev_dir / f'{token}_{suffix}_prev.png'

            calib = calibrations.get(suffix)
            cam_height = calib['height'] if calib and calib['height'] else None
            cam_width = calib['width'] if calib and calib['width'] else None
            sensor_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            sensor_trans = np.zeros(3, dtype=np.float32)
            intrinsic = np.eye(3, dtype=np.float32)
            distortion = np.zeros(4, dtype=np.float32)
            cam_model = 'polynomial'
            radial_params = None

            if calib is not None:
                rot_carla = quaternion_wxyz_to_matrix(calib['quaternion'])
                sensor_rot = matrix_to_quaternion_wxyz(carla_to_fastbev_rotation(rot_carla))
                sensor_trans = carla_to_fastbev_translation(calib['translation'])
                intrinsic = calib['intrinsic'].astype(np.float32)
                distortion = calib['distortion'].astype(np.float32)
                cam_model = calib['model']
                radial_params = calib.get('radial_params')

            if cam_height is None or cam_width is None:
                img = mmcv.imread(img_path)
                cam_height, cam_width = img.shape[:2]

            box2d_path = box2d_dir / f'{token}_{suffix}.txt'
            bboxes, label_ids, label_names = [], [], []
            if box2d_path.exists():
                with box2d_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        cls_name = parts[0]
                        if cls_name not in DET_CLASS_TO_ID:
                            continue
                        x1, y1, x2, y2 = map(float, parts[2:6])
                        class_id = DET_CLASS_TO_ID[cls_name]
                        bboxes.append([x1, y1, x2, y2])
                        label_ids.append(class_id)
                        label_names.append(DETECTION_CLASSES[class_id])

            bbox_array = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
            label_id_array = np.asarray(label_ids, dtype=np.int64).reshape(-1)
            label_name_array = np.asarray(label_names, dtype=object)

            # 构建传感器->ego 4x4，再求逆得到 lidar(假设=ego)->cam 外参
            sensor2ego_mat = np.eye(4, dtype=np.float32)
            sensor2ego_mat[:3, :3] = quaternion_wxyz_to_matrix(sensor_rot)
            sensor2ego_mat[:3, 3] = sensor_trans
            lidar2cam = np.linalg.inv(sensor2ego_mat)  # 假设 lidar=ego

            cam_info = dict(
                data_path=str(img_path),
                prev_data_path=str(prev_path) if prev_path.exists() else '',
                height=int(cam_height),
                width=int(cam_width),
                timestamp=frame_timestamp,
                sensor2ego_rotation=sensor_rot.tolist(),
                sensor2ego_translation=sensor_trans.tolist(),
                ego2global_rotation=ego_rotation_list,
                ego2global_translation=ego_translation_list,
                sensor2lidar_rotation=np.eye(3, dtype=np.float32),
                sensor2lidar_translation=np.zeros(3, dtype=np.float32),
                cam_intrinsic=intrinsic,
                cam_distortion=distortion,
                cam_model=cam_model,
                cam_radial_params=radial_params,
                extrinsic=lidar2cam.tolist(),  # lidar(=ego) -> cam
            )
            cam_info['annos'] = dict(
                bbox=bbox_array,
                category_id=label_id_array,
                category_name=label_name_array,
            )
            cams[cam_name] = cam_info

            ann_info['mv_bboxes'].append(bbox_array)
            ann_info['mv_labels'].append(label_id_array)

            semantic_png = sem_dir / f'{token}_{suffix}.png'
            if semantic_png.exists():
                ann_info.setdefault('semantic_paths', {})[cam_name] = str(semantic_png)

            motion_png = motion_dir / f'{token}_{suffix}.png'
            if motion_png.exists():
                ann_info.setdefault('motion_paths', {})[cam_name] = str(motion_png)

        ann_info['bboxes'] = [np.asarray(b, dtype=np.float32).reshape(-1, 4)
                              for b in ann_info['mv_bboxes']]
        ann_info['labels'] = [np.asarray(l, dtype=np.int64).reshape(-1)
                              for l in ann_info['mv_labels']]
        ann_info['bev_seg_classes'] = BEV_CLASS_NAMES

        bev_sem_path = sem_dir / f'{token}_BEV.png'
        if bev_sem_path.exists():
            bev_sem_map = mmcv.imread(bev_sem_path, flag='unchanged').astype(np.uint8)
            if args.bev_center:
                bev_sem_map = center_bev_map(bev_sem_map)
            if args.bev_rotate:
                # 将 BEV 图绕 ego yaw 旋转到车体坐标
                quat = vehicle_pose['rotation']
                rot = quaternion_wxyz_to_matrix(quat)
                yaw_rad = math.atan2(rot[1, 0], rot[0, 0])
                bev_sem_map = rotate_bev_map(bev_sem_map, angle_deg=-math.degrees(yaw_rad))
            bev_mask = convert_bev_semantic(bev_sem_map)
            bev_mask = align_bev_mask(
                bev_mask,
                target_shape=bev_target_shape,
                flip_y=args.bev_flip_y)
            np.save(bev_mask_dir / f'{token}.npy', bev_mask)
            ann_info['gt_bev_seg'] = str(bev_mask_dir / f'{token}.npy')
            if bev_mask_shape is None:
                bev_mask_shape = list(bev_mask.shape)

        info = dict(
            token=token,
            scene_name='synwoodscape',
            frame_id=int(token),
            timestamp=frame_timestamp,
            cams=cams,
            ann_info=ann_info,
            gt_boxes=gt_boxes,
            gt_names=gt_names,
            gt_velocity=gt_velocity,
            num_lidar_pts=zeros,
            num_radar_pts=zeros,
            valid_flag=np.ones_like(zeros, dtype=bool),
            lidar_path='',
            sweeps=[],
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=[0.0, 0.0, 0.0, 1.0],
            ego2global_translation=ego_translation_list,
            ego2global_rotation=ego_rotation_list,
            prev=None,
            next=None,
            velo=np.zeros(2, dtype=np.float32),
        )
        infos.append(info)

    bbox_class_list = list(DETECTION_CLASSES)
    class2id = DET_CLASS_TO_ID
    for info in infos:
        info['ann_info']['mv_bboxes'] = [np.asarray(b, dtype=np.float32).reshape(-1, 4)
                                         for b in info['ann_info'].get('mv_bboxes', [])]
        info['ann_info']['mv_labels'] = [np.asarray(l, dtype=np.int64).reshape(-1)
                                         for l in info['ann_info'].get('mv_labels', [])]
        for cam_name, cam_meta in info['cams'].items():
            cam_ann = cam_meta.get('annos')
            if cam_ann is None:
                cam_ann = dict()
                info['cams'][cam_name]['annos'] = cam_ann
            bbox_arr = cam_ann.get('bbox')
            if bbox_arr is None:
                bbox_arr = np.zeros((0, 4), dtype=np.float32)
            else:
                bbox_arr = np.asarray(bbox_arr, dtype=np.float32).reshape(-1, 4)
            id_arr = cam_ann.get('category_id')
            if id_arr is None:
                name_arr = np.asarray(cam_ann.get('category_name', []), dtype=object)
                id_arr = np.array([class2id.get(name, -1) for name in name_arr], dtype=np.int64)
            else:
                id_arr = np.asarray(id_arr, dtype=np.int64).reshape(-1)
                name_arr = cam_ann.get('category_name')
                if name_arr is None or len(name_arr) != len(id_arr):
                    name_arr = np.asarray(
                        [DETECTION_CLASSES[idx] if 0 <= idx < len(DETECTION_CLASSES) else 'unknown'
                         for idx in id_arr],
                        dtype=object)
                else:
                    name_arr = np.asarray(name_arr, dtype=object)
            cam_ann['bbox'] = bbox_arr
            cam_ann['category_name'] = name_arr
            cam_ann['category_id'] = id_arr

    instance_class_list = list(DETECTION_CLASSES)
    bbox_class_list = list(DETECTION_CLASSES)
    camera_models_meta = {
        cam_name: calibrations.get(suffix, {}).get('model', 'polynomial')
        for cam_name, suffix in CAM_MAP.items()
    }
    metadata = dict(
        version='synwoodscape-v0.1.1',
        dataset='SynWoodScape',
        camera_types=list(CAM_MAP.keys()),
        split=args.split,
        instance_classes=instance_class_list,
        bbox2d_classes=bbox_class_list,
        bev_seg=dict(
            class_names=BEV_CLASS_NAMES,
            mask_shape=bev_mask_shape or []
        ),
        camera_models=camera_models_meta,
        det_class_map=SEMANTIC_TO_DET,
    )
    mmcv.dump(dict(metadata=metadata, infos=infos), args.output)
    print(f'写出 SynWoodScape info 至 {args.output}，样本数 {len(infos)}')
    print('3D 类别统计 TOP10:', instance_counter.most_common(10))


if __name__ == '__main__':
    import pickle  # noqa: E402
    main()
