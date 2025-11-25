#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用原始 Syn/WoodScape info 中的标定数据，基于 fisheye LUT 的方式
生成多相机 BEV 覆盖图。
"""

import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet.datasets import build_dataset

import mmdet3d.datasets.woodscape_dataset  # noqa: F401

CAM_SUFFIX = {
    'CAM_FRONT': 'FV',
    'CAM_FRONT_LEFT': 'MVL',
    'CAM_FRONT_RIGHT': 'MVR',
    'CAM_BACK': 'RV',
}

DEPTH_MIN = 0.5
DEPTH_MAX = 80.0
GROUND_FALLBACK_MAX = 15.0
GROUND_DIR_THRESHOLD = -0.7  # only consider rays heading downward
GROUND_HEIGHT_LIMIT = 0.8  # tolerance when comparing actual depth vs ground intersection
GROUND_INTERSECT_TOL = 1.5


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize BEV coverage via fisheye LUT')
    parser.add_argument('config', help='config path')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--level', type=int, default=0, help='voxel level if multi-scale')
    parser.add_argument('--ground-z', type=float, default=0.0, help='采样颜色时假设的地面高度')
    parser.add_argument('--out-dir', default='work_dirs/bev_stitch_vis')
    return parser.parse_args()


def build_split_dataset(cfg, split):
    data_cfg = deepcopy(cfg.data[split])
    test_pipeline = deepcopy(cfg.data.get('test', {}).get('pipeline'))
    if isinstance(data_cfg, dict) and data_cfg.get('type') == 'ClassBalancedDataset':
        data_cfg['test_mode'] = True
        data_cfg['dataset'] = deepcopy(data_cfg['dataset'])
        inner = data_cfg['dataset']
        inner['test_mode'] = True
        if test_pipeline is not None:
            inner['pipeline'] = test_pipeline
    else:
        data_cfg['test_mode'] = True
        if test_pipeline is not None:
            data_cfg['pipeline'] = test_pipeline
    return build_dataset(data_cfg)


def select_voxel_params(cfg_model, level):
    vox_cfg = cfg_model.get('n_voxels')
    size_cfg = cfg_model.get('voxel_size')
    if not vox_cfg or not size_cfg:
        raise KeyError('model 中缺少 n_voxels/voxel_size')
    if isinstance(vox_cfg[0], (list, tuple)):
        n_voxels = np.array(vox_cfg[level], dtype=np.int64)
    else:
        n_voxels = np.array(vox_cfg, dtype=np.int64)
    if isinstance(size_cfg[0], (list, tuple)):
        voxel_size = np.array(size_cfg[level], dtype=np.float32)
    else:
        voxel_size = np.array(size_cfg, dtype=np.float32)
    return n_voxels, voxel_size


def sanitize_distortion(raw_dist):
    """SynWoodScape 的 k1~k4 字段实际记录焦距等信息，将其视为畸变会导致 LUT 崩溃。"""
    if raw_dist is None:
        return np.zeros(4, dtype=np.float32)
    arr = np.asarray(raw_dist, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(4, dtype=np.float32)
    if float(np.max(np.abs(arr))) > 10.0:
        return np.zeros(4, dtype=np.float32)
    padded = np.zeros(4, dtype=np.float32)
    valid = min(4, arr.size)
    padded[:valid] = arr[:valid]
    return padded


def invert_fisheye_theta(theta_d, distortion):
    if distortion is None:
        return theta_d
    coeffs = np.asarray(distortion, dtype=np.float32).reshape(-1)
    if coeffs.size == 0 or np.allclose(coeffs, 0):
        return theta_d
    padded = np.zeros(4, dtype=np.float32)
    padded[:min(4, coeffs.size)] = coeffs[:4]
    k1, k2, k3, k4 = padded
    theta = theta_d.copy()
    for _ in range(6):
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        poly = 1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        f = theta * poly - theta_d
        if np.max(np.abs(f)) < 1e-6:
            break
        theta3 = theta2 * theta
        theta5 = theta4 * theta
        theta7 = theta6 * theta
        deriv = poly + theta * (
            2 * k1 * theta + 4 * k2 * theta3 + 6 * k3 * theta5 + 8 * k4 * theta7
        )
        deriv = np.clip(deriv, 1e-6, None)
        theta = theta - f / deriv
    return theta


def quat_to_rotmat(quat):
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)


def build_calibrations_from_raw(raw_info, camera_names):
    rot_l2e = quat_to_rotmat(raw_info['lidar2ego_rotation'])
    trans_l2e = np.asarray(raw_info['lidar2ego_translation'], dtype=np.float32)

    intr_list = []
    extr_list = []
    dist_list = []
    models = []
    heights = []
    widths = []

    for name in camera_names:
        cam = raw_info['cams'][name]
        intr_list.append(torch.tensor(cam['cam_intrinsic'], dtype=torch.float32))
        heights.append(int(cam.get('height', 0)))
        widths.append(int(cam.get('width', 0)))

        rot_cam2ego = quat_to_rotmat(cam.get('sensor2ego_rotation', [1, 0, 0, 0]))
        trans_cam2ego = np.asarray(cam.get('sensor2ego_translation', [0, 0, 0]), dtype=np.float32)
        rot_ego2cam = rot_cam2ego.T
        trans_ego2cam = -rot_ego2cam @ trans_cam2ego

        rot_l2c = rot_ego2cam @ rot_l2e
        trans_l2c = rot_ego2cam @ trans_l2e + trans_ego2cam

        extr = np.eye(4, dtype=np.float32)
        extr[:3, :3] = rot_l2c
        extr[:3, 3] = trans_l2c
        extr_list.append(torch.tensor(extr, dtype=torch.float32))

        dist_list.append(sanitize_distortion(cam.get('cam_distortion')))
        models.append(cam.get('cam_model', 'polynomial'))

    calibrations = prepare_calibrations(
        intrinsic=torch.stack(intr_list, dim=0),
        extrinsics=extr_list,
        distortion=dist_list,
        model_per_cam=models,
    )
    return calibrations, int(max(heights)), int(max(widths))


def resolve_data_path(path_str, data_root=None):
    path = Path(path_str)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        rel = path
        candidates.append(REPO_ROOT / rel)
        candidates.append(Path.cwd() / rel)
        if data_root:
            root = Path(data_root)
            if not root.is_absolute():
                root = REPO_ROOT / root
            candidates.append(root / rel)
            candidates.append(root / rel.name)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def load_camera_image(cam_meta, data_root):
    img_path = resolve_data_path(cam_meta.get('data_path', ''), data_root)
    img = mmcv.imread(str(img_path), flag='color')
    if img is None:
        raise FileNotFoundError(f'无法读取图像 {img_path}')
    return mmcv.bgr2rgb(img), img_path


def load_depth_map(token, cam_name, img_path):
    suffix = CAM_SUFFIX.get(cam_name)
    if suffix is None:
        raise KeyError(f'未知相机 {cam_name}, 缺少深度映射')
    dataset_dir = img_path.parents[1]
    depth_path = dataset_dir / 'depth_maps' / 'raw_data' / f'{token}_{suffix}.npy'
    if not depth_path.exists():
        raise FileNotFoundError(f'无法读取深度图 {depth_path}')
    depth = np.load(str(depth_path))
    return depth.astype(np.float32)


def prepare_depth_samples(depth, image, intrinsic, distortion):
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u, v)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_prime = (grid_u - cx) / fx
    y_prime = (grid_v - cy) / fy
    theta_d = np.sqrt(x_prime ** 2 + y_prime ** 2)
    theta = invert_fisheye_theta(theta_d, distortion)
    phi = np.arctan2(y_prime, x_prime)
    sin_theta = np.sin(theta)
    dir_x = sin_theta * np.cos(phi)
    dir_y = sin_theta * np.sin(phi)
    dir_z = np.cos(theta)
    dirs = np.stack([dir_x, dir_y, dir_z], axis=-1).reshape(-1, 3).astype(np.float32)
    depth_flat = depth.reshape(-1).astype(np.float32)
    colors = image.reshape(-1, 3).astype(np.uint8)
    return dirs, depth_flat, colors


def get_cam2lidar_transform(raw_info, cam_name):
    rot_l2e = quat_to_rotmat(raw_info['lidar2ego_rotation'])
    trans_l2e = np.asarray(raw_info['lidar2ego_translation'], dtype=np.float32)
    cam = raw_info['cams'][cam_name]
    rot_c2e = quat_to_rotmat(cam.get('sensor2ego_rotation', [1, 0, 0, 0]))
    trans_c2e = np.asarray(cam.get('sensor2ego_translation', [0, 0, 0]), dtype=np.float32)
    rot = rot_l2e.T @ rot_c2e
    trans = rot_l2e.T @ (trans_c2e - trans_l2e)
    return rot.astype(np.float32), trans.astype(np.float32)


def scatter_points_to_bev(points, colors, n_voxels, pc_range):
    nx, ny = int(n_voxels[0]), int(n_voxels[1])
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    valid = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    if not np.any(valid):
        return np.zeros((ny, nx, 3), dtype=np.uint8), np.zeros((ny, nx), dtype=bool)
    pts = points[valid]
    cols = colors[valid]
    scale_x = nx / (x_max - x_min)
    scale_y = ny / (y_max - y_min)
    x_idx = np.floor((pts[:, 0] - x_min) * scale_x).astype(np.int32)
    y_idx = np.floor((pts[:, 1] - y_min) * scale_y).astype(np.int32)
    x_idx = np.clip(x_idx, 0, nx - 1)
    y_idx = np.clip(y_idx, 0, ny - 1)
    depth_metric = np.linalg.norm(pts[:, :2], axis=1)
    lin_idx = y_idx * nx + x_idx
    order = np.argsort(depth_metric)
    lin_sorted = lin_idx[order]
    uniq, first = np.unique(lin_sorted, return_index=True)
    chosen = order[first]
    flat_colors = np.zeros((ny * nx, 3), dtype=np.uint8)
    flat_mask = np.zeros(ny * nx, dtype=bool)
    flat_colors[lin_idx[chosen]] = cols[chosen]
    flat_mask[lin_idx[chosen]] = True
    return flat_colors.reshape(ny, nx, 3), flat_mask.reshape(ny, nx)


def fill_bev_gaps(color_map, mask, inpaint_radius=3, erode_iters=0):
    missing = ~mask
    if not np.any(missing):
        return color_map, mask
    mask_uint8 = missing.astype(np.uint8) * 255
    if erode_iters > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=erode_iters)
    color_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(color_bgr, mask_uint8, inpaint_radius, cv2.INPAINT_TELEA)
    filled = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    return filled.astype(np.uint8), mask


def load_bev_mask(raw_info, data_root):
    bev = (raw_info.get('ann_info') or {}).get('gt_bev_seg')
    if bev is None:
        return None
    path = resolve_data_path(bev, data_root)
    if path.suffix == '.npy':
        return np.load(path)
    return plt.imread(path)


def summarize_coverage(bev_masks, camera_names):
    total = bev_masks[0].size if bev_masks else 0
    print('====== BEV 覆盖统计 ======')
    for name, mask in zip(camera_names, bev_masks):
        covered = int(np.count_nonzero(mask))
        ratio = covered / max(total, 1)
        print(f'[{name:<15}] {covered}/{total} ({ratio:.1%})')
    combined = int(np.count_nonzero(np.any(bev_masks, axis=0)))
    print(f'[Combined       ] {combined}/{total} ({combined / max(total, 1):.1%})')
    print('==========================')


def _prepare_gt_display(gt_mask):
    if gt_mask is None:
        return None
    mask = np.asarray(gt_mask)
    if mask.ndim == 3 and mask.shape[2] not in (3, 4):
        return np.argmax(mask, axis=-1)
    if mask.ndim == 3 and mask.shape[2] in (3, 4):
        return mask
    if mask.ndim == 3 and mask.shape[0] in (3, 4):
        return np.argmax(mask, axis=0)
    return mask.squeeze()


def plot_stitching(cam_colors, fused_color, gt_mask, cam_names, pc_range, out_path):
    gt_display = _prepare_gt_display(gt_mask)
    panels = [(f'{name} Stitch', img) for name, img in zip(cam_names, cam_colors)]
    panels.append(('Fused Stitch', fused_color))
    if gt_display is not None:
        panels.append(('GT BEV', gt_display))
    cols = 3
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    extent = [pc_range[0], pc_range[3], pc_range[1], pc_range[4]]
    for idx, (title, mask) in enumerate(panels):
        ax = axes[idx // cols, idx % cols]
        if mask.ndim == 3 and mask.shape[2] in (3, 4):
            ax.imshow(mask, origin='lower', extent=extent)
        else:
            ax.imshow(mask, origin='lower', extent=extent, cmap='viridis')
        ax.set_title(title, fontproperties='SimHei')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, linewidth=0.6, color='#cccccc', alpha=0.6)
    for rest in range(len(panels), rows * cols):
        axes[rest // cols, rest % cols].axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = build_split_dataset(cfg, args.split)
    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    raw_idx = args.index % len(base_dataset.data_infos)
    raw_info = base_dataset.data_infos[raw_idx]
    camera_names = getattr(base_dataset, 'camera_types', None)
    if camera_names is None:
        camera_names = sorted(raw_info['cams'].keys())
    else:
        camera_names = list(camera_names)
    n_voxels, voxel_size = select_voxel_params(cfg.model, args.level)
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float32)
    bev_colors = []
    bev_masks = []
    fused_color = np.zeros((int(n_voxels[1]), int(n_voxels[0]), 3), dtype=np.uint8)
    fused_mask = np.zeros((int(n_voxels[1]), int(n_voxels[0])), dtype=bool)
    data_root = getattr(base_dataset, 'data_root', None)
    for cam_name in camera_names:
        cam_meta = raw_info['cams'][cam_name]
        image, img_path = load_camera_image(cam_meta, data_root)
        depth = load_depth_map(raw_info['token'], cam_name, img_path)
        intrinsic = np.asarray(cam_meta['cam_intrinsic'], dtype=np.float32)
        distortion = sanitize_distortion(cam_meta.get('cam_distortion'))
        dirs_cam, depth_flat, colors_flat = prepare_depth_samples(depth, image, intrinsic, distortion)
        rot_cam2lidar, trans_cam2lidar = get_cam2lidar_transform(raw_info, cam_name)
        dirs_lidar = dirs_cam @ rot_cam2lidar.T
        cam_origin = trans_cam2lidar.astype(np.float32)
        dir_z_all = dirs_lidar[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_ground_all = (args.ground_z - cam_origin[2]) / dir_z_all
        valid_depth_mask = (
            np.isfinite(depth_flat)
            & (depth_flat > DEPTH_MIN)
            & (depth_flat < DEPTH_MAX)
        )
        ground_hit_mask = (
            valid_depth_mask
            & (dir_z_all < GROUND_DIR_THRESHOLD)
            & np.isfinite(t_ground_all)
            & (t_ground_all > 0)
            & (np.abs(depth_flat - t_ground_all) < GROUND_INTERSECT_TOL)
        )
        points_list = []
        colors_list = []
        if np.any(ground_hit_mask):
            pts = cam_origin + dirs_lidar[ground_hit_mask] * t_ground_all[ground_hit_mask][:, None]
            points_list.append(pts)
            colors_list.append(colors_flat[ground_hit_mask])
        fallback_mask = (
            (~valid_depth_mask)
            & (dir_z_all < GROUND_DIR_THRESHOLD)
            & np.isfinite(t_ground_all)
            & (t_ground_all > 0)
            & (t_ground_all < GROUND_FALLBACK_MAX)
        )
        if np.any(fallback_mask):
            pts_fb = cam_origin + dirs_lidar[fallback_mask] * t_ground_all[fallback_mask][:, None]
            points_list.append(pts_fb)
            colors_list.append(colors_flat[fallback_mask])
        if not points_list:
            bev_colors.append(np.zeros((int(n_voxels[1]), int(n_voxels[0]), 3), dtype=np.uint8))
            bev_masks.append(np.zeros((int(n_voxels[1]), int(n_voxels[0])), dtype=bool))
            continue
        points_lidar = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)
        color_map, cover_mask = scatter_points_to_bev(points_lidar, colors, n_voxels, pc_range)
        bev_colors.append(color_map)
        bev_masks.append(cover_mask)
        assign = cover_mask & (~fused_mask)
        fused_color[assign] = color_map[assign]
        fused_mask[assign] = True
    summarize_coverage(bev_masks, camera_names)
    gt_mask = load_bev_mask(raw_info, getattr(base_dataset, 'data_root', None))
    fused_color, _ = fill_bev_gaps(fused_color, fused_mask, inpaint_radius=2, erode_iters=1)
    out_path = Path(args.out_dir) / f'bev_stitch_{raw_idx:05d}.png'
    plot_stitching(bev_colors, fused_color, gt_mask, camera_names, pc_range, out_path)
    print(f'[BEV Stitch] saved -> {out_path}')


if __name__ == '__main__':
    main()
