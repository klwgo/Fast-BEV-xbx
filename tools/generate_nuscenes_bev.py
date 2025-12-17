# -*- coding: utf-8 -*-
"""
使用 nuScenes 高精地图生成 BEV 语义/标线/障碍掩码，供 Fast-BEV HeadA/B/D 训练。

标签定义（可根据需要调整）：
- drivable：基于 drivable_area 多边形，二分类（0=非可行驶，1=可行驶）。
- marking：基于 lane 左/右边界绘制的细线，二分类（0=背景，1=标线）。
- obstacle：基于 3D 检测框在地面的投影，二分类（0=背景，1=障碍）。

分辨率与范围：从配置读取 `voxel_size` 与 `n_voxels`，生成网格范围：
    x_range = [-0.5*H*vs, 0.5*H*vs], y_range = [-0.5*W*vs, 0.5*W*vs]
其中 H/W 分别取 `n_voxels[0]/n_voxels[1]`。

输出：
- 每帧生成 npy 掩码：drivable_mask、marking_mask、obstacle_mask。
- 保存路径形如 `out_dir/<sample_token>_drivable.npy` 等。
- 可选将掩码路径写回新的 info pkl（仅附加到 ann_info 中）。

用法示例：
python tools/generate_nuscenes_bev.py \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --config configs/woodscape/fastbev_syn_headabc.py \
  --out-dir work_dirs/nuscenes_bev_masks \
  --save-pkl work_dirs/nuscenes_infos_train_bev.pkl \
  --split train
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import mmcv
import numpy as np
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def load_cfg_shape(cfg_path):
    cfg = mmcv.Config.fromfile(cfg_path, import_custom_modules=False)
    voxel_size = cfg.model.get('voxel_size', [[0.35, 0.35, 1.0]])[0]
    n_voxels = cfg.model.get('n_voxels', [[675, 650, 6]])[0]
    vs = float(voxel_size[0])
    H, W = int(n_voxels[0]), int(n_voxels[1])
    x_range = (-0.5 * H * vs, 0.5 * H * vs)
    y_range = (-0.5 * W * vs, 0.5 * W * vs)
    return vs, H, W, x_range, y_range


def transform_points(points, pose):
    """全局坐标 -> 车体坐标."""
    rot = Quaternion(pose['rotation']).rotation_matrix
    trans = np.asarray(pose['translation'], dtype=np.float32)
    pts = points - trans.reshape(1, 3)
    return pts @ rot.T


def world_to_grid(points, x_range, y_range, vs, H, W):
    """车体系下点 -> 栅格坐标 (row, col)."""
    x, y = points[:, 0], points[:, 1]  # x 前进, y 左
    x_max = x_range[1]
    y_min = y_range[0]
    row = (x_max - x) / vs
    col = (y - y_min) / vs
    rc = np.stack([row, col], axis=1)
    mask = (
        (row >= 0)
        & (row < H)
        & (col >= 0)
        & (col < W)
    )
    return rc, mask


def draw_polygons(mask, polygons, pose, x_range, y_range, vs, value=1):
    for poly in polygons:
        if poly is None:
            continue
        pts = np.asarray(poly.exterior.coords, dtype=np.float32)
        pts_ego = transform_points(np.pad(pts, ((0, 0), (0, 1)), constant_values=0), pose)[:, :2]
        rc, valid = world_to_grid(
            np.concatenate([pts_ego, np.zeros((pts_ego.shape[0], 1), dtype=np.float32)], axis=1),
            x_range,
            y_range,
            vs,
            mask.shape[0],
            mask.shape[1],
        )
        if valid.sum() < 3:
            continue
        poly_rc = rc[valid][:, ::-1].astype(np.int32)  # (row,col)->(y,x) for cv2
        cv2.fillPoly(mask, [poly_rc], value)


def draw_lane_markings(mask, lane_records, nusc_map, pose, x_range, y_range, vs, thickness=2):
    for rec in lane_records:
        for key in ['left_lane_divider_segments', 'right_lane_divider_segments']:
            seg_tokens = rec.get(key, [])
            for seg in seg_tokens:
                line = nusc_map.extract_line(seg)
                if line is None:
                    continue
                coords = np.asarray(line.coords, dtype=np.float32)
                pts = np.pad(coords, ((0, 0), (0, 1)), constant_values=0)
                pts_ego = transform_points(pts, pose)[:, :2]
                rc, valid = world_to_grid(
                    np.concatenate([pts_ego, np.zeros((pts_ego.shape[0], 1), dtype=np.float32)], axis=1),
                    x_range,
                    y_range,
                    vs,
                    mask.shape[0],
                    mask.shape[1],
                )
                if valid.sum() < 2:
                    continue
                poly_rc = rc[valid][:, ::-1].astype(np.int32)
                cv2.polylines(mask, [poly_rc], isClosed=False, color=1, thickness=thickness)


def draw_obstacles(mask, boxes, pose, x_range, y_range, vs):
    rot = Quaternion(pose['rotation']).rotation_matrix
    trans = np.asarray(pose['translation'], dtype=np.float32)
    x_max = x_range[1]
    y_min = y_range[0]
    for box in boxes:
        corners = box.corners().T  # (8,3) in global
        pts = (corners - trans.reshape(1, 3)) @ rot.T
        pts_bev = pts[:, :2]
        # footprint
        footprint = pts_bev[[0, 1, 2, 3], :]
        rc = np.stack(
            [
                (x_max - footprint[:, 0]) / vs,
                (footprint[:, 1] - y_min) / vs,
            ],
            axis=1,
        )
        if np.any(np.isnan(rc)):
            continue
        poly = rc[:, ::-1].astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)


def main():
    parser = argparse.ArgumentParser(description="生成 nuScenes BEV 语义/标线/障碍掩码")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="train", choices=["train", "val", "trainval"])
    parser.add_argument("--config", default="configs/woodscape/fastbev_syn_headabc.py")
    parser.add_argument("--out-dir", default="work_dirs/nuscenes_bev_masks")
    parser.add_argument("--save-pkl", default=None, help="若指定则保存新的 info pkl，并写入 bev_seg 路径")
    parser.add_argument("--lane-thickness", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true", default=True, help="已生成的样本跳过，便于断点续跑")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vs, H, W, x_range, y_range = load_cfg_shape(args.config)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # 官方 split 场景列表
    split_scenes = create_splits_scenes()[f"{args.split}"]
    scene_tokens = set([s["token"] for s in nusc.scene if s["name"] in split_scenes])

    existing_infos = []
    if args.save_pkl and Path(args.save_pkl).exists():
        try:
            with open(args.save_pkl, "rb") as f:
                prev = pickle.load(f)
            existing_infos = prev.get("infos", [])
            print(f"Loaded existing infos: {len(existing_infos)}")
        except Exception as e:
            print(f"[warn] load existing pkl failed: {e}")

    new_infos = []
    # 预过滤样本，跳过已存在掩码的 token
    to_process = []
    for sample in nusc.sample:
        if sample["scene_token"] not in scene_tokens:
            continue
        # 取 keyframe 的 FRONT 图像对应 ego pose
        token = sample["token"]
        if args.skip_existing:
            drv_path = Path(args.out_dir) / f"{token}_drivable.npy"
            obs_path = Path(args.out_dir) / f"{token}_obstacle.npy"
            if drv_path.exists() and obs_path.exists():
                continue
        to_process.append(sample)

    for sample in mmcv.track_iter_progress(to_process):
        sample_token = sample["token"]
        sd_front = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = nusc.get("ego_pose", sd_front["ego_pose_token"])
        map_name = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        nusc_map = NuScenesMap(dataroot=args.dataroot, map_name=map_name)

        drivable_mask = np.zeros((H, W), dtype=np.uint8)
        marking_mask = np.zeros((H, W), dtype=np.uint8)
        obstacle_mask = np.zeros((H, W), dtype=np.uint8)

        # drivable 多边形
        drivable_polys = []
        for d in nusc_map.drivable_area:
            for tok in d.get("polygon_tokens", []):
                poly = nusc_map.extract_polygon(tok)
                if poly is not None:
                    drivable_polys.append(poly)
        draw_polygons(drivable_mask, drivable_polys, ego_pose, x_range, y_range, vs, value=1)

        # 标线：nuScenes 无直接 lane divider token，暂置 0（可后续基于 lane 多边形边界生成）

        # 障碍：sample_annotation boxes 投影
        ann_tokens = sample["anns"]
        boxes = []
        for ann in ann_tokens:
            ann_rec = nusc.get("sample_annotation", ann)
            box = Box(
                center=ann_rec["translation"],
                size=ann_rec["size"],
                orientation=Quaternion(ann_rec["rotation"]),
                name=ann_rec["category_name"],
            )
            boxes.append(box)
        draw_obstacles(obstacle_mask, boxes, ego_pose, x_range, y_range, vs)

        np.save(Path(args.out_dir) / f"{sample_token}_drivable.npy", drivable_mask)
        np.save(Path(args.out_dir) / f"{sample_token}_marking.npy", marking_mask)
        np.save(Path(args.out_dir) / f"{sample_token}_obstacle.npy", obstacle_mask)

        new_infos.append(
            dict(
                token=sample_token,
                bev_seg_path=str(Path(args.out_dir) / f"{sample_token}_drivable.npy"),
                bev_marking_path=str(Path(args.out_dir) / f"{sample_token}_marking.npy"),
                bev_obstacle_path=str(Path(args.out_dir) / f"{sample_token}_obstacle.npy"),
                bev_seg_classes=["non_drivable", "drivable"],
            )
        )

    if args.save_pkl:
        all_infos = existing_infos + new_infos
        with open(args.save_pkl, "wb") as f:
            pickle.dump(dict(infos=all_infos, metadata=dict(version=args.version)), f)
        print(f"Saved info to {args.save_pkl}, new={len(new_infos)}, total={len(all_infos)}")


if __name__ == "__main__":
    main()
