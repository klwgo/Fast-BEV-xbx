# -*- coding: utf-8 -*-
"""
将 SynWoodScape 的 4 路鱼眼图像逆向投影到 BEV（假设地面平面 z=0），并可叠加 GT 掩码。

用法：
python tools/project_synwoodscape_bev.py \
  --pkl data/synwoodscape_infos_val.pkl \
  --data-root . \
  --index 0 \
  --out work_dirs/syn_bev_vis_0.png
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sanitize_distortion(raw_dist):
    """清洗畸变参数，防止异常值；最多取前 4 个参数。"""
    if raw_dist is None:
        return np.zeros(4, dtype=np.float64)
    arr = np.asarray(raw_dist, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros(4, dtype=np.float64)
    if float(np.max(np.abs(arr))) > 10.0:
        return np.zeros(4, dtype=np.float64)
    padded = np.zeros(4, dtype=np.float64)
    valid = min(4, arr.size)
    padded[:valid] = arr[:valid]
    return padded

def undistort_fisheye(img, K, D):
    """使用 fisheye 模型去畸变，失败回退 pinhole。"""
    h, w = img.shape[:2]
    K = np.asarray(K, dtype=np.float64)
    D = sanitize_distortion(D).reshape(-1, 1)
    try:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3, dtype=np.float64), K, (w, h), cv2.CV_16SC2
        )
        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undist, K
    except Exception as e:
        print(f"[warn] fisheye undistort failed ({e}), fallback to pinhole undistort.")
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        undist = cv2.undistort(img, K, D, None, newK)
        return undist, newK

def quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def load_info(pkl_path, idx):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["infos"][idx]


def bev_grid(cfg):
    vx, vy = cfg["voxel_size"]
    H, W = cfg["n_voxels"]
    x_min, y_min, _, x_max, y_max, _ = cfg["pc_range"]
    xs = x_min + (np.arange(W) + 0.5) * vx
    # y 轴从上到下递减，保持与 GT 掩码方向一致
    ys = y_max - (np.arange(H) + 0.5) * vy
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")  # (H,W)
    pts = np.stack([grid_x, grid_y, np.zeros_like(grid_x)], axis=-1)  # (H,W,3)
    return pts


def project_cam(cam, cam_name, bev_pts, img, bev_rgb, bev_count, save_udist_dir=None):
    """使用鱼眼投影（等距）将 BEV 网格点投到图像，再采样回填。"""
    # 先去畸变并保存，随后使用去畸变后的图像与新内参投影
    K = np.array(cam["cam_intrinsic"], dtype=np.float32)
    dist = cam.get("cam_distortion", None)
    if dist is None:
        dist = cam.get("distortions", None)
    cam_model = cam.get("cam_model", "fisheye")
    if dist is not None:
        if cam_model.lower().startswith("fisheye"):
            img, K = undistort_fisheye(img, K, dist)
        else:
            dist = sanitize_distortion(dist)
            h, w = img.shape[:2]
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            img = cv2.undistort(img, K, dist, None, newK)
            K = newK.astype(np.float32)
    if save_udist_dir:
        os.makedirs(save_udist_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_udist_dir, f"{cam_name}_undist.png"), img)
    R_ce = quat_to_mat(cam["sensor2ego_rotation"])  # cam->ego
    t_ce = np.array(cam["sensor2ego_translation"], dtype=np.float32)
    R_ec = R_ce.T
    t_ec = -R_ec @ t_ce

    bev_h, bev_w = bev_pts.shape[:2]
    pts_world = bev_pts.reshape(-1, 3).T  # (3,N)
    pts_cam = R_ec @ pts_world + t_ec[:, None]  # (3,N)
    z = pts_cam[2]
    front_mask = z > 1e-3
    if not front_mask.any():
        return
    # 可选：忽略车身附近区域（半径 1.5m）
    xy_world = pts_world[:2].T
    car_mask = np.linalg.norm(xy_world, axis=1) < 1.5
    valid_mask = front_mask & (~car_mask)
    if not valid_mask.any():
        return
    pts_cam = pts_cam[:, valid_mask]
    idx_valid = np.nonzero(valid_mask)[0]

    # 鱼眼等距投影
    norm = np.linalg.norm(pts_cam, axis=0) + 1e-6
    dirs = pts_cam / norm
    x_d, y_d, z_d = dirs
    theta = np.arccos(z_d)
    phi = np.arctan2(y_d, x_d)
    r = theta.copy()
    if dist is not None:
        d = sanitize_distortion(dist)
        r2 = r * r
        r4 = r2 * r2
        r = r * (1 + d[0] * r2 + d[1] * r4)
    u = K[0, 0] * r * np.cos(phi) + K[0, 2]
    v = K[1, 1] * r * np.sin(phi) + K[1, 2]

    H_img, W_img = img.shape[:2]
    inside = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    if not inside.any():
        return
    u = u[inside].astype(np.int32)
    v = v[inside].astype(np.int32)
    idx_flat = idx_valid[inside]
    iy = idx_flat // bev_w
    ix = idx_flat % bev_w
    colors = img[v, u, ::-1]  # BGR->RGB
    depth = pts_cam[:, inside][2]  # z_cam after inside mask
    w = 1.0 / (depth + 1e-3)
    np.add.at(bev_rgb, (iy, ix), colors * w[:, None])
    np.add.at(bev_count, (iy, ix), w)


def overlay_mask(base, mask, color):
    out = base.copy()
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape != base.shape[:2]:
        if mask.T.shape == base.shape[:2]:
            mask = mask.T
        else:
            h = min(mask.shape[0], base.shape[0])
            w = min(mask.shape[1], base.shape[1])
            mask = mask[:h, :w]
            out = out[:h, :w]
    m = mask > 0
    out[m] = (0.3 * out[m] + 0.7 * np.array(color)).astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="data/synwoodscape_infos_val.pkl")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="work_dirs/syn_bev_vis_0.png")
    parser.add_argument("--save-undistort-dir", default=None, help="若指定，则保存每路相机的去畸变图像")
    args = parser.parse_args()

    cfg = dict(
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=[0.35, 0.35],
        n_voxels=[675, 650],
    )

    info = load_info(args.pkl, args.index)
    ann = info.get("ann_info", {})
    gt_path = ann.get("gt_bev_seg", None)
    gt_mask = None
    if gt_path:
        if not os.path.isabs(gt_path):
            gt_path = os.path.join(args.data_root, gt_path)
        if Path(gt_path).exists():
            gt_mask = np.load(gt_path)
            # 使用 GT 形状对齐网格尺寸
            H, W = gt_mask.shape[:2]
            cfg["n_voxels"] = [H, W]
            # 根据 pc_range 反推精确 voxel_size，保持范围不变
            x_min, y_min, _, x_max, y_max, _ = cfg["pc_range"]
            cfg["voxel_size"] = [
                (x_max - x_min) / float(W),
                (y_max - y_min) / float(H),
            ]

    bev_pts = bev_grid(cfg)
    H, W = cfg["n_voxels"]
    bev_rgb = np.zeros((H, W, 3), dtype=np.float32)
    bev_count = np.zeros((H, W), dtype=np.int32)

    for name, cam in info["cams"].items():
        img_path = cam["data_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(args.data_root, img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        project_cam(cam, name, bev_pts, img, bev_rgb, bev_count, args.save_undistort_dir)

    bev = np.zeros_like(bev_rgb, dtype=np.uint8)
    m = bev_count > 0
    if m.any():
        bev[m] = (bev_rgb[m] / bev_count[m, None]).astype(np.uint8)

    gt_overlay = None
    if gt_mask is not None:
        overlay = overlay_mask(bev, gt_mask, (0, 255, 0))
        gt_overlay = overlay

    fig, axs = plt.subplots(1, 2 if gt_overlay is not None else 1, figsize=(12, 6))
    if gt_overlay is None:
        axs = [axs]
    axs[0].imshow(bev)
    axs[0].set_title("Projected BEV (RGB)")
    axs[0].axis("off")
    if gt_overlay is not None:
        axs[1].imshow(gt_overlay)
        axs[1].set_title("Overlay with GT")
        axs[1].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
