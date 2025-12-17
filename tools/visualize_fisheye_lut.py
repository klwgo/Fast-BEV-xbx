# -*- coding: utf-8 -*-
"""
可视化鱼眼 LUT 的覆盖范围。

默认读取 SynWoodScape info，并基于相机标定生成 LUT，输出每路相机在 BEV 网格上的可见区域。

用法示例：
python tools/visualize_fisheye_lut.py \
  --pkl data/synwoodscape_infos_train.pkl \
  --index 0 \
  --n-voxels 675 650 6 \
  --voxel-size 0.35 0.35 1.0 \
  --stride 8 \
  --out-dir work_dirs/lut_vis
"""

import argparse
import math
import os
import pickle
from pathlib import Path
import sys
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import torch

# 让脚本在仓库内可直接加载 fisheye_lut（不依赖 mmcv 安装）
REPO_ROOT = Path(__file__).resolve().parents[1]
FISHEYE_LUT_PATH = REPO_ROOT / "mmdet3d" / "models" / "utils" / "fisheye_lut.py"
spec = importlib.util.spec_from_file_location("fisheye_lut_module", FISHEYE_LUT_PATH)
fisheye_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(fisheye_module)
build_fisheye_lut = fisheye_module.build_fisheye_lut
prepare_calibrations = fisheye_module.prepare_calibrations

# 本地实现 get_points，避免依赖完整 FastBEV/ mmcv
@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    n_vox = torch.as_tensor(n_voxels, dtype=torch.float32)
    voxel = torch.as_tensor(voxel_size, dtype=torch.float32)
    orig = torch.as_tensor(origin, dtype=torch.float32)
    grid = torch.stack(
        torch.meshgrid(
            [
                torch.arange(int(n_vox[0])),
                torch.arange(int(n_vox[1])),
                torch.arange(int(n_vox[2])),
            ]
        )
    )
    new_origin = orig - n_vox / 2.0 * voxel
    points = grid * voxel.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def _load_info(pkl_path, index):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos", data)
    if index >= len(infos):
        raise IndexError(f"index {index} out of range (total {len(infos)})")
    return infos[index]


def _invert_cam_to_lidar(rot, trans):
    """将 cam->lidar 变换取逆，得到 lidar->cam 外参 4x4。"""
    R = np.asarray(rot, dtype=np.float32).reshape(3, 3)
    t = np.asarray(trans, dtype=np.float32).reshape(3, 1)
    R_inv = R.T
    t_inv = -R_inv @ t
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = R_inv
    ext[:3, 3] = t_inv[:, 0]
    return ext


def _build_calibrations(info):
    cams = info["cams"]
    cam_names = list(cams.keys())
    intrinsics = []
    extrinsics = []
    distortions = []
    models = []

    for name in cam_names:
        meta = cams[name]
        intrinsics.append(np.asarray(meta["cam_intrinsic"], dtype=np.float32))
        distortions.append(np.asarray(meta.get("cam_distortion", []), dtype=np.float32))
        models.append(meta.get("cam_model", "polynomial"))
        ext = _invert_cam_to_lidar(
            meta["sensor2lidar_rotation"], meta["sensor2lidar_translation"]
        )
        extrinsics.append(ext)

    intr_stack = np.stack(intrinsics, axis=0)
    return prepare_calibrations(
        intrinsic=torch.as_tensor(intr_stack),
        extrinsics=[torch.as_tensor(e) for e in extrinsics],
        distortion=[torch.as_tensor(d) for d in distortions],
        model_per_cam=models,
    )


def visualize(valid_mask, out_dir):
    """valid_mask: [n_cams, vx, vy, vz] bool."""
    os.makedirs(out_dir, exist_ok=True)
    n_cams = valid_mask.shape[0]
    cover_any = valid_mask.any(dim=-1)  # [n_cams, vx, vy]
    union = cover_any.any(dim=0).numpy()

    for i in range(n_cams):
        img = cover_any[i].numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(img, origin="lower", cmap="gray")
        plt.title(f"cam {i} coverage")
        plt.tight_layout()
        out_path = Path(out_dir) / f"cam_{i}_coverage.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {out_path}")

    plt.figure(figsize=(6, 6))
    plt.imshow(union, origin="lower", cmap="viridis")
    plt.title("Union coverage (all cams)")
    plt.tight_layout()
    union_path = Path(out_dir) / "coverage_union.png"
    plt.savefig(union_path, dpi=200)
    plt.close()
    print(f"Saved {union_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="info pkl 路径（含 cams 标定）")
    ap.add_argument("--index", type=int, default=0, help="样本索引")
    ap.add_argument(
        "--n-voxels",
        type=int,
        nargs=3,
        default=[675, 650, 6],
        help="BEV 体素数 [vx vy vz]",
    )
    ap.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        default=[0.35, 0.35, 1.0],
        help="体素尺寸 (m)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=8,
        help="特征下采样倍率（影响内参缩放）",
    )
    ap.add_argument(
        "--out-dir",
        default="work_dirs/lut_vis",
        help="可视化输出目录",
    )
    args = ap.parse_args()

    info = _load_info(args.pkl, args.index)
    calibrations = _build_calibrations(info)

    # 使用首个相机的分辨率，假设各视角一致
    first_cam = next(iter(info["cams"].values()))
    h_img, w_img = int(first_cam["height"]), int(first_cam["width"])
    height = math.ceil(h_img / float(args.stride))
    width = math.ceil(w_img / float(args.stride))

    n_voxels = torch.as_tensor(args.n_voxels, dtype=torch.float32)
    voxel_size = torch.as_tensor(args.voxel_size, dtype=torch.float32)
    origin = torch.zeros(3, dtype=torch.float32)
    points = get_points(
        n_voxels=n_voxels,
        voxel_size=voxel_size,
        origin=origin,
    )

    lut, valid = build_fisheye_lut(
        points=points,
        cameras=calibrations,
        height=height,
        width=width,
        stride=args.stride,
    )

    # valid: [n_cams, n_voxels] -> [n_cams, vx, vy, vz]
    vx, vy, vz = [int(x) for x in args.n_voxels]
    valid = valid.view(valid.shape[0], vx, vy, vz)
    visualize(valid, args.out_dir)


if __name__ == "__main__":
    main()
