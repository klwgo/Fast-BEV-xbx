# -*- coding: utf-8 -*-
"""
单视角鱼眼畸变校正可视化脚本。

用法示例：
python tools/undistort_fisheye.py \
  --pkl data/synwoodscape_infos_train.pkl \
  --index 0 \
  --cam CAM_FRONT \
  --out work_dirs/undistort_demo/CAM_FRONT_00000.png
"""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


def sanitize_distortion(raw_dist):
    """清洗畸变参数，SynWoodScape 部分字段可能是噪声."""
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
    """使用 OpenCV fisheye 重映射，若失败则回退到 pinhole 模型."""
    h, w = img.shape[:2]
    K = np.asarray(K, dtype=np.float64)
    D = sanitize_distortion(D).reshape(-1, 1)
    try:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3, dtype=np.float64), K, (w, h), cv2.CV_16SC2
        )
        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undist
    except Exception as e:
        print(f"[warn] fisheye undistort failed ({e}), fallback to pinhole undistort.")
        undist = cv2.undistort(img, K, D)
        return undist


def main():
    parser = argparse.ArgumentParser(description="Fisheye 单视角畸变校正可视化")
    parser.add_argument("--pkl", required=True, help="info pkl 路径，如 data/synwoodscape_infos_train.pkl")
    parser.add_argument("--index", type=int, default=0, help="样本索引")
    parser.add_argument("--cam", default="CAM_FRONT", help="相机名，例如 CAM_FRONT/CAM_BACK 等")
    parser.add_argument("--out", default="undistort.png", help="输出图片路径")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    info = infos[args.index]
    cam_info = info["cams"][args.cam]
    img_path = cam_info["data_path"]
    K = cam_info.get("cam_intrinsic")
    if K is None:
        K = cam_info.get("intrinsics")
    D = cam_info.get("cam_distortion")
    if D is None:
        D = cam_info.get("distortions")
    cam_model = cam_info.get("cam_model", "polynomial")

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = undistort_fisheye(img, K, D)
    undist_rgb = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original {args.cam}")
    axes[0].axis("off")
    axes[1].imshow(undist_rgb)
    axes[1].set_title(f"Undistorted {args.cam}")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
