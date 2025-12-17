# -*- coding: utf-8 -*-
"""
可视化 nuScenes BEV GT 掩码（drivable / obstacle / marking）。

示例：
python tools/visualize_nuscenes_bev_gt.py \
  --pkl work_dirs/nuscenes_infos_val_bev_filtered_exist.pkl \
  --data-root . \
  --index 0 \
  --out work_dirs/nus_gt_vis_0.png
"""

import argparse
import os
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def load_info(pkl_path, idx):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos", [])
    if idx >= len(infos):
        raise IndexError(f"index {idx} out of range, total {len(infos)}")
    return infos[idx]


def load_mask(path):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return np.load(path)


def vis_masks(drivable, obstacle=None, marking=None, out="out.png"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Drivable")
    plt.axis("off")
    plt.imshow(drivable, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Obstacle")
    plt.axis("off")
    if obstacle is not None:
        plt.imshow(obstacle, cmap="gray")
    else:
        plt.text(0.5, 0.5, "None", ha="center", va="center")

    plt.subplot(1, 3, 3)
    plt.title("Marking")
    plt.axis("off")
    if marking is not None:
        plt.imshow(marking, cmap="gray")
    else:
        plt.text(0.5, 0.5, "None", ha="center", va="center")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="nuScenes bev info pkl")
    parser.add_argument("--data-root", default=".", help="prefix for relative paths")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="work_dirs/nus_gt_vis.png")
    args = parser.parse_args()

    info = load_info(args.pkl, args.index)
    ann = info.get("ann_info", {})

    drv_path = ann.get("gt_bev_seg")
    if drv_path and not os.path.isabs(drv_path):
        drv_path = os.path.join(args.data_root, drv_path)
    drv = load_mask(drv_path)
    if drv is None:
        raise FileNotFoundError(f"drivable mask not found: {drv_path}")
    # 若多通道，取第0通道
    if drv.ndim == 3:
        drv = drv[0]

    obs_path = ann.get("bev_obstacle_path")
    if obs_path and not os.path.isabs(obs_path):
        obs_path = os.path.join(args.data_root, obs_path)
    obs = load_mask(obs_path)
    if obs is not None and obs.ndim == 3:
        obs = obs[0]

    mark_path = ann.get("bev_marking_path")
    if mark_path and not os.path.isabs(mark_path):
        mark_path = os.path.join(args.data_root, mark_path)
    mark = load_mask(mark_path)
    if mark is not None and mark.ndim == 3:
        mark = mark[0]

    vis_masks(drv, obs, mark, out=args.out)


if __name__ == "__main__":
    main()
