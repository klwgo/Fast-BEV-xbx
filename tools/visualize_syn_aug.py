# -*- coding: utf-8 -*-
"""
可视化 SynWoodScape 训练管线增强后的多视角图像与 BEV 掩码。

用法：
python tools/visualize_syn_aug.py \
  --config configs/woodscape/fastbev_syn_headabc.py \
  --index 0 \
  --out-dir work_dirs/syn_aug_vis
"""

import argparse
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset


def to_numpy(img, norm_cfg):
    """img: (C,H,W) tensor -> uint8 HWC"""
    img = img.detach().cpu().numpy()
    img = img.transpose(1, 2, 0)
    mean = np.array(norm_cfg["mean"], dtype=np.float32)
    std = np.array(norm_cfg["std"], dtype=np.float32)
    img = img * std + mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def save_sample(sample, idx, out_dir, norm_cfg, cam_names):
    os.makedirs(out_dir, exist_ok=True)
    imgs = sample["img"].data  # list/stack of images (num_views, C, H, W)
    if isinstance(imgs, torch.Tensor):
        imgs = [imgs[i] for i in range(imgs.shape[0])]
    canvases = []
    for i, im in enumerate(imgs):
        arr = to_numpy(im, norm_cfg)
        canvases.append(arr)
        mmcv.imwrite(arr[:, :, ::-1], os.path.join(out_dir, f"{idx:05d}_{cam_names[i]}.png"))
    # BEV masks
    bev = []
    if "gt_drivable_mask" in sample:
        bev.append(sample["gt_drivable_mask"].data.cpu().numpy())
    if "gt_marking_mask" in sample:
        bev.append(sample["gt_marking_mask"].data.cpu().numpy())
    if "gt_marking_boundary" in sample:
        bev.append(sample["gt_marking_boundary"].data.cpu().numpy())
    for j, mask in enumerate(bev):
        mmcv.imwrite((mask > 0).astype(np.uint8) * 255, os.path.join(out_dir, f"{idx:05d}_mask{j}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", default="work_dirs/syn_aug_vis")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    norm_cfg = cfg.img_norm_cfg
    dataset = build_dataset(cfg.data.train)
    sample = dataset[args.index]
    cam_names = dataset.camera_types if hasattr(dataset, "camera_types") else [f"cam{i}" for i in range(len(sample["img"].data))]
    save_sample(sample, args.index, args.out_dir, norm_cfg, cam_names)
    print(f"Saved augmented sample to {args.out_dir}")


if __name__ == "__main__":
    main()
