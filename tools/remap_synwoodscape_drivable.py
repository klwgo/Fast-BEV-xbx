# -*- coding: utf-8 -*-
"""
将 SynWoodScape 的多类 BEV 掩码重新映射为二值可行驶掩码（0=非可行驶，1=可行驶），并生成新的 pkl。

用法示例：
python tools/remap_synwoodscape_drivable.py \
  --src data/synwoodscape_infos_train.pkl \
  --dst data/synwoodscape_infos_train_drivable.pkl \
  --out-dir work_dirs/syn_drivable_masks
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="原始 synwoodscape info pkl")
    parser.add_argument("--dst", required=True, help="输出 pkl")
    parser.add_argument(
        "--out-dir",
        default="work_dirs/syn_drivable_masks",
        help="二值掩码输出目录（相对工程根目录）",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pickle.load(open(args.src, "rb"))
    infos = data["infos"]

    # 定义可行驶类别名称（与 ann_info['bev_seg_classes'] 对应）
    drivable_names = {
        "road_surface",
        "free_space",
        "lane_marking",
        "parking_line",
        "ground",
    }

    updated_infos = []
    for info in tqdm(infos, desc="Remap"):
        ann = info.get("ann_info", {})
        classes = ann.get("bev_seg_classes", [])
        mask_path = ann.get("gt_bev_seg")
        if mask_path is None:
            updated_infos.append(info)
            continue
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(".", mask_path)
        if not os.path.exists(mask_path):
            updated_infos.append(info)
            continue
        mask = np.load(mask_path)
        # 多通道/单通道兼容，仅取整数标签
        if mask.ndim != 2:
            # 如果是 (C,H,W) one-hot，取 argmax
            mask = np.argmax(mask, axis=0)
        # 构建类别->idx
        name_to_idx = {n: i for i, n in enumerate(classes)}
        drv_idx = [name_to_idx[n] for n in drivable_names if n in name_to_idx]
        drv_binary = np.isin(mask, drv_idx).astype(np.uint8)
        # 保存新掩码，路径相对工程根目录
        token = info.get("token") or info.get("frame_id")
        out_path = out_dir / f"{token}_drivable.npy"
        np.save(out_path, drv_binary)
        # 更新 ann_info：覆盖 gt_bev_seg 为二值掩码
        new_ann = ann.copy()
        new_ann["gt_bev_seg"] = str(out_path)
        new_ann["bev_seg_classes"] = ["non_drivable", "drivable"]
        info = info.copy()
        info["ann_info"] = new_ann
        updated_infos.append(info)

    data["infos"] = updated_infos
    pickle.dump(data, open(args.dst, "wb"))
    print(f"Saved {args.dst}, infos={len(updated_infos)}; masks in {out_dir}")


if __name__ == "__main__":
    main()
