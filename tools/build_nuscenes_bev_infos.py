# -*- coding: utf-8 -*-
"""
根据已生成的 BEV 掩码目录和 nuScenes 元数据，生成可直接用于训练/验证的 info pkl（train/val 分开）。

要求：
- 掩码目录内已有 <sample_token>_drivable.npy（必需），可选 <sample_token>_marking.npy / <sample_token>_obstacle.npy。
- dataroot 下存在 nuScenes 完整目录（samples/、sweeps/、v1.0-trainval/*.json、maps/）。

输出：
- infos 列表按 split 过滤（train/val），仅保留有掩码的样本。
- metadata.version 记录输入的 version。

用法示例：
python tools/build_nuscenes_bev_infos.py \
  --dataroot /mnt/new_data/nus/v1.0-trainval_full \
  --version v1.0-trainval \
  --mask-dir work_dirs/nuscenes_bev_masks_full_v1 \
  --split train \
  --out work_dirs/nuscenes_infos_train_bev_v1.pkl
"""

import argparse
import pickle
from pathlib import Path

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


def main():
    parser = argparse.ArgumentParser(description="从 BEV 掩码目录生成 nuScenes train/val info pkl")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    split_scenes = set(create_splits_scenes()[args.split])

    infos = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        if scene["name"] not in split_scenes:
            continue
        token = sample["token"]
        drv_path = mask_dir / f"{token}_drivable.npy"
        if not drv_path.exists():
            continue
        mark_path = mask_dir / f"{token}_marking.npy"
        obs_path = mask_dir / f"{token}_obstacle.npy"
        info = dict(
            token=token,
            bev_seg_path=str(drv_path),
            bev_seg_classes=["non_drivable", "drivable"],
        )
        if mark_path.exists():
            info["bev_marking_path"] = str(mark_path)
        if obs_path.exists():
            info["bev_obstacle_path"] = str(obs_path)
        infos.append(info)

    payload = dict(infos=infos, metadata=dict(version=args.version, split=args.split))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {len(infos)} infos to {args.out} (split={args.split})")


if __name__ == "__main__":
    main()
