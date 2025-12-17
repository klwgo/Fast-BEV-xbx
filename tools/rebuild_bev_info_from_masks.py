# -*- coding: utf-8 -*-
"""
根据已生成的 BEV 掩码目录重新生成 info pkl。

假设掩码文件命名为：
  <token>_drivable.npy
  <token>_marking.npy   （可选）
  <token>_obstacle.npy  （可选）

输出 pkl 结构：
{
    "infos": [
        {
            "token": "...",
            "bev_seg_path": "..._drivable.npy",
            "bev_marking_path": "..._marking.npy",     # 若存在
            "bev_obstacle_path": "..._obstacle.npy",   # 若存在
            "bev_seg_classes": ["non_drivable", "drivable"]
        },
        ...
    ],
    "metadata": {"version": "v1.0-trainval"}
}
"""

import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="从 BEV 掩码目录重建 info pkl")
    parser.add_argument("--mask-dir", required=True, help="掩码目录，含 *_drivable.npy 等")
    parser.add_argument("--out", required=True, help="输出 pkl 路径")
    parser.add_argument("--version", default="v1.0-trainval", help="数据版本记录到 metadata")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    infos = []
    for drv_file in mask_dir.glob("*_drivable.npy"):
        token = drv_file.name.replace("_drivable.npy", "")
        marking_file = mask_dir / f"{token}_marking.npy"
        obstacle_file = mask_dir / f"{token}_obstacle.npy"
        info = dict(
            token=token,
            bev_seg_path=str(drv_file),
            bev_seg_classes=["non_drivable", "drivable"],
        )
        if marking_file.exists():
            info["bev_marking_path"] = str(marking_file)
        if obstacle_file.exists():
            info["bev_obstacle_path"] = str(obstacle_file)
        infos.append(info)

    payload = dict(infos=infos, metadata=dict(version=args.version))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {len(infos)} infos to {args.out}")


if __name__ == "__main__":
    main()
