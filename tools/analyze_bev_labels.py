#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速统计 BEV/2D/3D 标注占比，用于排查正负样本极端不平衡问题。

示例：
    python tools/analyze_bev_labels.py --ann data/synwoodscape_infos_train.pkl
    python tools/analyze_bev_labels.py --ann data/synwoodscape_infos_val.pkl --limit 50
"""

import argparse
import os
import pickle
from collections import Counter, defaultdict

import numpy as np


def load_infos(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # 兼容 {'infos': [...]} 与直接 list
    if isinstance(data, dict) and "infos" in data:
        infos = data["infos"]
        meta = data.get("metadata", {})
    else:
        infos = data
        meta = {}
    return infos, meta


def analyze(args):
    infos, meta = load_infos(args.ann)
    bev_classes = meta.get("bev_seg", {}).get("class_names")
    bev_num_classes = len(bev_classes) if bev_classes else None

    bev_counter = np.zeros(bev_num_classes or 0, dtype=np.int64)
    bev_nonzero_frames = 0
    bev_missing = 0
    bev_shapes = Counter()

    mv_counts = []
    box3d_counts = []

    for idx, info in enumerate(infos):
        if args.limit and idx >= args.limit:
            break
        ann = info.get("ann_info", {})
        # ---------------- BEV mask ----------------
        bev_path = ann.get("gt_bev_seg")
        if bev_path is None:
            bev_missing += 1
        else:
            if not os.path.isabs(bev_path):
                bev_path = os.path.join(os.path.dirname(args.ann), "..", bev_path)
                bev_path = os.path.abspath(bev_path)
            if os.path.exists(bev_path):
                bev_mask = np.load(bev_path)
                if bev_num_classes is None:
                    bev_num_classes = int(bev_mask.max()) + 1
                    bev_counter = np.zeros(bev_num_classes, dtype=np.int64)
                flat = bev_mask.reshape(-1)
                bins = np.bincount(flat, minlength=bev_num_classes)
                bev_counter[: len(bins)] += bins
                if flat.any():
                    bev_nonzero_frames += 1
                bev_shapes[bev_mask.shape] += 1
            else:
                bev_missing += 1
        # ---------------- 2D bbox ----------------
        mv_bboxes = ann.get("mv_bboxes") or []
        mv_total = sum((arr.shape[0] for arr in mv_bboxes if hasattr(arr, "shape")), 0)
        mv_counts.append(mv_total)
        # ---------------- 3D bbox ----------------
        boxes_3d = info.get("gt_boxes")
        if boxes_3d is None and "gt_bboxes_3d" in ann:
            boxes_3d = ann["gt_bboxes_3d"]
        num_box3d = boxes_3d.shape[0] if hasattr(boxes_3d, "shape") else 0
        box3d_counts.append(num_box3d)

    total_frames = min(len(infos), args.limit or len(infos))
    print(f"Loaded {total_frames} samples from {args.ann}")
    if bev_num_classes:
        total_pixels = bev_counter.sum()
        print("\n[BEV mask]")
        print(f"  frames with mask: {total_frames - bev_missing} / {total_frames}")
        print(f"  frames with non-zero: {bev_nonzero_frames} / {total_frames}")
        print("  class pixel ratios:")
        for i in range(bev_num_classes):
            name = bev_classes[i] if bev_classes and i < len(bev_classes) else f"class_{i}"
            ratio = bev_counter[i] / max(total_pixels, 1)
            print(f"    {i}:{name:>18}  {ratio:.6f}  (count={bev_counter[i]})")
        print(f"  mask shape distribution: {dict(bev_shapes)}")
    else:
        print("\n[BEV mask] no class info found or masks missing.")

    mv_counts = np.array(mv_counts, dtype=np.int32)
    box3d_counts = np.array(box3d_counts, dtype=np.int32)
    print("\n[2D bbox per frame]")
    print(f"  mean={mv_counts.mean():.2f}, median={np.median(mv_counts):.2f}, max={mv_counts.max()}, nonzero={np.count_nonzero(mv_counts)}")
    print("[3D bbox per frame]")
    print(f"  mean={box3d_counts.mean():.2f}, median={np.median(box3d_counts):.2f}, max={box3d_counts.max()}, nonzero={np.count_nonzero(box3d_counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BEV/2D/3D annotations distribution")
    parser.add_argument("--ann", required=True, help="Path to *_infos_*.pkl")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only analyze first N samples (0 for all)")
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
