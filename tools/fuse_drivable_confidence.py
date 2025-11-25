#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 export_multitask_gt.py 生成的 JSON 做可行驶区域置信度融合."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Fuse drivable confidence from multitask JSON')
    parser.add_argument('input', help='原始 export_multitask_gt JSON 路径')
    parser.add_argument('--output', help='输出路径（默认覆盖输入）')
    parser.add_argument('--neighbor-thr', type=int, default=2,
                        help='将 ambiguous 像素提升为 positive 所需的正样邻居数量')
    parser.add_argument('--min-neighbor', type=int, default=1,
                        help='若某个 positive 像素的正邻居少于该值，则视为噪点')
    return parser.parse_args()


def decode_rle(runs: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
    if not runs:
        return np.zeros(shape, dtype=np.uint8)
    flat = np.concatenate([
        np.full(length, value, dtype=np.uint8) for value, length in runs
    ])
    total = shape[0] * shape[1]
    if flat.size < total:
        flat = np.pad(flat, (0, total - flat.size), constant_values=0)
    return flat[:total].reshape(shape)


def encode_rle(mask: np.ndarray) -> List[List[int]]:
    flat = mask.reshape(-1).astype(np.uint8)
    runs: List[List[int]] = []
    prev = flat[0]
    length = 1
    for value in flat[1:]:
        if value == prev:
            length += 1
        else:
            runs.append([int(prev), int(length)])
            prev = value
            length = 1
    runs.append([int(prev), int(length)])
    # 仅保留 value==1 的段落，方便压缩
    filtered = [[1, length] for value, length in runs if value == 1]
    return filtered


def count_neighbors(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask.astype(np.uint8), 1, mode='constant', constant_values=0)
    neighbors = np.zeros_like(mask, dtype=np.int32)
    for dx in range(3):
        for dy in range(3):
            if dx == 1 and dy == 1:
                continue
            neighbors += pad[dx:dx + mask.shape[0], dy:dy + mask.shape[1]]
    return neighbors


def fuse_frame(frame: dict, neighbor_thr: int, min_neighbor: int):
    drivable = frame.get('drivable')
    if not drivable:
        return
    shape = tuple(drivable.get('shape', []))
    if len(shape) != 2:
        return
    pos = decode_rle(drivable.get('positive_rle', []), shape).astype(bool)
    amb = decode_rle(drivable.get('ambiguous_rle', []), shape).astype(bool)
    neighbors = count_neighbors(pos)

    promote = amb & (neighbors >= neighbor_thr)
    fused = pos | promote

    # 移除孤立 positive 像素
    fused &= (count_neighbors(fused) >= min_neighbor)

    remain_amb = amb & (~promote)

    drivable['positive_rle'] = encode_rle(fused.astype(np.uint8))
    drivable['ambiguous_rle'] = encode_rle(remain_amb.astype(np.uint8))


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    data = json.loads(input_path.read_text())
    for frame in data.get('frames', []):
        fuse_frame(frame, args.neighbor_thr, args.min_neighbor)

    output_path.write_text(json.dumps(data, ensure_ascii=False))
    print(f'saved fused JSON to {output_path}')


if __name__ == '__main__':
    main()
