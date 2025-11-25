#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化 export_multitask_gt.py 生成的 JSON 中的单帧结果."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='可视化多任务 JSON (drivable/marking/obstacle)')
    parser.add_argument('json', help='export_multitask_gt.py 生成的 JSON 路径')
    parser.add_argument('--index', type=int, default=0, help='可视化的帧索引（默认 0）')
    parser.add_argument('--token', help='也可以直接指定 token，优先级高于 index')
    parser.add_argument('--output', help='输出图片路径（默认仅弹窗显示）')
    parser.add_argument('--dpi', type=int, default=200, help='保存图片时使用的 dpi')
    return parser.parse_args()


def _decode_rle(runs: List[List[int]], shape) -> np.ndarray:
    if not runs:
        return np.zeros(shape, dtype=np.uint8)
    flat = []
    for value, length in runs:
        flat.append(np.full(length, value, dtype=np.uint8))
    arr = np.concatenate(flat, axis=0)
    return arr.reshape(shape)


def _apply_mask(canvas: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5):
    if not mask.any():
        return
    color_arr = np.asarray(color, dtype=np.float32)
    mask = mask.astype(bool)
    canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color_arr


def _world_to_pixel(x: float, y: float, pc_range, width: int, height: int, flip_y: bool = True):
    x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    x_scale = width / max(x_max - x_min, 1e-6)
    y_scale = height / max(y_max - y_min, 1e-6)
    col = (x - x_min) * x_scale
    row = (y - y_min) * y_scale
    if flip_y:
        row = height - row
    return col, row


def _clip_box(bbox, pc_range):
    x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, x_min), x_max)
    x2 = min(max(x2, x_min), x_max)
    y1 = min(max(y1, y_min), y_max)
    y2 = min(max(y2, y_min), y_max)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _select_frame(data: Dict, token: Optional[str], index: int) -> Dict:
    frames = data.get('frames', [])
    if token:
        for item in frames:
            if item.get('token') == token:
                return item
        raise KeyError(f'未找到 token={token} 的帧')
    if not frames:
        raise RuntimeError('frames 为空')
    index = max(0, min(index, len(frames) - 1))
    return frames[index]


def main():
    args = parse_args()
    data = json.loads(Path(args.json).read_text())
    frame = _select_frame(data, args.token, args.index)
    meta = data.get('meta', {})
    shape = tuple(frame['drivable']['shape'])
    height, width = shape

    base = np.zeros((height, width, 3), dtype=np.float32)
    drivable = frame.get('drivable', {})
    drivable_pos = _decode_rle(drivable.get('positive_rle', []), shape)
    drivable_amb = _decode_rle(drivable.get('ambiguous_rle', []), shape)
    _apply_mask(base, drivable_pos == 1, color=(0.0, 0.5, 0.0), alpha=0.6)
    _apply_mask(base, drivable_amb == 1, color=(0.9, 0.8, 0.0), alpha=0.6)

    marking_colors = {
        'lane_marking': (0.2, 0.4, 0.9),
        'parking_line': (0.9, 0.2, 0.9),
        'other_ground_marking': (1.0, 0.5, 0.0),
        'zebra_crossing': (1.0, 1.0, 1.0),
    }
    markings = frame.get('markings', {}).get('channels', {})
    for name, runs in markings.items():
        mask = _decode_rle(runs, shape)
        color = marking_colors.get(name, (0.8, 0.8, 0.8))
        _apply_mask(base, mask == 1, color, alpha=0.8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base, origin='lower', extent=[0, width, 0, height])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title(f"token={frame.get('token')} idx={frame.get('index')}")
    ax.axis('off')

    pc_range = meta.get('point_cloud_range', [-50, -50, -5, 50, 50, 3])
    obstacles = frame.get('obstacles', [])
    for obs in obstacles:
        clipped = _clip_box(obs['bbox'], pc_range)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        c0, r0 = _world_to_pixel(x1, y1, pc_range, width, height)
        c1, r1 = _world_to_pixel(x2, y2, pc_range, width, height)
        x_left = min(c0, c1)
        y_bottom = min(r0, r1)
        rect = plt.Rectangle((x_left, y_bottom), abs(c1 - c0), abs(r1 - r0),
                             edgecolor='red', linewidth=1.0, fill=False)
        ax.add_patch(rect)
        label = obs.get('class_name', str(obs.get('class_id')))
        tx = np.clip(x_left, 0, width - 1)
        ty = np.clip(y_bottom + abs(r1 - r0), 0, height - 1)
        ax.text(tx, ty, label, color='red', fontsize=6, va='bottom')

    fig.tight_layout()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f'saved to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
