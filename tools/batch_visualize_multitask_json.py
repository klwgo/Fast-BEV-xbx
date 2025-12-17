#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量可视化 export_multitask_gt.py 生成的 JSON（drivable/marking/obstacle）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='批量可视化多任务 JSON')
    parser.add_argument('json', help='JSON 路径（export_multitask_gt.py 生成）')
    parser.add_argument('--out-dir', help='输出目录，默认与 JSON 同名加 _vis')
    parser.add_argument('--start', type=int, default=0, help='起始帧索引')
    parser.add_argument('--max-frames', type=int, default=0, help='最多可视化的帧数，0 表示全部')
    parser.add_argument('--dpi', type=int, default=200, help='保存图片的 DPI')
    parser.add_argument('--no-ambiguous', action='store_true', help='不绘制 ambiguous_rle 区域')
    parser.add_argument('--no-marking', action='store_true', help='不绘制标线')
    parser.add_argument('--no-obstacle', action='store_true', help='不绘制障碍物框')
    return parser.parse_args()


def _decode_rle(runs: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
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


def _draw_obstacles(ax, obstacles: List[Dict], pc_range, width: int, height: int):
    color_map = {
        'pedestrian': 'red',
        'four-wheeler vehicle': 'orange',
        'two-wheeler vehicle': 'cyan',
    }
    for obs in obstacles:
        clipped = _clip_box(obs['bbox'], pc_range)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        c0, r0 = _world_to_pixel(x1, y1, pc_range, width, height)
        c1, r1 = _world_to_pixel(x2, y2, pc_range, width, height)
        x_left = min(c0, c1)
        y_bottom = min(r0, r1)
        rect = plt.Rectangle(
            (x_left, y_bottom),
            abs(c1 - c0),
            abs(r1 - r0),
            edgecolor=color_map.get(obs.get('class_name'), 'white'),
            linewidth=1.0,
            fill=False)
        ax.add_patch(rect)
        label = obs.get('class_name', str(obs.get('class_id', 'obj')))
        tx = np.clip(x_left, 0, width - 1)
        ty = np.clip(y_bottom + abs(r1 - r0), 0, height - 1)
        ax.text(tx, ty, label, color='white', fontsize=6, va='bottom', ha='left')


def _visualize_frame(
    frame: Dict,
    meta: Dict,
    *,
    draw_ambiguous: bool = True,
    draw_marking: bool = True,
    draw_obstacle: bool = True):
    shape = tuple(frame['drivable']['shape'])
    height, width = shape
    base = np.zeros((height, width, 3), dtype=np.float32)

    drivable = frame.get('drivable', {})
    drivable_pos = _decode_rle(drivable.get('positive_rle', []), shape)
    _apply_mask(base, drivable_pos == 1, color=(0.0, 0.6, 0.0), alpha=0.6)
    if draw_ambiguous:
        drivable_amb = _decode_rle(drivable.get('ambiguous_rle', []), shape)
        _apply_mask(base, drivable_amb == 1, color=(0.9, 0.8, 0.0), alpha=0.6)

    marking_colors = {
        'lane_marking': (0.2, 0.4, 0.9),
        'parking_line': (0.9, 0.2, 0.9),
        'other_ground_marking': (1.0, 0.5, 0.0),
        'zebra_crossing': (1.0, 1.0, 1.0),
    }
    if draw_marking:
        markings = frame.get('markings', {}).get('channels', {})
        for name, runs in markings.items():
            mask = _decode_rle(runs, shape)
            color = marking_colors.get(name, (0.8, 0.8, 0.8))
            _apply_mask(base, mask == 1, color, alpha=0.8)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base, origin='lower', extent=[0, width, 0, height])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_title(f"token={frame.get('token')} idx={frame.get('index')}")

    pc_range = meta.get('point_cloud_range', [-50, -50, -5, 50, 50, 3])
    if draw_obstacle:
        obstacles = frame.get('obstacles', [])
        _draw_obstacles(ax, obstacles, pc_range, width, height)
    fig.tight_layout(pad=0.2)
    return fig


def main():
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.is_file():
        raise FileNotFoundError(f'{json_path} 不存在')

    data = json.loads(json_path.read_text())
    frames: List[Dict] = data.get('frames', [])
    if not frames:
        raise RuntimeError('frames 为空，无法可视化')

    meta: Dict = data.get('meta', {})
    out_dir = Path(args.out_dir) if args.out_dir else json_path.with_suffix('') / 'vis'
    out_dir.mkdir(parents=True, exist_ok=True)

    start = max(args.start, 0)
    max_frames = args.max_frames if args.max_frames > 0 else len(frames)
    selected = frames[start:start + max_frames]

    for i, frame in enumerate(selected, start=start):
        fig = _visualize_frame(
            frame,
            meta,
            draw_ambiguous=not args.no_ambiguous,
            draw_marking=not args.no_marking,
            draw_obstacle=not args.no_obstacle)
        token = frame.get('token', f'{i:04d}')
        out_path = out_dir / f'{i:04d}_{token}.png'
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f'saved {out_path}')


if __name__ == '__main__':
    main()
