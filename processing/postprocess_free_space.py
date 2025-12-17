#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从多任务 JSON 生成 free-space 掩码（drivable 扣除障碍物），可选写回 RLE。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='泊车 free-space 后处理')
    parser.add_argument('json', help='输入多任务 JSON（含 drivable/obstacles）')
    parser.add_argument('--out-dir', help='输出目录，默认在 JSON 同级目录下生成 *_post')
    parser.add_argument('--include-ambiguous', action='store_true', help='将 ambiguous 区域视为可行驶')
    parser.add_argument('--obstacle-margin', type=float, default=0.5, help='障碍物安全裕度（米）')
    parser.add_argument('--min-area', type=int, default=50, help='最小保留连通域面积（像素），0 关闭')
    parser.add_argument('--open-iter', type=int, default=1, help='形态学 opening 次数，去噪点')
    parser.add_argument('--close-iter', type=int, default=1, help='形态学 closing 次数，平滑边界/填小孔')
    parser.add_argument('--max-hole-area', type=int, default=200, help='填充小孔的最大面积，0 关闭')
    parser.add_argument('--save-json', action='store_true', help='将 free_space_rle 写回新 JSON')
    parser.add_argument('--dpi', type=int, default=200, help='保存 PNG 的 DPI')
    return parser.parse_args()


def _decode_rle(runs: List[List[int]], shape: Tuple[int, int]) -> np.ndarray:
    if not runs:
        return np.zeros(shape, dtype=np.uint8)
    flat = []
    for value, length in runs:
        flat.append(np.full(length, value, dtype=np.uint8))
    arr = np.concatenate(flat, axis=0)
    return arr.reshape(shape)


def _encode_rle(mask: np.ndarray) -> List[List[int]]:
    flat = mask.astype(np.uint8).ravel()
    if flat.size == 0:
        return []
    runs: List[List[int]] = []
    last = flat[0]
    count = 1
    for val in flat[1:]:
        if val == last:
            count += 1
        else:
            runs.append([int(last), int(count)])
            last = val
            count = 1
    runs.append([int(last), int(count)])
    return runs


def _world_to_pixel(x: float, y: float, pc_range, width: int, height: int):
    x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
    x_scale = width / max(x_max - x_min, 1e-6)
    y_scale = height / max(y_max - y_min, 1e-6)
    col = (x - x_min) * x_scale
    row = (y - y_min) * y_scale
    row = height - row  # flip y 以匹配 JSON 掩码坐标
    return col, row


def _build_obstacle_mask(obstacles: List[Dict], shape: Tuple[int, int], pc_range, margin: float):
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for obs in obstacles:
        bbox = obs.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin
        x_min, y_min, x_max, y_max = pc_range[0], pc_range[1], pc_range[3], pc_range[4]
        x1 = min(max(x1, x_min), x_max)
        x2 = min(max(x2, x_min), x_max)
        y1 = min(max(y1, y_min), y_max)
        y2 = min(max(y2, y_min), y_max)
        if x2 <= x1 or y2 <= y1:
            continue

        c0, r0 = _world_to_pixel(x1, y1, pc_range, width, height)
        c1, r1 = _world_to_pixel(x2, y2, pc_range, width, height)
        c_min, c_max = sorted([c0, c1])
        r_min, r_max = sorted([r0, r1])

        c0i = int(np.floor(max(c_min, 0)))
        c1i = int(np.ceil(min(c_max, width)))
        r0i = int(np.floor(max(r_min, 0)))
        r1i = int(np.ceil(min(r_max, height)))
        if c1i <= c0i or r1i <= r0i:
            continue
        mask[r0i:r1i, c0i:c1i] = True
    return mask


def _filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    if ndimage is None:
        print('warning: scipy 未安装，跳过连通域过滤')
        return mask
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
    if not keep_labels:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_labels)


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return mask
    if ndimage is None:
        print('warning: scipy 未安装，跳过小孔填充')
        return mask
    inv = ~mask
    labeled, num = ndimage.label(inv)
    if num == 0:
        return mask
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ])))
    sizes = ndimage.sum(inv, labeled, range(1, num + 1))
    for idx, size in enumerate(sizes, start=1):
        if idx in border_labels:
            continue
        if size <= max_area:
            mask[labeled == idx] = True
    return mask


def _morphology(mask: np.ndarray, open_iter: int, close_iter: int) -> np.ndarray:
    if ndimage is None:
        if open_iter > 0 or close_iter > 0:
            print('warning: scipy 未安装，跳过 opening/closing')
        return mask
    structure = np.ones((3, 3), dtype=bool)
    out = mask
    if open_iter > 0:
        out = ndimage.binary_opening(out, structure=structure, iterations=open_iter)
    if close_iter > 0:
        out = ndimage.binary_closing(out, structure=structure, iterations=close_iter)
    return out


def process_frame(frame: Dict, meta: Dict, *, include_ambiguous: bool, margin: float, min_area: int):
    shape = tuple(frame['drivable']['shape'])
    drivable = frame.get('drivable', {})
    drivable_pos = _decode_rle(drivable.get('positive_rle', []), shape).astype(bool)
    mask = drivable_pos.copy()
    if include_ambiguous:
        drivable_amb = _decode_rle(drivable.get('ambiguous_rle', []), shape).astype(bool)
        mask |= drivable_amb

    pc_range = meta.get('point_cloud_range', [-50, -50, -5, 50, 50, 3])
    obstacles = frame.get('obstacles', [])
    obs_mask = _build_obstacle_mask(obstacles, shape, pc_range, margin)
    free_space = mask & ~obs_mask
    return free_space.astype(np.uint8)


def main():
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.is_file():
        raise FileNotFoundError(f'{json_path} 不存在')

    data = json.loads(json_path.read_text())
    frames = data.get('frames', [])
    if not frames:
        raise RuntimeError('frames 为空，无法处理')
    meta = data.get('meta', {})

    out_dir = Path(args.out_dir) if args.out_dir else json_path.with_suffix('') / 'post'
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / 'free_space_png'
    png_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        free = process_frame(
            frame,
            meta,
            include_ambiguous=args.include_ambiguous,
            margin=args.obstacle_margin,
            min_area=args.min_area)
        free = _morphology(free.astype(bool), args.open_iter, args.close_iter)
        free = _fill_small_holes(free, args.max_hole_area)
        free = _filter_small_components(free, args.min_area)
        free = free.astype(np.uint8)

        token = frame.get('token', f'{idx:04d}')
        png_path = png_dir / f'{idx:04d}_{token}_free.png'
        plt.imsave(png_path, free, cmap='gray', dpi=args.dpi)
        if args.save_json:
            frame['free_space_rle'] = _encode_rle(free)
        print(f'saved {png_path}')

    if args.save_json:
        json_out = out_dir / f'{json_path.stem}_with_free.json'
        json_out.write_text(json.dumps(data))
        print(f'saved free_space_rle to {json_out}')


if __name__ == '__main__':
    main()
