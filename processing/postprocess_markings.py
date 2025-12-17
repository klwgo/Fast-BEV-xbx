#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对多任务 JSON 中的标线进行后处理（去噪、填缝、膨胀/腐蚀），并输出可视化与可选 RLE。"""

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
    parser = argparse.ArgumentParser(description='标线后处理')
    parser.add_argument('json', help='输入多任务 JSON（含 markings.channels）')
    parser.add_argument('--out-dir', help='输出目录，默认 JSON 同名目录下的 post_marking')
    parser.add_argument('--channels', nargs='*', help='指定处理的通道，默认全部')
    parser.add_argument('--open-iter', type=int, default=1, help='opening 次数，去除散点噪声')
    parser.add_argument('--close-iter', type=int, default=1, help='closing 次数，填补小裂缝')
    parser.add_argument('--dilate-iter', type=int, default=0, help='膨胀次数，加粗线条')
    parser.add_argument('--erode-iter', type=int, default=0, help='腐蚀次数，细化线条')
    parser.add_argument('--max-hole-area', type=int, default=50, help='填充小孔的最大面积，0 关闭')
    parser.add_argument('--min-area', type=int, default=20, help='最小连通域面积过滤，0 关闭')
    parser.add_argument('--save-json', action='store_true', help='将后处理后的 RLE 写回 JSON (markings_post)')
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


def _morphology(mask: np.ndarray, open_iter: int, close_iter: int, dilate_iter: int, erode_iter: int) -> np.ndarray:
    if ndimage is None:
        if any(x > 0 for x in [open_iter, close_iter, dilate_iter, erode_iter]):
            print('warning: scipy 未安装，跳过形态学操作')
        return mask
    structure = np.ones((3, 3), dtype=bool)
    out = mask
    if open_iter > 0:
        out = ndimage.binary_opening(out, structure=structure, iterations=open_iter)
    if close_iter > 0:
        out = ndimage.binary_closing(out, structure=structure, iterations=close_iter)
    if dilate_iter > 0:
        out = ndimage.binary_dilation(out, structure=structure, iterations=dilate_iter)
    if erode_iter > 0:
        out = ndimage.binary_erosion(out, structure=structure, iterations=erode_iter)
    return out


def _process_channel(mask: np.ndarray, args) -> np.ndarray:
    m = mask.astype(bool)
    m = _morphology(m, args.open_iter, args.close_iter, args.dilate_iter, args.erode_iter)
    m = _fill_small_holes(m, args.max_hole_area)
    m = _filter_small_components(m, args.min_area)
    return m.astype(np.uint8)


def _draw_overlay(shape: Tuple[int, int], masks: Dict[str, np.ndarray], title: str, out_path: Path, dpi: int):
    height, width = shape
    base = np.zeros((height, width, 3), dtype=np.float32)
    colors = {
        'lane_marking': (0.2, 0.4, 0.9),
        'parking_line': (0.9, 0.2, 0.9),
        'other_ground_marking': (1.0, 0.5, 0.0),
        'zebra_crossing': (1.0, 1.0, 1.0),
    }
    for name, mask in masks.items():
        color = colors.get(name, (0.8, 0.8, 0.8))
        m = mask.astype(bool)
        base[m] = (1 - 0.2) * base[m] + 0.2 * np.asarray(color, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base, origin='lower', extent=[0, width, 0, height])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    ax.set_title(title)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    json_path = Path(args.json)
    if not json_path.is_file():
        raise FileNotFoundError(f'{json_path} 不存在')

    data = json.loads(json_path.read_text())
    frames = data.get('frames', [])
    if not frames:
        raise RuntimeError('frames 为空，无法处理')

    out_dir = Path(args.out_dir) if args.out_dir else json_path.with_suffix('') / 'post_marking'
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / 'mask_png'
    mask_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        mark = frame.get('markings', {}).get('channels', {})
        if not mark:
            continue
        shape = tuple(frame['drivable']['shape'])
        channel_names = args.channels if args.channels else list(mark.keys())
        processed: Dict[str, np.ndarray] = {}
        for name in channel_names:
            runs = mark.get(name, [])
            mask = _decode_rle(runs, shape)
            mask_p = _process_channel(mask, args)
            processed[name] = mask_p
            ch_dir = mask_dir / name
            ch_dir.mkdir(parents=True, exist_ok=True)
            token = frame.get('token', f'{idx:04d}')
            png_path = ch_dir / f'{idx:04d}_{token}_{name}.png'
            plt.imsave(png_path, mask_p, cmap='gray', dpi=args.dpi)

        token = frame.get('token', f'{idx:04d}')
        overlay_path = vis_dir / f'{idx:04d}_{token}_marking.png'
        _draw_overlay(shape, processed, f'token={token} idx={frame.get("index", idx)}', overlay_path, args.dpi)
        print(f'saved {overlay_path}')

        if args.save_json:
            frame.setdefault('markings_post', {}).setdefault('channels', {})
            for name, mask in processed.items():
                frame['markings_post']['channels'][name] = _encode_rle(mask)

    if args.save_json:
        json_out = out_dir / f'{json_path.stem}_with_markings_post.json'
        json_out.write_text(json.dumps(data))
        print(f'saved markings_post to {json_out}')


if __name__ == '__main__':
    main()
