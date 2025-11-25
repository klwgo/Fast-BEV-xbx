#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在指定相机图像上可视化 Syn/WoodScape 的 2D GT 框."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:  # noqa: F401
    import sitecustomize  # type: ignore
except Exception:  # pragma: no cover
    sitecustomize = None  # type: ignore

import mmcv
from mmcv import Config, DictAction


def parse_args():
    parser = argparse.ArgumentParser(description='可视化 Syn/WoodScape 2D GT 框')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--index', type=int, default=0, help='样本索引')
    parser.add_argument('--token', help='可选，指定 token 定位样本')
    parser.add_argument('--cam-names', nargs='+', help='需要可视化的相机列表（默认使用 camera_types 全部相机）')
    parser.add_argument('--output', default='work_dirs/vis_2d_bbox.png', help='输出图片路径')
    parser.add_argument('--show-text', action='store_true', help='是否在框旁显示类别文字（默认不显示，仅用图例说明）')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help="使用 key=value 覆写 config，如 dataset_choice='synwoodscape'")
    return parser.parse_args()


def _select_dataset_cfg(cfg: Config, split: str) -> dict:
    data_cfg = cfg.data[split]
    if isinstance(data_cfg, (list, tuple)):
        raise ValueError('暂不支持多 dataset 组合')
    return data_cfg.copy()


def _resolve_path(path_value, data_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (data_root, ROOT_DIR):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f'无法定位文件：{path_value}')


def _select_info(infos: List[dict], token: Optional[str], index: int):
    if token:
        for info in infos:
            if info.get('token') == token:
                return info
        raise KeyError(f'未找到 token={token} 的样本')
    index = max(0, min(index, len(infos) - 1))
    return infos[index]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    dataset_cfg = _select_dataset_cfg(cfg, args.split)
    dataset_choice = getattr(cfg, 'dataset_choice', None)
    registry = getattr(cfg, 'dataset_registry', None)
    default_root = dataset_cfg.get('data_root')
    default_ann = dataset_cfg.get('ann_file')
    if isinstance(registry, dict) and dataset_choice in registry:
        info = registry[dataset_choice]
        dataset_cfg['data_root'] = info.get('data_root', default_root)
        ann_key = 'train_ann' if args.split == 'train' else 'val_ann'
        dataset_cfg['ann_file'] = info.get(ann_key, default_ann)

    data_root_value = dataset_cfg.get('data_root', ROOT_DIR)
    candidates = []
    path_obj = Path(data_root_value)
    candidates.append(path_obj)
    if not path_obj.is_absolute():
        candidates.append((ROOT_DIR / path_obj))
    if default_root:
        candidates.append((ROOT_DIR / default_root))
    data_root = None
    for cand in candidates:
        cand = Path(cand).resolve()
        if cand.exists():
            data_root = cand
            break
    if data_root is None:
        raise FileNotFoundError('未能解析 data_root，请检查配置路径')

    ann_candidates = []
    ann_value = dataset_cfg.get('ann_file')
    if ann_value:
        ann_path = Path(ann_value)
        if ann_path.is_absolute():
            ann_candidates.append(ann_path)
        else:
            ann_candidates.append(data_root / ann_path)
            ann_candidates.append(ROOT_DIR / ann_path)
    if default_ann:
        ann_candidates.append(ROOT_DIR / default_ann)
    ann_file = None
    for cand in ann_candidates:
        cand = Path(cand).resolve()
        if cand.exists():
            ann_file = cand
            break
    if ann_file is None:
        raise FileNotFoundError('未找到 ann_file，请检查配置')

    with ann_file.open('rb') as f:
        records = pickle.load(f)
    infos = records['infos']
    info = _select_info(infos, args.token, args.index)
    ann = info.get('ann_info', {})

    cam_types = dataset_cfg.get('camera_types')
    if cam_types is None:
        cam_types = list(info.get('cams', {}).keys())
    target_cams = args.cam_names or cam_types
    target_cams = [cam for cam in target_cams if cam in cam_types]
    if not target_cams:
        raise ValueError('cam-names 与 camera_types 不匹配，无法可视化')

    mv_bboxes = ann.get('mv_bboxes', [])
    mv_labels = ann.get('mv_labels', [])
    class_names = ann.get('bbox_class_names', [])
    cmap = plt.colormaps['tab10']
    colors = [cmap(i % cmap.N) for i in range(max(len(class_names), 1))]
    handles = []
    for idx, name in enumerate(class_names):
        handles.append(patches.Patch(color=colors[idx], label=name))

    cols = 2
    rows = int(np.ceil(len(target_cams) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(target_cams):]:
        ax.axis('off')
    for ax, cam_name in zip(axes, target_cams):
        if cam_name not in info['cams']:
            ax.axis('off')
            continue
        cam_idx = cam_types.index(cam_name)
        if cam_idx >= len(mv_bboxes):
            ax.axis('off')
            continue
        boxes = np.asarray(mv_bboxes[cam_idx], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(mv_labels[cam_idx], dtype=np.int64).reshape(-1)
        cam_meta = info['cams'][cam_name]
        img_path = _resolve_path(cam_meta['data_path'], data_root)
        img = mmcv.imread(str(img_path))
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        ax.imshow(img)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1.2,
                                     edgecolor=colors[int(label) % len(colors)],
                                     facecolor='none')
            ax.add_patch(rect)
            if args.show_text:
                name = class_names[label] if 0 <= label < len(class_names) else str(label)
                ax.text(x1, max(y1 - 2, 0), name, color='yellow', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
        ax.set_title(cam_name)
        ax.axis('off')

    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"token={info.get('token')}")
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved to {output_path}')


if __name__ == '__main__':
    main()
