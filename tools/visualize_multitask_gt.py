#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接使用 WoodScape/SynWoodScape 的 GT BEV mask 可视化 Head-A/B/D 标签。"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:  # noqa: F401 - 触发 sitecustomize 补丁
    import sitecustomize  # type: ignore
except Exception:  # pragma: no cover
    sitecustomize = None  # type: ignore

from mmcv import Config, DictAction

import importlib.util
import types

PREPROCESS_PATH = ROOT_DIR / 'mmdet3d/datasets/pipelines/preprocess.py'

if 'mmdet.datasets' not in sys.modules:
    mmdet_stub = types.ModuleType('mmdet')
    datasets_stub = types.ModuleType('mmdet.datasets')

    class _SimpleRegistry:
        def register_module(self, *args, **kwargs):
            def decorator(obj):
                return obj

            return decorator

    datasets_stub.PIPELINES = _SimpleRegistry()
    mmdet_stub.datasets = datasets_stub
    sys.modules['mmdet'] = mmdet_stub
    sys.modules['mmdet.datasets'] = datasets_stub

spec = importlib.util.spec_from_file_location('fastbev.preprocess', PREPROCESS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f'无法加载 {PREPROCESS_PATH}')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GenerateBEVMultitaskTargets = getattr(module, 'GenerateBEVMultitaskTargets')


def parse_args():
    parser = argparse.ArgumentParser(description='可视化 Fast-BEV BEV GT (drivable/marking/obstacle)')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--index', type=int, default=0, help='可视化的样本索引')
    parser.add_argument('--token', help='可选，指定 token 定位具体样本')
    parser.add_argument('--output', help='输出图片路径（默认弹窗显示）')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help="使用 key=value 覆写 config，如 dataset_choice='synwoodscape'")
    return parser.parse_args()


def _select_dataset_cfg(cfg: Config, split: str) -> Dict:
    data_cfg = cfg.data[split]
    if isinstance(data_cfg, (list, tuple)):
        raise ValueError('暂不支持多 dataset 组合')
    return data_cfg.copy()


def _resolve_path(path_value, data_root: Path) -> Optional[Path]:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (data_root, ROOT_DIR):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def _load_bev_mask(value, data_root: Path):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = _resolve_path(value, data_root)
        if path is None:
            return None
        if path.suffix.lower() == '.npy':
            return np.load(path)
        return plt.imread(path)
    return np.asarray(value)


def _select_info(infos: List[Dict], token: Optional[str], index: int):
    if token:
        for info in infos:
            if info.get('token') == token:
                return info
        raise KeyError(f'未找到 token={token} 的样本')
    index = max(0, min(index, len(infos) - 1))
    return infos[index]


def _build_colors(names: List[str]):
    palette = [
        (0.9, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.4, 0.9),
        (0.9, 0.5, 0.1),
        (0.9, 0.2, 0.9),
        (0.2, 0.9, 0.9),
    ]
    return {name: palette[idx % len(palette)] for idx, name in enumerate(names)}


def _blend(canvas, mask, color, alpha=0.8):
    if mask is None:
        return
    mask = mask.astype(bool)
    if not mask.any():
        return
    color_arr = np.asarray(color, dtype=np.float32)
    canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color_arr


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    dataset_cfg = _select_dataset_cfg(cfg, args.split)
    default_data_root = dataset_cfg.get('data_root')
    default_ann = dataset_cfg.get('ann_file')
    dataset_choice = getattr(cfg, 'dataset_choice', None)
    registry = getattr(cfg, 'dataset_registry', None)
    if isinstance(registry, dict) and dataset_choice in registry:
        info = registry[dataset_choice]
        dataset_cfg['data_root'] = info.get('data_root', default_data_root)
        ann_key = 'train_ann' if args.split == 'train' else 'val_ann'
        dataset_cfg['ann_file'] = info.get(ann_key, default_ann)
    bev_cfg = dataset_cfg.get('bev_target_generator')
    if not bev_cfg:
        raise RuntimeError('数据集未配置 bev_target_generator')
    generator = GenerateBEVMultitaskTargets(**bev_cfg)

    data_root_value = dataset_cfg.get('data_root', ROOT_DIR)
    data_root = Path(data_root_value)
    candidates = [data_root]
    if not data_root.is_absolute():
        candidates.append((ROOT_DIR / data_root_value))
    if default_data_root:
        candidates.append((ROOT_DIR / default_data_root))
    data_root_resolved = None
    for cand in candidates:
        cand = cand.resolve()
        if cand.exists():
            data_root_resolved = cand
            break
    if data_root_resolved is None:
        raise FileNotFoundError('无法定位 data_root，请检查配置')
    data_root = data_root_resolved

    ann_value = dataset_cfg.get('ann_file')
    ann_candidates = []
    ann_path = Path(ann_value) if ann_value else None
    if ann_path is not None:
        if ann_path.is_absolute():
            ann_candidates.append(ann_path)
        else:
            ann_candidates.append((data_root / ann_path))
            ann_candidates.append((ROOT_DIR / ann_path))
    if default_ann:
        ann_candidates.append((ROOT_DIR / default_ann))
    ann_file = None
    for cand in ann_candidates:
        cand = Path(cand).resolve()
        if cand.exists():
            ann_file = cand
            break
    if ann_file is None:
        raise FileNotFoundError('无法定位 ann_file，请检查配置')
    with ann_file.open('rb') as f:
        records = pickle.load(f)
    infos = records['infos']
    info = _select_info(infos, args.token, args.index)

    ann_meta = info.get('ann_info', {})
    bev_mask = _load_bev_mask(ann_meta.get('gt_bev_seg'), data_root)
    if bev_mask is None:
        raise RuntimeError('当前样本缺少 gt_bev_seg')
    bev_classes = ann_meta.get('bev_seg_classes') or []
    boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
    class_names = list(dataset_cfg.get('classes', []))
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    gt_names = info.get('gt_names', [])
    labels = np.asarray([name_to_id.get(str(name), -1) for name in gt_names], dtype=np.int64)
    ann_stub = dict(
        gt_bboxes_3d=boxes,
        gt_labels_3d=labels,
        bev_mask_shape=np.asarray(bev_mask).shape,
    )
    targets = generator.build_targets(dict(ann_info=ann_stub), bev_mask, bev_classes)
    if not targets:
        raise RuntimeError('未成功生成多任务标签')

    drivable = targets['gt_drivable_mask']
    marking = targets['gt_marking_mask']
    obstacle = targets.get('gt_obstacle_mask')
    height, width = drivable.shape
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[drivable == 1] = (0.0, 0.5, 0.0)
    canvas[drivable == 2] = (0.8, 0.6, 0.0)

    marking_names = list(getattr(generator, 'marking_classes', []))
    marking_colors = _build_colors(marking_names)
    for idx, name in enumerate(marking_names, start=1):
        _blend(canvas, marking == idx, marking_colors[name], alpha=0.9)

    obstacle_ann = None
    if obstacle is not None:
        obs_names = list(getattr(generator, 'obstacle_class_names', []))
        obs_colors = _build_colors(obs_names)
        obstacle_ann = []
        for cls_idx, name in enumerate(obs_names, start=1):
            mask = obstacle == cls_idx
            _blend(canvas, mask, obs_colors[name], alpha=0.9)
            if mask.any():
                ys, xs = np.nonzero(mask)
                cy = ys.mean()
                cx = xs.mean()
                obstacle_ann.append((cx, cy, name))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(canvas, origin='lower')
    ax.set_title(f"token={info.get('token')} idx={args.index}")
    ax.axis('off')
    if obstacle_ann:
        for cx, cy, name in obstacle_ann:
            ax.text(cx, cy, name, color='yellow', fontsize=6,
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1',
                                                       facecolor='black', alpha=0.4, edgecolor='none'))
    fig.tight_layout()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        print(f'saved to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
