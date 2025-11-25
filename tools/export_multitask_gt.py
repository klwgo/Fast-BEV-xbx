#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 WoodScape/SynWoodScape 的 GT 转成 Head-A/B/D 输出格式并写入 JSON.

该脚本直接复用 ``GenerateBEVMultitaskTargets`` 的标签生成逻辑，
因此输出结果与训练/推理时模型 Multitask Head 的输入完全一致。
当前实现覆盖以下任务：

* A：可行驶区域（以 0/1 二值栅格形式编码）
* B：路面标志/标线（按类别拆分成多通道二值图）
* D：障碍物 BEV 框列表（来源于 3D GT）

示例：

```
python tools/export_multitask_gt.py \
    configs/woodscape/fastbev_multitask_fisheye.py \
    --split val \
    --output work_dirs/woodscape_multitask_val.json \
    --max-samples 50
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import importlib.util
import types
import numpy as np

# ---------------------------------------------------------------------------
# 兼容 Python 3.12：确保 repo 根目录在 sys.path，优先导入 sitecustomize
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:  # noqa: F401 - 仅用于触发兼容补丁
    import sitecustomize  # type: ignore
except Exception:
    sitecustomize = None  # type: ignore

import mmcv
from mmcv import Config, DictAction

import importlib.util

_PREPROCESS_PATH = ROOT_DIR / 'mmdet3d/datasets/pipelines/preprocess.py'

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

_spec = importlib.util.spec_from_file_location('fastbev.bev_preprocess', _PREPROCESS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f'无法从 {_PREPROCESS_PATH} 加载 GenerateBEVMultitaskTargets')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GenerateBEVMultitaskTargets = getattr(_module, 'GenerateBEVMultitaskTargets')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='导出可行驶区域/标志/障碍物的 GT 结果（Head-A/B/D 输出格式）')
    parser.add_argument('config', help='Fast-BEV 配置文件路径')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val',
                        help='选择 cfg.data 中的哪个 split，默认 val')
    parser.add_argument('--output', required=True,
                        help='生成的 JSON 路径（若目录不存在会自动创建）')
    parser.add_argument('--max-samples', type=int, default=0,
                        help='最多导出多少个样本（0 表示全部样本）')
    parser.add_argument('--skip-empty', action='store_true',
                        help='若样本缺少 gt_bev_seg / gt_bboxes_3d，直接跳过且不写入统计信息')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='每处理多少帧打印一次进度；设置为 0 可关闭日志')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='以空格分隔的 key=value 形式覆盖 config 中的设置，例如 '
                             "--cfg-options dataset_choice='synwoodscape'")
    return parser.parse_args()


def _select_dataset_cfg(cfg: Config, split: str) -> Dict:
    if split not in cfg.data:
        raise KeyError(f'cfg.data 中不存在 split "{split}"')
    data_cfg = cfg.data[split]
    if isinstance(data_cfg, (list, tuple)):
        raise ValueError(f'{split} split 包含多 dataset 组合，暂不支持自动展开')
    data_cfg = data_cfg.copy()
    # RepeatDataset 等打包场景：取内部真正的数据集配置
    inner = data_cfg.get('dataset') if isinstance(data_cfg, dict) else None
    if inner is not None:
        if isinstance(inner, (list, tuple)):
            raise ValueError('暂不支持嵌套多 dataset 组合')
        data_cfg = inner.copy()
    return data_cfg


def _encode_binary_mask(mask: np.ndarray) -> List[List[int]]:
    arr = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return []
    runs: List[List[int]] = []
    prev = int(arr[0])
    length = 1
    for value in arr[1:]:
        value = int(value)
        if value == prev:
            length += 1
        else:
            runs.append([prev, length])
            prev = value
            length = 1
    runs.append([prev, length])
    return runs


def _boxes_to_obstacles(box_array: np.ndarray,
                        labels: np.ndarray,
                        class_names: Sequence[str]) -> List[Dict]:
    if box_array is None or labels is None:
        return []
    box_array = np.asarray(box_array)
    if box_array.size == 0:
        return []
    labels = np.asarray(labels).reshape(-1)
    obstacles = []
    for box, label in zip(box_array, labels):
        if len(box) < 5:
            continue
        cx = float(box[0])
        cy = float(box[1])
        dx = abs(float(box[3]))
        dy = abs(float(box[4]))
        x1 = cx - dx / 2.0
        x2 = cx + dx / 2.0
        y1 = cy - dy / 2.0
        y2 = cy + dy / 2.0
        cls_idx = int(label)
        cls_name = None
        if class_names and 0 <= cls_idx < len(class_names):
            cls_name = class_names[cls_idx]
        obstacles.append(dict(
            bbox=[round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
            center=[round(cx, 4), round(cy, 4)],
            size=[round(dx, 4), round(dy, 4)],
            class_id=cls_idx,
            class_name=cls_name,
            confidence=1.0,
        ))
    return obstacles


def _build_frame_entry(idx: int,
                       data_info: Dict,
                       targets: Dict,
                       generator,
                       class_names: Sequence[str],
                       boxes: np.ndarray,
                       labels: np.ndarray) -> Dict:
    token = data_info.get('token') or f'frame-{idx:06d}'
    frame = dict(
        index=idx,
        token=token,
    )
    if 'frame_id' in data_info:
        frame['frame_id'] = int(data_info['frame_id'])
    if 'scene_name' in data_info:
        frame['scene_name'] = data_info['scene_name']

    drivable = np.asarray(targets.get('gt_drivable_mask'))
    frame['drivable'] = dict(
        shape=list(drivable.shape),
        positive_rle=_encode_binary_mask(drivable == 1),
        ambiguous_rle=_encode_binary_mask(drivable == 2),
    )

    marking = np.asarray(targets.get('gt_marking_mask'))
    marking_channels = {}
    for offset, name in enumerate(getattr(generator, 'marking_classes', []), start=1):
        marking_channels[name] = _encode_binary_mask(marking == offset)
    frame['markings'] = dict(
        shape=list(marking.shape),
        channels=marking_channels,
    )

    obstacles = _boxes_to_obstacles(boxes, labels, class_names)
    frame['obstacles'] = obstacles
    return frame


def _resolve_path(path_value, data_root: Path) -> Optional[Path]:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
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
        if path is None or not path.exists():
            return None
        if path.suffix.lower() == '.npy':
            return np.load(path)
        return mmcv.imread(str(path), flag='unchanged')
    return np.asarray(value)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    dataset_cfg = _select_dataset_cfg(cfg, args.split)
    dataset_choice = cfg.get('dataset_choice')
    registry = cfg.get('dataset_registry', {}) if hasattr(cfg, 'get') else {}
    if isinstance(registry, dict) and dataset_choice in registry:
        info = registry[dataset_choice]
        dataset_cfg['data_root'] = info.get('data_root', dataset_cfg.get('data_root'))
        ann_key = 'train_ann' if args.split == 'train' else 'val_ann'
        dataset_cfg['ann_file'] = info.get(ann_key, dataset_cfg.get('ann_file'))
    bev_cfg = dataset_cfg.get('bev_target_generator')
    if not bev_cfg:
        raise RuntimeError('当前数据集配置未提供 bev_target_generator，无法构建多任务标签')
    generator = GenerateBEVMultitaskTargets(**bev_cfg)

    data_root = Path(dataset_cfg.get('data_root', ROOT_DIR))
    if not data_root.is_absolute():
        data_root = (ROOT_DIR / data_root).resolve()
    ann_file = Path(dataset_cfg.get('ann_file'))
    if not ann_file.is_absolute():
        candidate = (ROOT_DIR / ann_file).resolve()
        if candidate.exists():
            ann_file = candidate
        else:
            ann_file = (data_root / ann_file).resolve()
    records = mmcv.load(str(ann_file))
    infos = records.get('infos', [])
    class_names = list(dataset_cfg.get('classes', ()))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    total = len(infos)
    frames: List[Dict] = []
    skipped = 0
    for idx, info in enumerate(infos):
        if args.max_samples and len(frames) >= args.max_samples:
            break
        ann_meta = info.get('ann_info', {})
        bev_value = ann_meta.get('gt_bev_seg')
        bev_mask = _load_bev_mask(bev_value, data_root)
        if bev_mask is None:
            skipped += 1
            if not args.skip_empty:
                print(f'[WARN] 样本 {idx} 缺少 gt_bev_seg，跳过')
            continue
        bev_classes = ann_meta.get('bev_seg_classes') or []
        boxes = np.asarray(info.get('gt_boxes', np.zeros((0, 9))), dtype=np.float32)
        names = info.get('gt_names', [])
        labels = np.asarray([class_to_id.get(str(name), -1) for name in names], dtype=np.int64)
        ann_stub = dict(
            gt_bboxes_3d=boxes,
            gt_labels_3d=labels,
            bev_mask_shape=bev_mask.shape,
        )
        targets = generator.build_targets(dict(ann_info=ann_stub), bev_mask, bev_classes)
        if not targets:
            skipped += 1
            if not args.skip_empty:
                print(f'[WARN] 样本 {idx} 无法生成多任务标签，跳过')
            continue
        if 'gt_drivable_mask' not in targets or 'gt_marking_mask' not in targets:
            skipped += 1
            if not args.skip_empty:
                print(f'[WARN] 样本 {idx} 缺少 drivable/marking 标签，跳过')
            continue
        frame_entry = _build_frame_entry(idx, info, targets, generator, class_names, boxes, labels)
        frames.append(frame_entry)
        if args.log_interval and len(frames) % args.log_interval == 0:
            print(f'[INFO] 已导出 {len(frames)} / {total} 帧')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pc_range = getattr(generator, 'point_cloud_range', None)
    if pc_range is not None:
        pc_range = [float(x) for x in np.asarray(pc_range).reshape(-1)]

    meta = dict(
        config=str(Path(args.config).resolve()),
        split=args.split,
        dataset='AnnotationFile',
        source_root=str(data_root),
        ann_file=str(ann_file),
        point_cloud_range=pc_range,
        drivable_classes=list(getattr(generator, 'drivable_class_names', [])),
        marking_classes=list(getattr(generator, 'marking_classes', [])),
        obstacle_classes=class_names,
        total_frames=total,
        exported_frames=len(frames),
        skipped_frames=skipped,
    )
    payload = dict(meta=meta, frames=frames)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f'[DONE] 已将 {len(frames)} 个样本写入 {out_path}')


if __name__ == '__main__':
    main()
