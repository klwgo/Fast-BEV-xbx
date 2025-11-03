#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pickle
from collections import Counter
from pathlib import Path
import sys
import types

import numpy as np


def _ensure_numpy_core_alias():
    """兼容由较新 NumPy 版本生成的 pickle。

    NumPy 2.x 会在 pickle 里引用 ``numpy._core``，而旧版 NumPy 只有
    ``numpy.core``。在加载前主动创建别名，避免 ModuleNotFoundError。
    """
    if 'numpy._core' in sys.modules:
        return
    core = getattr(np, 'core', None)
    if core is None:
        return

    core_alias = types.ModuleType('numpy._core')
    core_alias.__dict__.update(core.__dict__)
    sys.modules['numpy._core'] = core_alias

    for attr in dir(core):
        obj = getattr(core, attr)
        if isinstance(obj, types.ModuleType) and obj.__package__ and obj.__package__.startswith('numpy.core'):
            sys.modules[f'numpy._core.{attr}'] = obj
            setattr(core_alias, attr, obj)


_ensure_numpy_core_alias()

EXPECTED_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
VALID_CLASSES = {'pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'}

def is_numpy_array(x, ndim=None):
    ok = isinstance(x, np.ndarray)
    if ok and ndim is not None:
        ok = x.ndim == ndim
    return ok

def main(pkl_path: Path):
    with pkl_path.open('rb') as f:
        data = pickle.load(f)

    errors = []
    infos = data.get('infos')
    if not isinstance(infos, list) or len(infos) == 0:
        errors.append('缺少有效 infos 列表')
        infos = []

    class_counter = Counter()
    for idx, info in enumerate(infos):
        for key in ['gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag']:
            if key not in info:
                errors.append(f'[{idx}] 缺少 {key}')
        gt_boxes = info.get('gt_boxes')
        if not is_numpy_array(gt_boxes, ndim=2) or gt_boxes.shape[1] != 9:
            errors.append(f'[{idx}] gt_boxes 需为 (N,9) 的 numpy.ndarray')

        gt_names = info.get('gt_names')
        if not is_numpy_array(gt_names, ndim=1):
            errors.append(f'[{idx}] gt_names 需为 numpy.ndarray；当前类型 {type(gt_names)}')
        else:
            class_counter.update(gt_names.tolist())
            unknown = set(gt_names.tolist()) - VALID_CLASSES
            if unknown:
                errors.append(f'[{idx}] 出现未映射类别 {unknown}')

        for arr_key in ['gt_velocity', 'num_lidar_pts', 'num_radar_pts', 'valid_flag']:
            arr = info.get(arr_key)
            if not isinstance(arr, np.ndarray):
                errors.append(f'[{idx}] {arr_key} 应为 numpy.ndarray，当前类型 {type(arr)}')

        cams = info.get('cams', {})
        missing = set(EXPECTED_CAMERAS) - set(cams.keys())
        if missing:
            errors.append(f'[{idx}] cams 缺少视角 {missing}')
        for cam_name in EXPECTED_CAMERAS:
            cam = cams.get(cam_name)
            if cam is None:
                continue
            annos = cam.get('annos', {})
            for field in ['bbox', 'category_id', 'category_name']:
                if field not in annos:
                    errors.append(f'[{idx}] {cam_name} annos 缺少 {field}')
                elif not isinstance(annos[field], np.ndarray):
                    errors.append(f'[{idx}] {cam_name}.{field} 需为 numpy.ndarray，当前 {type(annos[field])}')

    if errors:
        print('❌ 验证未通过：')
        for msg in errors[:50]:
            print(' -', msg)
        if len(errors) > 50:
            print(f' ... 共 {len(errors)} 条错误')
    else:
        print('✅ 验证通过，一切必需字段与类型符合要求')

    if class_counter:
        print('类别统计 Top5:', class_counter.most_common(5))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='检查 WoodScape info 文件是否满足训练要求')
    parser.add_argument('pkl', type=Path, help='路径，例如 data/woodscape_infos_train.pkl')
    main(parser.parse_args().pkl)
