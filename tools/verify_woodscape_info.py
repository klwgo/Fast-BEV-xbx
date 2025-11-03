#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import types
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

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
OBSTACLE_CLASSES = {'pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'}
ROAD_MARKING_CLASSES = {
    'lane_marking',
    'parking_line',
    'other_ground_marking',
    'zebra_crossing',
}
DRIVABLE_AREA_CLASSES = {'road_surface', 'free_space'}

REF_MAIN_ZIP_GLOB = 'WoodScape_ICCV19*.zip'
REF_INSTANCE_INFO = 'WoodScape_ICCV19/instance_annotations/class_info.json'
REF_INSTANCE_ZIP = 'WoodScape_ICCV19/instance_annotations/instance_annotations.zip'
REF_BOX_INFO = 'WoodScape_ICCV19/box_2d_annotations/box_2d_annotation_info.json'
REF_BOX_ZIP = 'WoodScape_ICCV19/box_2d_annotations/box_2d_annotations.zip'
REF_SEM_INFO = 'WoodScape_ICCV19/semantic_annotations/seg_annotation_info.json'
REF_SEM_ZIP = 'WoodScape_ICCV19/semantic_annotations/semantic_annotations.zip'
REF_MOTION_INFO = 'WoodScape_ICCV19/motion_annotations/motion_annotation_info.json'
REF_MOTION_ZIP = 'WoodScape_ICCV19/motion_annotations/motion_annotations.zip'


def _detect_main_zip(root: Path) -> Path:
    root = Path(root)
    if root.is_file() and root.suffix == '.zip':
        return root
    candidates = sorted(root.glob(REF_MAIN_ZIP_GLOB))
    if not candidates:
        raise FileNotFoundError(f'未在 {root} 下找到 WoodScape 主数据包 ({REF_MAIN_ZIP_GLOB})')
    return candidates[0]


@contextmanager
def _open_nested_zip(parent: zipfile.ZipFile, inner_path: str):
    tmp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
    try:
        with parent.open(inner_path) as src:
            shutil.copyfileobj(src, tmp)
        tmp.close()
        inner = zipfile.ZipFile(tmp.name)
        try:
            yield inner
        finally:
            inner.close()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _load_json_from_zip(zip_obj: zipfile.ZipFile, path: str):
    with zip_obj.open(path) as f:
        return json.loads(f.read().decode('utf-8'))


def _load_reference_stats(root: Path):
    stats = {}
    main_zip_path = _detect_main_zip(root)
    with zipfile.ZipFile(main_zip_path) as main_zip:
        # semantic information
        try:
            seg_info = _load_json_from_zip(main_zip, REF_SEM_INFO)
            stats['semantic'] = {
                'class_names': seg_info.get('class_names', []),
                'class_indexes': seg_info.get('class_indexes', [])
            }
            with _open_nested_zip(main_zip, REF_SEM_ZIP) as seg_zip:
                pngs = [n for n in seg_zip.namelist() if n.lower().endswith('.png')]
                per_cam = Counter()
                for name in pngs:
                    cam = name.split('_')[-1].split('.')[0]
                    per_cam[cam] += 1
                stats['semantic']['image_count'] = len(pngs)
                stats['semantic']['per_camera'] = per_cam
        except KeyError:
            stats['semantic'] = None

        # instance polygons
        try:
            inst_info = _load_json_from_zip(main_zip, REF_INSTANCE_INFO)
            stats['instance'] = {'classes': inst_info.get('classes', []), 'counts': Counter()}
            with _open_nested_zip(main_zip, REF_INSTANCE_ZIP) as inst_zip:
                for name in inst_zip.namelist():
                    if not name.lower().endswith('.json'):
                        continue
                    data = json.load(inst_zip.open(name))
                    if not data:
                        continue
                    record = data[next(iter(data))]
                    annotations = record.get('annotation', [])
                    for ann in annotations:
                        tags = ann.get('tags') or []
                        for tag in tags:
                            stats['instance']['counts'][tag] += 1
        except KeyError:
            stats['instance'] = None

        # 2D boxes
        try:
            box_info = _load_json_from_zip(main_zip, REF_BOX_INFO)
            stats['box2d'] = {'classes': box_info.get('classes', []), 'counts': Counter()}
            with _open_nested_zip(main_zip, REF_BOX_ZIP) as box_zip:
                for name in box_zip.namelist():
                    if not name.lower().endswith('.txt'):
                        continue
                    lines = box_zip.read(name).decode('utf-8').strip().splitlines()
                    for line in lines:
                        if not line:
                            continue
                        cls = line.split(',')[0].strip()
                        stats['box2d']['counts'][cls] += 1
        except KeyError:
            stats['box2d'] = None

        # motion masks
        try:
            motion_info = _load_json_from_zip(main_zip, REF_MOTION_INFO)
            stats['motion'] = {'class_names': motion_info.get('class_names', [])}
            with _open_nested_zip(main_zip, REF_MOTION_ZIP) as motion_zip:
                pngs = [n for n in motion_zip.namelist() if n.lower().endswith('.png')]
                stats['motion']['image_count'] = len(pngs)
        except KeyError:
            stats['motion'] = None

    return stats



def _extract_seg_meta(metadata):
    if not isinstance(metadata, dict):
        return set(), None
    possible_keys = ['bev_seg', 'segmentation', 'bev_segmentation', 'map_seg']
    for key in possible_keys:
        seg = metadata.get(key)
        if not isinstance(seg, dict):
            continue
        for name_key in ['class_names', 'classes', 'labels']:
            names = seg.get(name_key)
            if isinstance(names, (list, tuple)):
                return set(names), seg
    return set(), None


def _has_bev_seg(info):
    """Return True if BEV segmentation is present for a sample."""
    if isinstance(info, dict):
        if 'gt_bev_seg' in info and is_numpy_array(info['gt_bev_seg']):
            return True
        seg_path = info.get('bev_seg_path') or info.get('gt_bev_seg_path')
        if isinstance(seg_path, str) and seg_path:
            return True
        ann = info.get('ann_info')
        if isinstance(ann, dict):
            seg = ann.get('gt_bev_seg')
            if is_numpy_array(seg) or (isinstance(seg, str) and seg):
                return True
    return False

def is_numpy_array(x, ndim=None):
    ok = isinstance(x, np.ndarray)
    if ok and ndim is not None:
        ok = x.ndim == ndim
    return ok

def main(pkl_path: Path,
         obstacle_classes=OBSTACLE_CLASSES,
         marking_classes=ROAD_MARKING_CLASSES,
         drivable_classes=DRIVABLE_AREA_CLASSES,
         disable_bev_checks=False,
         reference_root: Path = None):
    with pkl_path.open('rb') as f:
        data = pickle.load(f)

    errors = []
    infos = data.get('infos')
    if not isinstance(infos, list) or len(infos) == 0:
        errors.append('缺少有效 infos 列表')
        infos = []

    metadata = data.get('metadata', {})
    seg_class_names, seg_meta = _extract_seg_meta(metadata)
    if not disable_bev_checks:
        if not seg_meta:
            errors.append('metadata 缺少 BEV/语义分割描述（期待字段如 "bev_seg" 或 "segmentation"）')
        else:
            missing_mark = marking_classes - seg_class_names
            missing_drive = drivable_classes - seg_class_names
            if missing_mark:
                errors.append(f'元信息缺少路面标识类别: {sorted(missing_mark)}')
            if missing_drive:
                errors.append(f'元信息缺少可行驶区域类别: {sorted(missing_drive)}')

            mask_shape = seg_meta.get('mask_shape')
            if mask_shape and (len(mask_shape) < 2 or mask_shape[0] <= 0 or mask_shape[1] <= 0):
                errors.append(f'元信息中的 mask_shape 非法: {mask_shape}')

    class_counter = Counter()
    obstacle_presence = Counter()
    bev_missing = 0
    bbox_counter = Counter()
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
            for cls in obstacle_classes:
                if cls in gt_names:
                    obstacle_presence[cls] += 1

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
            cat_names = annos.get('category_name')
            if is_numpy_array(cat_names, ndim=1):
                bbox_counter.update(cat_names.tolist())

        if not disable_bev_checks and not _has_bev_seg(info):
            bev_missing += 1
            errors.append(f'[{idx}] 未找到 BEV/语义分割标注（期待字段 gt_bev_seg 或 bev_seg_path）')

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
    missing_obstacles = [cls for cls in obstacle_classes if obstacle_presence[cls] == 0]
    if missing_obstacles:
        print('⚠️  数据集中缺少以下障碍物类别样本：', missing_obstacles)
    if not disable_bev_checks:
        if bev_missing:
            print(f'⚠️  有 {bev_missing} 条样本缺少 BEV/语义分割信息')
        else:
            print('BEV/语义分割标注检查通过')
        if seg_meta:
            print('语义/BEV 类别：', sorted(seg_class_names))

    if bbox_counter:
        print('2D 框类别统计 Top5:', bbox_counter.most_common(5))

    if reference_root is not None:
        try:
            ref_stats = _load_reference_stats(reference_root)
        except Exception as exc:
            print(f'⚠️  参考数据统计失败：{exc}')
        else:
            print('—— WoodScape 原始数据参考 ——')
            if ref_stats.get('instance'):
                inst = ref_stats['instance']
                print(f'实例类别总数: {len(inst["classes"])}')
                missing_inst = [c for c in inst['classes']
                                if c not in class_counter and c not in bbox_counter and c not in seg_class_names]
                if missing_inst:
                    preview = missing_inst[:20]
                    suffix = ' …' if len(missing_inst) > 20 else ''
                    print('⚠️  pkl 中缺少以下实例类别:', preview, suffix)
                print('参考实例类别 Top5:', inst['counts'].most_common(5))
            if ref_stats.get('box2d'):
                box = ref_stats['box2d']
                print(f'2D 框类别全集: {box["classes"]}')
                missing_box = [c for c in box['classes'] if c not in bbox_counter]
                if missing_box:
                    print('⚠️  pkl 缺少以下 2D 框类别:', missing_box)
                for cls in box['classes']:
                    ref_cnt = box['counts'].get(cls, 0)
                    pkl_cnt = bbox_counter.get(cls, 0)
                    print(f' - {cls}: 参考 {ref_cnt}, pkl {pkl_cnt}')
            if ref_stats.get('semantic'):
                sem = ref_stats['semantic']
                print(f'语义分割类别全集: {sem["class_names"]}')
                if not disable_bev_checks:
                    missing_sem = set(sem['class_names']) - seg_class_names
                    if missing_sem:
                        print('⚠️  metadata 缺少以下语义类别:', sorted(missing_sem))
                print(f'语义标签张数: {sem.get("image_count", 0)}, 分视角: {dict(sem.get("per_camera", {}))}')
            if ref_stats.get('motion'):
                mot = ref_stats['motion']
                print(f'动态掩码类别: {mot["class_names"]}')
                print(f'动态掩码张数: {mot.get("image_count", 0)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='检查 WoodScape info 文件是否满足 Fast-BEV 多任务训练要求')
    parser.add_argument('pkl', type=Path, help='路径，例如 data/woodscape_infos_train.pkl')
    parser.add_argument('--obstacle-classes', nargs='+', default=sorted(OBSTACLE_CLASSES),
                        help='期望包含的 3D 障碍物类别（默认: 风险三类）')
    parser.add_argument('--marking-classes', nargs='+', default=sorted(ROAD_MARKING_CLASSES),
                        help='路面标志/线段类别')
    parser.add_argument('--drivable-classes', nargs='+', default=sorted(DRIVABLE_AREA_CLASSES),
                        help='可行驶区域类别')
    parser.add_argument('--skip-bev', action='store_true',
                        help='跳过 BEV/语义分割字段检查，仅验证 3D 检测数据')
    parser.add_argument('--ref-root', type=Path,
                        help='WoodScape 原始数据目录或主压缩包路径，用于对比类别与数量')

    args = parser.parse_args()
    main(
        args.pkl,
        obstacle_classes=set(args.obstacle_classes),
        marking_classes=set(args.marking_classes),
        drivable_classes=set(args.drivable_classes),
        disable_bev_checks=args.skip_bev,
        reference_root=args.ref_root)
