#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WoodScape 数据增强转换脚本。

这个脚本用于在已有的基础 info（通常只包含 3D 框）上，补充 WoodScape
原始数据集中提供的 2D 检测、语义 / 动态掩码等信息，以便 Fast-BEV
进行多任务联合训练。

使用方式示例：
    python tools/convert_woodscape_full.py \
        --base-info data/woodscape_infos_train.pkl \
        --raw-root /mnt/new_data/woodspace/woodscape_extracted \
        --output data/woodscape_infos_full_train.pkl \
        --bev-root /mnt/new_data/woodspace/bev_masks \
        --split train
"""

import argparse  # 解析命令行参数
import json  # 读取 json 标注
import os  # 文件路径处理
from pathlib import Path  # 更友好的路径操作
import warnings  # 输出转换过程中的提示

import mmcv  # 读写 pkl / 图像等
import numpy as np  # 数组处理

# ---------------------------------------------------------------------- #
# 常量定义：用于定位各类原始标注文件所在的子目录。
# ---------------------------------------------------------------------- #
CAMERA_ORDER = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
BBOX2D_SUBDIR = 'box_2d_annotations/box_2d_annotations'
SEMANTIC_SUBDIR = 'semantic_annotations/semantic_annotations/gtLabels'
MOTION_SUBDIR = 'motion_annotations/motion_annotations/gtLabels'
SEMANTIC_INFO_FILE = 'semantic_annotations/seg_annotation_info.json'
MOTION_INFO_FILE = 'motion_annotations/motion_annotation_info.json'
BBOX2D_CLASSES = ['vehicles', 'person', 'bicycle', 'traffic_light', 'traffic_sign']
BBOX2D_CLASS2ID = {name: idx for idx, name in enumerate(BBOX2D_CLASSES)}


def parse_args():
    """解析用户输入的参数。"""
    parser = argparse.ArgumentParser(description='整合 WoodScape 原始标注')
    parser.add_argument('--base-info', type=Path, required=True,
                        help='已有的基础 info.pkl，必须包含 3D 标注。')
    parser.add_argument('--raw-root', type=Path, required=True,
                        help='WoodScape 原始数据解压根目录，需包含 box_2d_annotations 等子目录。')
    parser.add_argument('--output', type=Path, required=True,
                        help='输出的增强版 info.pkl 路径。')
    parser.add_argument('--bev-root', type=Path, default=None,
                        help='预先计算好的 BEV 掩码目录（.npy 或 .png），可选。')
    parser.add_argument('--split', type=str, default='train',
                        help='数据拆分名称，写入 metadata 方便记录。')
    parser.add_argument('--semantic-classes', nargs='+', default=[
        'road_surface', 'free_space',
        'lane_marking', 'parking_line', 'other_ground_marking', 'zebra_crossing'
    ], help='BEV 掩码包含的语义类别顺序，默认覆盖泊车相关六类。')
    parser.add_argument('--mask-shape', type=int, nargs=3, default=[200, 200, 6],
                        help='BEV 掩码的尺寸 (H, W, C)，默认 200x200x6。')
    parser.add_argument('--overwrite', action='store_true',
                        help='若输出文件已存在，是否允许覆盖。')
    return parser.parse_args()


def read_txt_bboxes(txt_path: Path):
    """读取单个 txt 中的 2D 框信息。"""
    if not txt_path.exists():
        warnings.warn(f'未找到 2D 框文件: {txt_path}')
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    bboxes = []
    labels = []
    with txt_path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            cls_name = parts[0]
            if cls_name not in BBOX2D_CLASS2ID:
                warnings.warn(f'未知 2D 类别 {cls_name}，自动跳过')
                continue
            x1, y1, x2, y2 = map(float, parts[2:6])
            bboxes.append([x1, y1, x2, y2])
            labels.append(BBOX2D_CLASS2ID[cls_name])
    return np.asarray(bboxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def attach_multiview_annotations(info: dict,
                                 raw_root: Path,
                                 bev_root: Path,
                                 semantic_classes,
                                 mask_shape):
    """根据图片名称为单条样本补充多任务标注。"""
    ann_info = info.setdefault('ann_info', {})
    mv_bboxes = []
    mv_labels = []

    for cam_name in CAMERA_ORDER:
        cam_info = info['cams'].get(cam_name, {})
        img_path = cam_info.get('data_path', '')
        file_name = os.path.basename(img_path.replace('\\', '/'))
        stem = os.path.splitext(file_name)[0]

        bbox_txt = raw_root / BBOX2D_SUBDIR / f'{stem}.txt'
        boxes, labels = read_txt_bboxes(bbox_txt)
        mv_bboxes.append(boxes)
        mv_labels.append(labels)

        cam_ann = cam_info.setdefault('annos', {})
        cam_ann['bbox'] = boxes
        cam_ann['category_id'] = labels
        cam_ann['category_name'] = np.array([BBOX2D_CLASSES[idx] for idx in labels], dtype=object)

        semantic_png = raw_root / SEMANTIC_SUBDIR / f'{stem}.png'
        if semantic_png.exists():
            ann_info.setdefault('semantic_paths', {})[cam_name] = str(semantic_png)

        motion_png = raw_root / MOTION_SUBDIR / f'{stem}.png'
        if motion_png.exists():
            ann_info.setdefault('motion_paths', {})[cam_name] = str(motion_png)

    ann_info['mv_bboxes'] = mv_bboxes
    ann_info['mv_labels'] = mv_labels
    ann_info['bboxes'] = mv_bboxes
    ann_info['labels'] = mv_labels
    ann_info['bbox_class_names'] = list(BBOX2D_CLASSES)

    if bev_root is not None:
        bev_path_npy = bev_root / f"{info['token']}.npy"
        bev_path_png = bev_root / f"{info['token']}.png"
        if bev_path_npy.exists():
            ann_info['gt_bev_seg'] = str(bev_path_npy)
        elif bev_path_png.exists():
            ann_info['gt_bev_seg'] = str(bev_path_png)
        else:
            warnings.warn(f'未找到 BEV 掩码: {bev_path_npy} / {bev_path_png}')

    ann_info['bev_seg_classes'] = list(semantic_classes)
    ann_info['bev_mask_shape'] = list(mask_shape)
    return info


def load_semantic_metadata(raw_root: Path):
    """读取 WoodScape 官方语义标签定义."""
    info_path = raw_root / SEMANTIC_INFO_FILE
    if not info_path.exists():
        warnings.warn(f'未找到语义标签定义文件: {info_path}')
        return None
    try:
        seg_meta = mmcv.load(info_path)
    except Exception as exc:
        warnings.warn(f'读取 {info_path} 失败: {exc}')
        return None

    class_names = list(seg_meta.get('class_names', []))
    palette = seg_meta.get('class_colors') or seg_meta.get('palette')
    class_indexes = seg_meta.get('class_indexes')
    mask_shape = seg_meta.get('mask_shape')
    if not mask_shape and class_names:
        # WoodScape 原始语义 mask 与相机图像同分辨率(720x1280)
        mask_shape = [720, 1280, len(class_names)]

    return dict(
        class_names=class_names,
        palette=palette,
        class_indexes=class_indexes,
        mask_shape=mask_shape,
        source='WoodScape semantic_annotations'
    )


def load_motion_metadata(raw_root: Path):
    """读取 WoodScape 动态掩码类别."""
    info_path = raw_root / MOTION_INFO_FILE
    if not info_path.exists():
        warnings.warn(f'未找到动态掩码定义文件: {info_path}')
        return None
    try:
        motion_meta = mmcv.load(info_path)
    except Exception as exc:
        warnings.warn(f'读取 {info_path} 失败: {exc}')
        return None

    class_names = list(motion_meta.get('class_names', []))
    return dict(
        class_names=class_names,
        source='WoodScape motion_annotations'
    )


def main():
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'输出文件 {args.output} 已存在，可使用 --overwrite 覆盖')

    base = mmcv.load(args.base_info)
    infos = base.get('infos', [])
    if not infos:
        raise ValueError('基础 info 中没有 infos 字段或为空')

    raw_root = args.raw_root
    if not raw_root.exists():
        raise FileNotFoundError(f'原始数据目录不存在: {raw_root}')

    bev_root = args.bev_root
    if bev_root is not None and not bev_root.exists():
        warnings.warn(f'BEV 掩码目录 {bev_root} 不存在，忽略 BEV 标注')
        bev_root = None

    class_info_path = raw_root / 'instance_annotations/class_info.json'
    instance_classes = []
    if class_info_path.exists():
        try:
            class_meta = mmcv.load(class_info_path)
            instance_classes = class_meta.get('classes', [])
        except Exception as exc:
            warnings.warn(f'读取 {class_info_path} 失败: {exc}')
    else:
        warnings.warn(f'未找到 {class_info_path}，无法写入 instance_classes 元数据')

    semantic_meta = load_semantic_metadata(raw_root)
    motion_meta = load_motion_metadata(raw_root)

    processed_infos = []
    for info in mmcv.track_iter_progress(infos):  # 逐条样本扩充标注
        new_info = attach_multiview_annotations(
            info=info,
            raw_root=raw_root,
            bev_root=bev_root,
            semantic_classes=args.semantic_classes,
            mask_shape=args.mask_shape)
        processed_infos.append(new_info)

    metadata = dict(base.get('metadata', {}))
    metadata['split'] = args.split
    metadata.setdefault('bev_seg', {})
    metadata['bev_seg']['class_names'] = list(args.semantic_classes)
    metadata['bev_seg']['mask_shape'] = list(args.mask_shape)
    metadata['bev_seg']['source'] = 'Fast-BEV custom BEV grid'
    metadata['bbox2d_classes'] = list(BBOX2D_CLASSES)
    if instance_classes:
        metadata['instance_classes'] = list(instance_classes)
    if semantic_meta:
        metadata['segmentation'] = semantic_meta
    if motion_meta:
        metadata['motion'] = motion_meta

    output = dict(metadata=metadata, infos=processed_infos)
    mmcv.dump(output, args.output)
    print(f'成功写出增强后的 info 到 {args.output}')


if __name__ == '__main__':
    main()
