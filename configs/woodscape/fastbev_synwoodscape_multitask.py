# -*- coding: utf-8 -*-
"""
Fast-BEV 在 SynWoodScape 上的多任务训练配置。

在 woodscape 基础配置的骨干与 head 结构上，重用 Head-A（BEV 语义）、
Head-B（路面标志）与 Head-D（多视角检测），只是在数据根目录、类别
映射与评测设置上切换到 SynWoodScape。

环境变量可覆盖默认数据路径：
    - SYN_DATA_ROOT：    SynWoodScape 图像/标注根目录
    - SYN_TRAIN_INFO：   训练集 info.pkl
    - SYN_VAL_INFO：     验证集 info.pkl
"""

import os

_base_ = ['./fastbev_woodscape_fisheye.py']

# 复用基础常量，若上游路径调整可在此覆盖
file_client_args = dict(backend='disk')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
num_views = 4
n_times = 1

# --------------------------------------------------------------------------- #
# 数据路径 / 类别定义
# --------------------------------------------------------------------------- #
syn_data_root = os.environ.get('SYN_DATA_ROOT', 'data/synwoodscape/')
syn_train_info = os.environ.get('SYN_TRAIN_INFO', 'data/synwoodscape_infos_train.pkl')
syn_val_info = os.environ.get('SYN_VAL_INFO', 'data/synwoodscape_infos_val.pkl')

# SynWoodScape 的 3D 类别沿用官方三类
class_names = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')

# Head-A / Head-B 使用的 BEV 语义类别（road/free_space + 线条）
bev_seg_classes = (
    'road_surface', 'free_space',
    'lane_marking', 'parking_line', 'other_ground_marking', 'zebra_crossing'
)

# Head-D（2D FCOS）类映射（SynWoodScape 只提供三大类）
bbox2d_classes = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')

camera_types = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
]

# --------------------------------------------------------------------------- #
# 数据处理流水线：仅加载 2D / BEV 标注（不再读取 3D 框）
# --------------------------------------------------------------------------- #
train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=n_times,
        expected_views=num_views,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args)
        ]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_bbox=True,
        with_label=True,
        with_bev_seg=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(
        type='Collect3D',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bev_seg'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info', 'img_norm_cfg',
                   'ann_info'))
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=n_times,
        expected_views=num_views,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=file_client_args)
        ]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(
        type='Collect3D',
        keys=['img'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info', 'img_norm_cfg',
                   'ann_info'))
]

# --------------------------------------------------------------------------- #
# 模型：调整 head 类别数，确保 loss 计算与 SynWoodScape 标签一致
# --------------------------------------------------------------------------- #
model = dict(
    bbox_head=None,  # 仅保留 Head-A/B/D
    n_voxels=[[520, 540, 6]],  # 维持原始 Fast-BEV 分辨率，最终由 head 按需插值到 GT 675x650
    voxel_size=[[0.5, 0.5, 1.0]],
    neck_3d=dict(out_channels=192),
    seg_head=dict(
        type='WoodscapeBEVSegHead',
        num_classes=len(bev_seg_classes),
        in_channels=192,
        mid_channels=96,
        pre_upsample=dict(size=(675, 650), mode='bilinear', align_corners=False),
        refine_head=True,
        loss_seg=dict(
            type='CrossEntropyLoss',
            class_weight=[9.0, 11.0, 8.0, 8.0, 6.0, 0.5],
            loss_weight=0.8),
        loss_aux=dict(
            type='DiceLoss',
            reduction='mean',
            loss_weight=0.25),
        loss_aux_class_weight=[9.0, 11.0, 8.0, 8.0, 6.0, 0.5],
        binary_aux_losses=[
            dict(
                class_index=2,
                ignore_index=255,
                loss_name='lane_aux',
                loss_weight=0.08,
                pos_weight=3.0),
            dict(
                class_index=3,
                ignore_index=255,
                loss_name='veg_aux',
                loss_weight=0.06,
                pos_weight=2.0),
            dict(
                class_index=4,
                ignore_index=255,
                loss_name='ground_aux',
                loss_weight=0.06,
                pos_weight=2.0),
        ]),
    bbox_head_2d=dict(num_classes=len(bbox2d_classes)),
)

# --------------------------------------------------------------------------- #
# 数据：复用基础 pipeline，但切换根目录 / info 文件 / 类别
# --------------------------------------------------------------------------- #
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        data_root=syn_data_root,
        ann_file=syn_train_info,
        classes=class_names,
        camera_types=camera_types,
        pipeline=train_pipeline,
        with_box2d=True,
    ),
    val=dict(
        data_root=syn_data_root,
        ann_file=syn_val_info,
        classes=class_names,
        camera_types=camera_types,
        pipeline=test_pipeline,
        with_box2d=True,
    ),
    test=dict(
        data_root=syn_data_root,
        ann_file=syn_val_info,
        classes=class_names,
        camera_types=camera_types,
        pipeline=test_pipeline,
        with_box2d=True,
    ),
)

# --------------------------------------------------------------------------- #
# 训练/评估设置：延续 woodscape 默认值，只是延长跑数并打开多任务评测
# --------------------------------------------------------------------------- #
runner = dict(type='EpochBasedRunner', max_epochs=40)
total_epochs = 40
evaluation = dict(interval=5, eval_2d=True, eval_3d=False, eval_bev=True, score_thr=0.0)

optimizer = dict(
    type='AdamW2',
    lr=1.5e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=25., norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=True,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-3,
    min_lr=5e-5)

custom_hooks = [
    dict(
        type='SegLossWarmupHook',
        module_attr='seg_head',
        warmup_iters=4000,
        start_scale=0.3,
        end_scale=0.8,
    )
]

del os
