# -*- coding: utf-8 -*-
"""
Fast-BEV 在 SynWoodScape 仿真数据上的预训练配置。

相较于 woodscape 微调配置：
    - 禁用 BEV 语义分支（合成数据未提供一致的 BEV 标签）
    - 覆盖数据管线，移除 gt_bev_seg 相关字段
    - 默认指向 `data/synwoodscape_infos_*.pkl`，请在实际训练前生成相应 pkl
"""

import os

_base_ = ['./fastbev_woodscape_fisheye.py']

# 复用基础配置的关键常量。若后续基础配置有改动，请同步更新这里。
file_client_args = dict(backend='disk')
num_views = 4
n_times = 1
# SynWoodScape 仿真环境中的 3D 框分布远超过默认 [-50, 50] 覆盖范围。
# 依据 tools/debug_woodscape_samples.py 的统计，扩大至约 ±130m，y 轴上限 170m。
point_cloud_range = [-130, -100, -2, 130, 170, 6]
class_names = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')
dataset_type = 'WoodScapeMultiViewDataset'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)
camera_types = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
bbox2d_classes = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')
bev_seg_classes = (
    'road_surface',
    'lane_marking',
    'sidewalk',
    'vegetation',
    'ground',
    'background',
)
bn_norm_cfg = dict(type='BN', requires_grad=True)

# --------------------------------------------------------------------------- #
# 数据路径：支持通过环境变量覆盖，便于在不同服务器间切换
# --------------------------------------------------------------------------- #
syn_data_root = os.environ.get(
    'SYN_DATA_ROOT',
    '/mnt/new_data/woodspace/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/')
syn_train_info = os.environ.get(
    'SYN_TRAIN_INFO',
    'data/synwoodscape_infos_train.pkl')
syn_val_info = os.environ.get(
    'SYN_VAL_INFO',
    'data/synwoodscape_infos_val.pkl')

# --------------------------------------------------------------------------- #
# 模型：同步更新 2D/BEV 头部类别数
# --------------------------------------------------------------------------- #
model = dict(
    with_cp=False,
    bbox_head_2d=None,
    seg_head=None,
    bbox_head=dict(
        pre_anchor_topk=16,
        bbox_thr=0.4,
        alpha=0.35,
        pos_loss_weight=0.5,
        neg_loss_weight=0.5,
        assigner_per_size=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-130, -100, -2.0, 130, 170, 6.0]],
            sizes=[
                # 行人：参考均值 (0.8,0.85,1.7)，提供紧/松两个尺寸
                [0.7, 0.8, 1.6],
                [0.9, 1.0, 1.9],
                # 两轮：均值 (0.8,1.46,1.46)
                [0.7, 1.2, 1.3],
                [0.9, 1.8, 1.6],
                # 四轮：均值 (1.57,3.07,1.71)，覆盖小轿车到大货车
                [1.2, 2.4, 1.6],
                [1.6, 3.4, 1.8],
                [2.1, 4.8, 2.1],
                [2.6, 6.8, 2.4],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        # ------------------------------
        # 强化 3D 检测分支的监督：相比默认配置放大 loss 权重，
        # 让优化器更多关注 3D 正样本的学习，缓解 AP 长期为 0 的问题。
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),      # 降低权重，避免梯度爆炸
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=0.6),
        loss_dir=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.8)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.35,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0] * 7 + [0.2, 0.2],
        pos_weight=-1,
        debug=False),
    n_voxels=[[520, 540, 6]],
    voxel_size=[[0.5, 0.5, 1.0]],
    backbone=dict(
        norm_cfg=bn_norm_cfg,
        norm_eval=False),
    neck=dict(norm_cfg=bn_norm_cfg),
    neck_3d=dict(norm_cfg=bn_norm_cfg),
)

# --------------------------------------------------------------------------- #
# 数据处理流水线：加载 3D、2D、BEV 全量监督
# --------------------------------------------------------------------------- #
syn_train_pipeline = [
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
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bev_seg=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info', 'img_norm_cfg',
                   'ann_info', 'box_type_3d', 'box_mode_3d'))
]

syn_test_pipeline = [
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
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bev_seg=False),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info', 'img_norm_cfg',
                   'ann_info', 'box_type_3d', 'box_mode_3d'))
]

# --------------------------------------------------------------------------- #
# 数据集设置：指向 SynWoodScape 预训练 pkl
# --------------------------------------------------------------------------- #
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=syn_data_root,
        ann_file=syn_train_info,
        pipeline=syn_train_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        with_box2d=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=syn_data_root,
        ann_file=syn_val_info,
        pipeline=syn_test_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        with_box2d=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=syn_data_root,
        ann_file=syn_val_info,
        pipeline=syn_test_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        with_box2d=False,
    ),
)

# --------------------------------------------------------------------------- #
# 训练超参调节：降低学习率、防止梯度爆炸
# --------------------------------------------------------------------------- #
optimizer = dict(
    type='AdamW2',
    lr=2e-5,
    betas=(0.95, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))

fp16 = dict(loss_scale='dynamic')

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=8000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=5e-7,
    by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=40)
total_epochs = 40

evaluation = dict(interval=5, eval_2d=False, eval_3d=True, eval_bev=False, score_thr=0.0)

del os
