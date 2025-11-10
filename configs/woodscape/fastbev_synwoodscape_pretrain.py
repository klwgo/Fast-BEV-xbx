# -*- coding: utf-8 -*-
"""
Fast-BEV 在 SynWoodScape 仿真数据上的预训练配置。

相较于 woodscape 微调配置：
    - 禁用 BEV 语义分支（合成数据未提供一致的 BEV 标签）
    - 覆盖数据管线，移除 gt_bev_seg 相关字段
    - 默认指向 `data/synwoodscape_infos_*.pkl`，请在实际训练前生成相应 pkl
"""

from os import environ

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
syn_data_root = environ.get(
    'SYN_DATA_ROOT',
    '/mnt/new_data/woodspace/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/')
syn_train_info = environ.get(
    'SYN_TRAIN_INFO',
    'data/synwoodscape_infos_train.pkl')
syn_val_info = environ.get(
    'SYN_VAL_INFO',
    'data/synwoodscape_infos_val.pkl')

# --------------------------------------------------------------------------- #
# 模型：同步更新 2D/BEV 头部类别数
# --------------------------------------------------------------------------- #
model = dict(
    with_cp=False,
    bbox_head_2d=dict(num_classes=len(bbox2d_classes)),
    seg_head=dict(num_classes=len(bev_seg_classes)),
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
    dict(type='PrepareSynWoodscape2DTargets'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bev_seg=True),
    dict(type='LoadSynWoodscapeBEVSeg'),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
dict(
        type='Collect3D',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bev_seg',
              'mv_bboxes', 'mv_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info',
                   'mv_bboxes', 'mv_labels', 'ann_info',
                   'box_type_3d', 'box_mode_3d'))
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
    dict(type='PrepareSynWoodscape2DTargets'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bev_seg=True),
    dict(type='LoadSynWoodscapeBEVSeg'),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
              'gt_bev_seg', 'mv_bboxes', 'mv_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info',
                   'mv_bboxes', 'mv_labels', 'ann_info',
                   'box_type_3d', 'box_mode_3d'))
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
        with_box2d=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=syn_data_root,
        ann_file=syn_val_info,
        pipeline=syn_test_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        with_box2d=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=syn_data_root,
        ann_file=syn_val_info,
        pipeline=syn_test_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        with_box2d=True,
    ),
)

# --------------------------------------------------------------------------- #
# 训练超参调节：降低学习率、防止梯度爆炸
# --------------------------------------------------------------------------- #
optimizer = dict(
    type='AdamW2',
    lr=3e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=10.0, norm_type=2))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False)

evaluation = dict(interval=5, eval_2d=True, eval_3d=True, eval_bev=True, score_thr=0.0)
