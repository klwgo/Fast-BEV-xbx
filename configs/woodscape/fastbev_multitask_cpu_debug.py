# -*- coding: utf-8 -*-
"""极简 CPU 自检配置：Fast-BEV 多任务 (Head-A/B/D)"""
from copy import deepcopy

# ------------------------------- 基本设定 ----------------------------------
num_views = 4
n_times = 1
base_bev_channels = 16
voxel_z = 2
point_cloud_range = [-30, -30, -4, 30, 30, 2]
norm_cfg = dict(type='BN', requires_grad=True)

# dataset 设置：直接使用 mock 数据，并通过 load_interval 抽样以减轻负担。
shared_dataset_args = dict(
    type='WoodScapeMultiViewDataset',
    data_root='data/woodscape_mock/',
    pipeline=[],  # placeholder，稍后在 data dict 中覆盖
    ann_file='data/woodscape_mock/woodscape_infos_train.pkl',
    classes=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
    modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
    camera_types=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK'],
    test_mode=False,
    box_type_3d='LiDAR',
    sequential=False,
    n_times=1,
    with_box2d=False,
    filter_empty_gt=False,
    bev_target_generator=dict(
        drivable_positive=['road_surface', 'free_space', 'lane_marking', 'parking_line'],
        drivable_ambiguous=['other_ground_marking', 'zebra_crossing'],
        marking_classes=['lane_marking', 'parking_line', 'other_ground_marking', 'zebra_crossing'],
        slot_line_classes=['parking_line'],
        slot_endpoint_classes=['zebra_crossing'],
        slot_endpoint_max_neighbors=2,
        drivable_class_names=('non_drivable', 'drivable', 'uncertain'),
        slot_channel_names=('slot_line', 'slot_endpoint'),
        obstacle_classes=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
        point_cloud_range=point_cloud_range,
        with_slot=False,
        with_obstacle=True,
        with_occlusion=False,
    ),
)

file_client_args = dict(backend='disk')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=num_views, n_times=n_times, expected_views=num_views,
         transforms=[dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_bbox=False, with_label=False, with_bev_seg=True),
    dict(type='GenerateBEVMultitaskTargets'),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=shared_dataset_args['classes'], with_gt=False, with_label=False),
    dict(type='Collect3D', keys=['img', 'gt_drivable_mask', 'gt_marking_mask', 'gt_marking_boundary', 'gt_obstacle_mask'])
]

test_pipeline = [
    dict(type='MultiViewPipeline', sequential=False, n_images=num_views, n_times=n_times, expected_views=num_views,
         transforms=[dict(type='LoadImageFromFile', file_client_args=file_client_args)]),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=shared_dataset_args['classes'], with_gt=False, with_label=False),
    dict(type='Collect3D', keys=['img'])
]

model = dict(
    type='FastBEV',
    style='v1',
    num_views=num_views,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        init_cfg=None,
        style='pytorch',
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=base_bev_channels,
        num_outs=4,
        norm_cfg=norm_cfg,
    ),
    neck_fuse=dict(in_channels=[base_bev_channels * num_views], out_channels=[base_bev_channels * num_views]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=base_bev_channels * num_views,
        out_channels=64,
        num_layers=3,
        stride=1,
        is_transpose=False,
        fuse=dict(in_channels=base_bev_channels * num_views * voxel_z, out_channels=base_bev_channels * num_views),
        norm_cfg=norm_cfg,
    ),
    seg_head=None,
    bbox_head=None,
    bbox_head_2d=None,
    multitask_head=dict(
        type='FisheyeBEVMultiTaskHead',
        in_channels=64,
        shared_channels=64,
        drivable_classes=3,
        marking_classes=5,
        slot_channels=2,
        obstacle_classes=4,
        enable_heads=dict(drivable=True, marking=True, slot=False, obstacle=True, occlusion=False),
        loss_drivable=dict(type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
        loss_marking=dict(type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
        loss_boundary=dict(type='BCEWithLogitsLoss', reduction='mean', loss_weight=0.3),
        loss_slot=dict(type='BCEWithLogitsLoss', reduction='mean', loss_weight=0.0),
        loss_obstacle=dict(type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
        loss_occlusion=dict(type='BCEWithLogitsLoss', reduction='mean', loss_weight=0.0),
    ),
    n_voxels=[[200, 200, voxel_z]],
    voxel_size=[[0.6, 0.6, 1.0]],
    fisheye_lut=dict(camera_model='fisheye', cache_dir=None, force_rebuild=False, fusion_mode='mean'),
    train_cfg=None,
    test_cfg=None,
)

optimizer = dict(type='AdamW', lr=5e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10., norm_type=2))

lr_config = dict(policy='poly', warmup=None, power=1.0, min_lr=0, by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 1)]
log_level = 'INFO'
work_dir = './work_dirs/selfcheck_multitask_cpu'
load_from = None
resume_from = None

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=deepcopy(shared_dataset_args),
    val=deepcopy(shared_dataset_args),
    test=deepcopy(shared_dataset_args),
)
data['train']['pipeline'] = train_pipeline
data['val']['pipeline'] = test_pipeline
data['test']['pipeline'] = test_pipeline

data['train']['load_interval'] = 20
data['val']['load_interval'] = 20
data['test']['load_interval'] = 20

data['val']['test_mode'] = True
data['test']['test_mode'] = True
