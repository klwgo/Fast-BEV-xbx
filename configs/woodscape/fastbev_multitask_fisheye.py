# -*- coding: utf-8 -*-
"""Fast-BEV 多任务（Head-A/B/C）训练配置，支持 SynWoodScape / WoodScape / Hybrid."""

import os.path as osp

# ----------------------------- 数据集选择开关 -----------------------------
# 可使用 --cfg-options dataset_choice='synwoodscape' 在命令行即时切换。
dataset_choice = 'woodscape'  # 默认真实数据，可切换为 synwoodscape / hybrid

# WoodScape 默认指向 data/woodscape，若 info 缺失自动回退到 mock 数据。
woodscape_root = './data/woodscape/'
if not osp.exists(osp.join(woodscape_root, 'woodscape_infos_train.pkl')):
    woodscape_root = './data/woodscape_mock/'

point_cloud_range = [-50, -50, -5, 50, 50, 3]

dataset_registry = dict(
    synwoodscape=dict(
        data_root='./data/synwoodscape/',
        train_ann='./data/synwoodscape_infos_train.pkl',
        val_ann='./data/synwoodscape_infos_val.pkl',
        bev_classes=[
            'road_surface', 'lane_marking', 'sidewalk', 'vegetation', 'ground', 'background'
        ]),
    woodscape=dict(
        data_root=woodscape_root,
        train_ann=osp.join(woodscape_root, 'woodscape_infos_train.pkl'),
        val_ann=osp.join(woodscape_root, 'woodscape_infos_val.pkl'),
        bev_classes=[
            'road_surface', 'free_space', 'lane_marking',
            'parking_line', 'other_ground_marking', 'zebra_crossing'
        ]),
)
del osp

bev_target_config = dict(
    drivable_positive=[
        'road_surface', 'free_space', 'lane_marking', 'parking_line'
    ],
    drivable_ambiguous=['other_ground_marking', 'zebra_crossing'],
    marking_classes=[
        'lane_marking', 'parking_line', 'other_ground_marking', 'zebra_crossing'
    ],
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
)

# 混合模式：将 Syn + Wood 数据串联训练，可用 --cfg-options dataset_choice='hybrid' 启用。
def build_dataset_cfg(name, split, pipeline, test_mode):
    info = dataset_registry[name]
    ann_key = 'train_ann' if split == 'train' else 'val_ann'
    return dict(
        type='WoodScapeMultiViewDataset',
        data_root=info['data_root'],
        ann_file=info[ann_key],
        pipeline=pipeline,
        classes=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
        modality=dict(
            use_lidar=False, use_camera=True, use_radar=False,
            use_map=False, use_external=False),
        camera_types=[
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK'
        ],
        test_mode=test_mode,
        box_type_3d='LiDAR',
        sequential=False,
        n_times=1,
        with_box2d=False,
        filter_empty_gt=False,
        bev_target_generator=dict(**bev_target_config))


# ----------------------------- 模型通用配置 -----------------------------
num_views = 4
n_times = 1
base_bev_channels = 64
voxel_z = 6

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

model = dict(
    type='FastBEV',
    style='v1',
    num_views=num_views,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=None,
        style='pytorch'),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=base_bev_channels,
        num_outs=4),
    neck_fuse=dict(
        in_channels=[base_bev_channels * num_views],
        out_channels=[base_bev_channels * num_views]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=base_bev_channels * num_views,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        fuse=dict(
            in_channels=base_bev_channels * num_views * voxel_z,
            out_channels=base_bev_channels * num_views),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=None,
    bbox_head=None,
    bbox_head_2d=None,
    multitask_head=dict(
        type='FisheyeBEVMultiTaskHead',
        in_channels=256,
        drivable_classes=3,
        marking_classes=5,
        slot_channels=2,
        obstacle_classes=4,
        enable_heads=dict(
            drivable=True,
            marking=True,
            slot=False,
            obstacle=True,
            occlusion=False),
        line_kernel=5,
        loss_drivable=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
        loss_marking=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
        loss_boundary=dict(
            type='BCEWithLogitsLoss', reduction='mean', loss_weight=0.5),
        loss_slot=dict(
            type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0),
        loss_obstacle=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=1.0),
        loss_occlusion=dict(
            type='BCEWithLogitsLoss', reduction='mean', loss_weight=1.0)),
    n_voxels=[[650, 675, voxel_z]],
    voxel_size=[[0.4, 0.4, 1.0]],
    fisheye_lut=dict(
        camera_model='fisheye',
        cache_dir='./work_dirs/lut_cache',
        force_rebuild=False,
        fusion_mode='mean',
        intrinsic_key='intrinsics',
        distortion_key='distortions',
        model_key='models'),
    train_cfg=None,
    test_cfg=None)

# ----------------------------- 训练/测试流水线 -----------------------------
train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=n_times,
        expected_views=num_views,
        transforms=[dict(type='LoadImageFromFile', file_client_args=dict(backend='disk'))]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=False,
        with_label=False,
        with_bev_seg=True),
    dict(
        type='GenerateBEVMultitaskTargets',
        **bev_target_config),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D',
         class_names=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
         with_gt=False,
         with_label=False),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_drivable_mask', 'gt_marking_mask',
            'gt_marking_boundary', 'gt_obstacle_mask'
        ])
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=n_times,
        expected_views=num_views,
        transforms=[dict(type='LoadImageFromFile', file_client_args=dict(backend='disk'))]),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D',
         class_names=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
         with_gt=False,
         with_label=False),
    dict(type='Collect3D', keys=['img'])
]


def build_train_dataset(choice):
    if choice == 'hybrid':
        return dict(
            type='ConcatDataset',
            datasets=[
                build_dataset_cfg('synwoodscape', 'train', train_pipeline, False),
                build_dataset_cfg('woodscape', 'train', train_pipeline, False)
            ],
            separate_eval=False)
    return build_dataset_cfg(choice, 'train', train_pipeline, False)


def build_eval_dataset(choice, split):
    return build_dataset_cfg(choice, split, test_pipeline, True)


eval_choice = dataset_choice if dataset_choice in dataset_registry else 'woodscape'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=build_train_dataset(dataset_choice),
    val=build_eval_dataset(eval_choice, 'val'),
    test=build_eval_dataset(eval_choice, 'val'))
del build_dataset_cfg
del build_train_dataset
del build_eval_dataset

# ----------------------------- 优化与调度 -----------------------------
optimizer = dict(type='AdamW', lr=4e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/fastbev_multitask_fisheye'
load_from = None
resume_from = None
workflow = [('train', 1)]
