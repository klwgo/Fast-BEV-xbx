# -*- coding: utf-8 -*-
"""Fast-BEV for nuScenes，多任务：HeadA(可行驶) + HeadD(障碍)。"""

point_cloud_range = [-50, -50, -5, 50, 50, 3]
voxel_size = [0.35, 0.35, 1.0]
# 实际 BEV 高度层数为 4，对应融合通道 64*6*4=1536，避免 neck 维度不匹配
n_voxels = [675, 650, 4]

bev_target_config = dict(
    drivable_positive=['drivable'],  # 基于生成的掩码，1=drivable
    drivable_ambiguous=[],
    marking_classes=[],  # 关闭标线
    slot_line_classes=[],
    slot_endpoint_classes=[],
    drivable_class_names=('non_drivable', 'drivable'),
    obstacle_classes=('obstacle',),
    point_cloud_range=point_cloud_range,
    with_slot=False,
    with_obstacle=True,
    with_occlusion=False,
)

num_views = 6  # nuScenes 默认 6 cameras
base_bev_channels = 64

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
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
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
        num_layers=5,
        stride=2,
        is_transpose=False,
        fuse=dict(
            # 实际输入通道来自体素高度4并聚合后的特征，运行时为 1024，按此对齐
            in_channels=1024,
            out_channels=base_bev_channels * num_views),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=None,
    bbox_head=None,
    bbox_head_2d=None,
    multitask_head=dict(
        type='FisheyeBEVMultiTaskHead',
        in_channels=256,
        drivable_classes=2,
        marking_classes=1,  # 占位
        slot_channels=1,
        obstacle_classes=2,
        enable_heads=dict(
            drivable=True,
            marking=False,
            slot=False,
            obstacle=True,
            occlusion=False),
        loss_drivable=dict(
            type='BCEWithLogitsLoss',
            ignore_index=255,
            weight=[0.4, 1.0],
            reduction='none',
            loss_weight=3.0),
        loss_obstacle=dict(
            type='BCEWithLogitsLoss',
            ignore_index=255,
            weight=[0.2, 1.0],
            loss_weight=5.0),
        pos_topk_ratio=0.3,
        neg_pos_ratio=3.0,
        balance_drivable=True,
        balance_obstacle=True,
        drivable_dilate_kernel=9),
    n_voxels=[n_voxels],
    voxel_size=[voxel_size],
    fisheye_lut=dict(
        camera_model='fisheye',  # 默认支持 fisheye/perspective，无需额外参数
        cache_dir=None,          # 避免多进程同时落盘导致写文件错误
        save_to_disk=False,
        force_rebuild=False,
        fusion_mode='mean',
        intrinsic_key='intrinsics',
        distortion_key='distortions',
        model_key='models'),
    train_cfg=None,
    test_cfg=dict(use_tta=False))

dataset_type = 'NuScenesMultiViewDataset'
data_root = '/mnt/new_data/nus/v1.0-trainval_full/'
train_ann = 'work_dirs/nuscenes_infos_train_bev_filtered_exist.pkl'
val_ann = 'work_dirs/nuscenes_infos_val_bev_filtered_exist.pkl'
class_names = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')

train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=1,
        expected_views=num_views,
        transforms=[dict(type='LoadImageFromFile', file_client_args=dict(backend='disk'))]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_bbox=False,
        with_label=False,
        with_bev_seg=True),
    dict(type='GenerateBEVMultitaskTargets', **bev_target_config),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D',
         class_names=class_names,
         with_gt=False,
         with_label=False),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_drivable_mask', 'gt_obstacle_mask'
        ])
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=1,
        expected_views=num_views,
        transforms=[dict(type='LoadImageFromFile', file_client_args=dict(backend='disk'))]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D',
         class_names=class_names,
         with_gt=False,
         with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=False,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        pipeline=test_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        pipeline=test_pipeline,
        classes=class_names,
        modality=dict(use_lidar=False, use_camera=True),
        test_mode=True))

optimizer = dict(type='AdamW', lr=6e-4, weight_decay=0.01)
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
evaluation = dict(interval=2, metric=['headA', 'headD'], eval_bev=False, eval_3d=False, eval_2d=False)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/nuscenes_headAD'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
fp16 = dict(loss_scale='dynamic')
