# -*- coding: utf-8 -*-
"""
Fast-BEV 在 WoodScape 4 目鱼眼数据集上的示例配置。
"""

file_client_args = dict(backend='disk')
backend_args = None

num_views = 4
n_times = 1
base_bev_channels = 64
voxel_z = 6  # default nz from n_voxels below
fuse2d_in_channels = base_bev_channels * num_views
fuse3d_in_channels = base_bev_channels * num_views * voxel_z
fuse_out_channels = base_bev_channels * num_views
camera_types = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
]

model = dict(
    type='FastBEV',
    style="v1",
    num_views=num_views,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=base_bev_channels,
        num_outs=4),
    neck_fuse=dict(
        in_channels=[fuse2d_in_channels],
        out_channels=[fuse_out_channels]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=base_bev_channels * num_views,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        fuse=dict(
            in_channels=fuse3d_in_channels,
            out_channels=fuse_out_channels
        ),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='FreeAnchor3DHead',
        is_transpose=True,
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        num_convs=0,
        use_direction_classifier=True,
        pre_anchor_topk=25,
        bbox_thr=0.5,
        gamma=2.0,
        alpha=0.5,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-50, -50, -1.8, 50, 50, -1.8]],
            sizes=[
                [0.8660, 2.5981, 1.],
                [0.5774, 1.7321, 1.],
                [1., 1., 1.],
                [0.4, 0.4, 1],
            ],
            custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    multi_scale_id=[0],
    n_voxels=[[250, 250, 6]],
    voxel_size=[[0.4, 0.4, 1.0]],
    fisheye_lut=dict(
        camera_model='fisheye',
        cache_dir='./work_dirs/lut_cache',
        force_rebuild=False,
        fusion_mode='mean'
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0] * 7 + [0.2, 0.2],
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        score_thr=0.05,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=500,
        use_scale_nms=True,
        use_tta=False,
        nms_across_levels=False,
        use_rotate_nms=True,
        nms_thr=0.2,
        nms_type_list=['rotate'] * 2 + ['circle'],
        nms_thr_list=[0.2, 0.2, 0.2],
        nms_radius_thr_list=[4, 12, 1],
        nms_rescale_factor=[1.0, 0.7, 1.0],
    )
)

point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')
dataset_type = 'WoodScapeMultiViewDataset'
data_root = './data/woodscape_mock/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

data_config = dict(
    src_size=(640, 640),
    input_size=(320, 640),
    resize=(0.0, 0.0),
    crop=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    test_input_size=(320, 640),
    test_resize=0.0,
    test_rotate=0.0,
    test_flip=False,
    pad=(0, 0, 0, 0),
    pad_divisor=32,
    pad_color=(0, 0, 0),
)

train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=n_times,
        expected_views=num_views,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=file_client_args)
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
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=num_views,
        n_times=n_times,
        expected_views=num_views,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=file_client_args)
        ]),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'woodscape_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=1,
        sequential=False,
        n_times=n_times,
        with_box2d=False,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'woodscape_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        test_mode=True,
        box_type_3d='LiDAR',
        sequential=False,
        n_times=n_times,
        with_box2d=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'woodscape_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        camera_types=camera_types,
        test_mode=True,
        box_type_3d='LiDAR',
        sequential=False,
        n_times=n_times,
        with_box2d=False),
)

optimizer = dict(
    type='AdamW2',
    lr=4e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False)

total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
evaluation = dict(interval=5)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'

load_from = None
resume_from = None
workflow = [('train', 1)]
