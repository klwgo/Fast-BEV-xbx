# -*- coding: utf-8 -*-
_base_ = ['./fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py']

file_client_args = dict(backend='disk')
data_root = './data/nuscenes/'

train_pipeline[0]['transforms'][0]['file_client_args'] = file_client_args
test_pipeline[0]['transforms'][0]['file_client_args'] = file_client_args

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        data_root=data_root,
        ann_file='data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl',
        version='v1.0-mini',
        pipeline=train_pipeline),
    val=dict(
        data_root=data_root,
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        version='v1.0-mini',
        pipeline=test_pipeline),
    test=dict(
        data_root=data_root,
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        version='v1.0-mini',
        pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=1)
total_epochs = 1
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
optimizer = dict(lr=0.0008)
lr_config = dict(warmup_iters=10)
load_from = None
