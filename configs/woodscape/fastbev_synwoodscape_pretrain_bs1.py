file_client_args = dict(backend='disk')
backend_args = None
num_views = 4
n_times = 1
base_bev_channels = 64
voxel_z = 6
fuse2d_in_channels = 256
fuse3d_in_channels = 1536
fuse_out_channels = 256
camera_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')
dataset_type = 'WoodScapeMultiViewDataset'
data_root = './data/woodscape_mock/'
bev_seg_classes = ('road_surface', 'lane_marking', 'sidewalk', 'vegetation',
                   'ground', 'background')
bbox2d_classes = ('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle')
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
    type='FastBEV',
    style='v1',
    num_views=4,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/Data/xvboxun/xbx/Fast-BEV-Fish/pretrained_models/resnet50-0676ba61.pth'
        ),
        style='pytorch'),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256], out_channels=[256]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=256,
        out_channels=256,
        num_layers=6,
        stride=2,
        is_transpose=False,
        fuse=dict(in_channels=1536, out_channels=256),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=dict(
        type='WoodscapeBEVSegHead',
        in_channels=256,
        num_classes=6,
        mid_channels=128,
        num_convs=2,
        loss_seg=dict(type='CrossEntropyLoss', loss_weight=1.0)),
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
            sizes=[[0.866, 2.5981, 1.0], [0.5774, 1.7321, 1.0],
                   [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
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
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=0.8),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8)),
    bbox_head_2d=dict(
        type='FCOSHead',
        num_classes=3,
        in_channels=64,
        stacked_convs=2,
        feat_channels=64,
        strides=[4, 8, 16, 32],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 100000000.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg_2d=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg_2d=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100),
    multi_scale_id=[0],
    n_voxels=[[250, 250, 6]],
    voxel_size=[[0.4, 0.4, 1.0]],
    fisheye_lut=dict(
        camera_model='fisheye',
        cache_dir='./work_dirs/lut_cache',
        force_rebuild=False,
        fusion_mode='mean',
        intrinsic_key='intrinsics',
        distortion_key='distortions',
        model_key='models'),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
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
        nms_type_list=['rotate', 'rotate', 'circle'],
        nms_thr_list=[0.2, 0.2, 0.2],
        nms_radius_thr_list=[4, 12, 1],
        nms_rescale_factor=[1.0, 0.7, 1.0]),
    with_cp=False)
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
    pad_color=(0, 0, 0))
train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=4,
        n_times=1,
        expected_views=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'))
        ]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bev_seg=True),
    dict(type='KittiSetOrigin', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('pedestrian', 'four-wheeler vehicle',
                     'two-wheeler vehicle')),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_bev_seg'
        ])
]
test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=4,
        n_times=1,
        expected_views=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'))
        ]),
    dict(type='KittiSetOrigin', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('pedestrian', 'four-wheeler vehicle',
                     'two-wheeler vehicle'),
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='WoodScapeMultiViewDataset',
        data_root=
        '/mnt/new_data/woodspace/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/',
        ann_file='data/synwoodscape_infos_train.pkl',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                sequential=False,
                n_images=4,
                n_times=1,
                expected_views=4,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk'))
                ]),
            dict(type='PrepareSynWoodscape2DTargets'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_bev_seg=True),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('pedestrian', 'four-wheeler vehicle',
                             'two-wheeler vehicle')),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d',
                    'gt_labels_3d', 'gt_bev_seg'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'lidar2img', 'img_info'))
        ],
        classes=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        camera_types=[
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK'
        ],
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=1,
        sequential=False,
        n_times=1,
        with_box2d=True,
        filter_empty_gt=False),
    val=dict(
        type='WoodScapeMultiViewDataset',
        data_root=
        '/mnt/new_data/woodspace/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/',
        ann_file='data/synwoodscape_infos_val.pkl',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                sequential=False,
                n_images=4,
                n_times=1,
                expected_views=4,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk'))
                ]),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('pedestrian', 'four-wheeler vehicle',
                             'two-wheeler vehicle'),
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        camera_types=[
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK'
        ],
        test_mode=True,
        box_type_3d='LiDAR',
        sequential=False,
        n_times=1,
        with_box2d=True),
    test=dict(
        type='WoodScapeMultiViewDataset',
        data_root=
        '/mnt/new_data/woodspace/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/',
        ann_file='data/synwoodscape_infos_val.pkl',
        pipeline=[
            dict(
                type='MultiViewPipeline',
                sequential=False,
                n_images=4,
                n_times=1,
                expected_views=4,
                transforms=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk'))
                ]),
            dict(
                type='KittiSetOrigin',
                point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('pedestrian', 'four-wheeler vehicle',
                             'two-wheeler vehicle'),
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=('pedestrian', 'four-wheeler vehicle', 'two-wheeler vehicle'),
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        camera_types=[
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK'
        ],
        test_mode=True,
        box_type_3d='LiDAR',
        sequential=False,
        n_times=1,
        with_box2d=True))
optimizer = dict(
    type='AdamW2',
    lr=0.0004,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0))))
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0,
    by_epoch=False)
total_epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
evaluation = dict(interval=5)
dist_params = dict(backend='nccl')
find_unused_parameters = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
environ = environ({
    'SHELL':
    '/bin/bash',
    'SESSION_MANAGER':
    'local/will-Precision-5820-Tower-X-Series:@/tmp/.ICE-unix/4845,unix/will-Precision-5820-Tower-X-Series:/tmp/.ICE-unix/4845',
    'QT_ACCESSIBILITY':
    '1',
    'XDG_CONFIG_DIRS':
    '/etc/xdg/xdg-ubuntu:/etc/xdg',
    'XDG_MENU_PREFIX':
    'gnome-',
    'GNOME_DESKTOP_SESSION_ID':
    'this-is-deprecated',
    'GTK_IM_MODULE':
    'fcitx',
    'CONDA_EXE':
    '/home/will/anaconda3/bin/conda',
    'GNOME_SHELL_SESSION_MODE':
    'ubuntu',
    'SSH_AUTH_SOCK':
    '/run/user/1000/keyring/ssh',
    'MEMORY_PRESSURE_WRITE':
    'c29tZSAyMDAwMDAgMjAwMDAwMAA=',
    'ELECTRON_RUN_AS_NODE':
    '1',
    'XMODIFIERS':
    '@im=fcitx',
    'DESKTOP_SESSION':
    'ubuntu',
    'GTK_MODULES':
    'gail:atk-bridge',
    'PWD':
    '/mnt/auto/preception/Fastbev/Fast-BEV-Fish',
    'GSETTINGS_SCHEMA_DIR':
    '/home/will/anaconda3/share/glib-2.0/schemas',
    'XDG_SESSION_DESKTOP':
    'ubuntu',
    'LOGNAME':
    'will',
    'XDG_SESSION_TYPE':
    'x11',
    'CONDA_PREFIX':
    '/home/will/anaconda3',
    'VSCODE_ESM_ENTRYPOINT':
    'vs/workbench/api/node/extensionHostProcess',
    'GPG_AGENT_INFO':
    '/run/user/1000/gnupg/S.gpg-agent:0:1',
    'SYSTEMD_EXEC_PID':
    '4877',
    'VSCODE_CODE_CACHE_PATH':
    '/home/will/.config/Code/CachedData/e3a5acfb517a443235981655413d566533107e92',
    'XAUTHORITY':
    '/run/user/1000/gdm/Xauthority',
    'GJS_DEBUG_TOPICS':
    'JS ERROR;JS LOG',
    'GNOME_ACCESSIBILITY':
    '1',
    'WINDOWPATH':
    '2',
    'HOME':
    '/home/will',
    'USERNAME':
    'will',
    'IM_CONFIG_PHASE':
    '1',
    'LANG':
    'zh_CN.UTF-8',
    'XDG_CURRENT_DESKTOP':
    'Unity',
    'VSCODE_IPC_HOOK':
    '/run/user/1000/vscode-1633ec6f-1.10-main.sock',
    'MEMORY_PRESSURE_WATCH':
    '/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/session.slice/org.gnome.Shell@x11.service/memory.pressure',
    'CONDA_PROMPT_MODIFIER':
    '(base) ',
    'QTWEBENGINE_DICTIONARIES_PATH':
    '/usr/share/hunspell-bdic/',
    'INVOCATION_ID':
    'fab3cbb8a14146d89209f4d81ff466d9',
    'MANAGERPID':
    '4301',
    'CHROME_DESKTOP':
    'code.desktop',
    'CLUTTER_IM_MODULE':
    'xim',
    'GJS_DEBUG_OUTPUT':
    'stderr',
    'ACCESSIBILITY_ENABLED':
    '1',
    'SDL_IM_MODULE':
    'fcitx',
    'LESSCLOSE':
    '/usr/bin/lesspipe %s %s',
    'XDG_SESSION_CLASS':
    'user',
    'LESSOPEN':
    '| /usr/bin/lesspipe %s',
    'USER':
    'will',
    'NO_PROXY':
    'http://127.0.0.1:7890',
    'CONDA_SHLVL':
    '1',
    'DISPLAY':
    ':1',
    'VSCODE_PID':
    '5981',
    'SHLVL':
    '1',
    'GSM_SKIP_SSH_AGENT_WORKAROUND':
    'true',
    'HTTPS_PROXY':
    'http://127.0.0.1:7890',
    'HTTP_PROXY':
    'http://127.0.0.1:7890',
    'QT_IM_MODULE':
    'fcitx',
    'VSCODE_CWD':
    '/home/will',
    'CONDA_PYTHON_EXE':
    '/home/will/anaconda3/bin/python',
    'VSCODE_CRASH_REPORTER_PROCESS_TYPE':
    'extensionHost',
    'XDG_RUNTIME_DIR':
    '/run/user/1000',
    'CONDA_DEFAULT_ENV':
    'base',
    'DEBUGINFOD_URLS':
    'https://debuginfod.ubuntu.com ',
    'CODEX_INTERNAL_ORIGINATOR_OVERRIDE':
    'codex_vscode',
    'JOURNAL_STREAM':
    '9:27921',
    'XDG_DATA_DIRS':
    '/usr/share/ubuntu:/usr/share/gnome:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop',
    'GDK_BACKEND':
    'x11',
    'PATH':
    '/home/will/.local/bin:/tmp/.tmpm9ABpi:/home/will/.local/bin:/home/will/.local/bin:/home/will/bin:/home/.../bin:/home/will/anaconda3/bin:/home/will/anaconda3/condabin:/home/will/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/home/will/.vscode/extensions/openai.chatgpt-0.4.19-linux-x64/bin/linux-x86_64',
    'GDMSESSION':
    'ubuntu',
    'ORIGINAL_XDG_CURRENT_DESKTOP':
    'ubuntu:GNOME',
    'DBUS_SESSION_BUS_ADDRESS':
    'unix:path=/run/user/1000/bus',
    'VSCODE_NLS_CONFIG':
    '{"userLocale":"zh-cn","osLocale":"zh-cn","resolvedLanguage":"zh-cn","defaultMessagesFile":"/usr/share/code/resources/app/out/nls.messages.json","languagePack":{"translationsConfigFile":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn/tcf.json","messagesFile":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn/e3a5acfb517a443235981655413d566533107e92/nls.messages.json","corruptMarkerFile":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn/corrupted.info"},"locale":"zh-cn","availableLanguages":{"*":"zh-cn"},"_languagePackId":"f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn","_languagePackSupport":true,"_translationsConfigFile":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn/tcf.json","_cacheRoot":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn","_resolvedLanguagePackCoreLocation":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn/e3a5acfb517a443235981655413d566533107e92","_corruptedFile":"/home/will/.config/Code/clp/f2c0f9b473c8dfd78c9b4e1173323dea.zh-cn/corrupted.info"}',
    'RUST_LOG':
    'warn',
    'GIO_LAUNCHED_DESKTOP_FILE_PID':
    '5981',
    'GIO_LAUNCHED_DESKTOP_FILE':
    '/usr/share/applications/code.desktop',
    'VSCODE_HANDLES_UNCAUGHT_ERRORS':
    'true',
    '_':
    '/home/will/anaconda3/bin/python',
    'QT_QPA_PLATFORM_PLUGIN_PATH':
    '/home/will/.local/lib/python3.12/site-packages/cv2/qt/plugins',
    'QT_QPA_FONTDIR':
    '/home/will/.local/lib/python3.12/site-packages/cv2/qt/fonts',
    'LD_LIBRARY_PATH':
    '/home/will/.local/lib/python3.12/site-packages/cv2/../../lib64:'
})
syn_data_root = '/mnt/new_data/woodspace/synwoodscape/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/'
syn_train_info = 'data/synwoodscape_infos_train.pkl'
syn_val_info = 'data/synwoodscape_infos_val.pkl'
syn_train_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=4,
        n_times=1,
        expected_views=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'))
        ]),
    dict(type='PrepareSynWoodscape2DTargets'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bev_seg=True),
    dict(type='KittiSetOrigin', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('pedestrian', 'four-wheeler vehicle',
                     'two-wheeler vehicle')),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_bev_seg'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'lidar2img', 'img_info'))
]
syn_test_pipeline = [
    dict(
        type='MultiViewPipeline',
        sequential=False,
        n_images=4,
        n_times=1,
        expected_views=4,
        transforms=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'))
        ]),
    dict(type='KittiSetOrigin', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('pedestrian', 'four-wheeler vehicle',
                     'two-wheeler vehicle'),
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]
