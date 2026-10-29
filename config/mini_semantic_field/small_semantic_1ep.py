amp = False
anno_root = 'data/nuscenes_cam/'
batch_size = 1
cuda_memory_log_interval = 50
cudnn_benchmark = False
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'loss.semantic_gaussian_field_loss',
    ])
data_aug_conf = dict(
    H=900,
    W=1600,
    bot_pct_lim=(
        0.0,
        0.0,
    ),
    final_dim=(
        256,
        704,
    ),
    rand_flip=True,
    resize_lim=(
        0.38,
        0.55,
    ),
    rot_lim=(
        0.0,
        0.0,
    ))
data_root = 'data/nuscenes/'
dataset_name_flag = 'occ3d'
eval_every_epochs = 1
expected_train_frames = 4096
expected_train_scenes = 698
expected_val_frames = 6019
expected_val_scenes = 150
fail_on_nonfinite_grad = True
field_arm = 'semantic'
field_cache = 'work_dirs/semantic_field_mini_v1/gt_cache'
finetune_base_lr = 2e-05
finetune_min_ratio = 0.1
finetune_warmup_updates = 50
grad_max_norm = 35
gradient_accumulation = 2
grid_shape = [
    200,
    200,
    16,
]
grid_size = 0.4
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
input_shape = (
    704,
    256,
)
load_from = 'work_dirs/small_camera_set_full_1ep/epoch_1.pth'
loss = dict(
    loss_cfgs=[
        dict(
            chunk_size=1024,
            class_weights=[
                10,
                5,
                10,
                5,
                5,
                10,
                10,
                5,
                10,
                5,
                5,
                1,
                5,
                1,
                1,
                2,
                1,
            ],
            empty_dist_thr=0.2,
            empty_weight=5.0,
            final_classification_weight=1.0,
            final_geometry_weight=1.0,
            final_gt_domain='camera',
            focal_gamma=2.0,
            gt_domain='camera',
            lambda_cls=2.0,
            lambda_pts=0.5,
            loss_mode='official_v1',
            pc_range=[
                -40.0,
                -40.0,
                -1.0,
                40.0,
                40.0,
                5.4,
            ],
            rare_classes=[
                0,
                2,
                5,
                8,
            ],
            rare_weight=10.0,
            smooth_l1_beta=0.2,
            stage_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='CenterSupportDiagnosticSetLoss'),
        dict(
            balance_cls_weight=True,
            empty_label=17,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756,
                1.06897009,
                1.30013094,
                1.07253735,
                0.94637502,
                1.10087012,
                1.26960524,
                1.06258364,
                1.189019,
                1.06217292,
                1.00595144,
                0.85706115,
                1.03923299,
                0.90867526,
                0.8936431,
                0.85486129,
                0.8527829,
                0.5,
            ],
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0, loss_voxel_lovasz_weight=1.0),
            num_classes=18,
            type='OccupancyLoss',
            use_dice_loss=False,
            use_focal_loss=False,
            use_lovasz_loss=True,
            use_sem_geo_scal_loss=False,
            weight=1.0),
        dict(
            mode='semantic',
            seed=928,
            type='SemanticGaussianFieldLoss',
            weight=0.5),
    ],
    type='MultiLoss')
loss_input_convertion = dict(
    gaussian='gaussian',
    occ_mask='occ_cam_mask',
    opus_point_valid_masks='opus_point_valid_masks',
    opus_pred_logits='opus_pred_logits',
    opus_pred_points='opus_pred_points',
    pred_occ='pred_occ',
    sampled_label='sampled_label',
    sampled_xyz='sampled_xyz')
max_epochs = 1
max_gt_points = 76800
min_lr_ratio = 0.1
model = dict(
    encoder=dict(
        attribute_stages=[
            5,
        ],
        decoder_checkpoint=True,
        dropout=0.1,
        embed_dims=256,
        feedforward_channels=512,
        initial_opacity=0.1,
        initial_scale=0.2,
        num_classes=17,
        num_decoder=6,
        num_frames=8,
        num_groups=4,
        num_heads=8,
        num_levels=4,
        num_points=2,
        num_refines=[
            1,
            4,
            8,
            16,
            32,
            64,
        ],
        num_views=6,
        pc_range=[
            -40.0,
            -40.0,
            -1.0,
            40.0,
            40.0,
            5.4,
        ],
        scale_max=0.4,
        scale_min=0.08,
        scales=[
            0.5,
        ],
        type='OPUSV1GaussianTemporalEncoder'),
    head=dict(
        apply_loss_type='fixed_5',
        cuda_kwargs=dict(
            D=16,
            H=200,
            W=200,
            grid_size=0.4,
            pc_min=[
                -40.0,
                -40.0,
                -1.0,
            ],
            scale_multiplier=3),
        dataset_type='nusc',
        empty_args=dict(mean=[
            0.0,
            0.0,
            -1.0,
        ], scale=[
            100.0,
            100.0,
            8.0,
        ]),
        empty_label=17,
        empty_logit_offset=0.0,
        initial_empty_bias=0.0,
        max_render_scale=0.4,
        num_classes=18,
        type='OPUSV1GaussianTrainableBiasHead',
        use_localagg_react=False,
        use_localaggprob=False,
        with_empty=True),
    image_amp_dtype='bfloat16',
    image_checkpoint=True,
    image_chunk_size=6,
    img_backbone=dict(
        depth=50,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=True, type='BN2d'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet',
        with_cp=False),
    img_backbone_out_indices=[
        0,
        1,
        2,
        3,
    ],
    img_neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type='FPN'),
    lifter=dict(
        embed_dims=256,
        learnable_features=False,
        num_queries=1200,
        reference_mode='direct',
        type='OPUSQueryLifter'),
    type='OPUSTemporalSegmentor')
num_frames = 8
occ3d_eval_mask = 'camera'
occ3d_path = 'data/occ3d/gts'
occ3d_return_keys = [
    'img',
    'projection_mat',
    'image_wh',
    'occ_label',
    'occ_xyz',
    'occ_cam_mask',
    'occ_lidar_mask',
    'occ_nonempty_mask',
    'occ_loss_mask',
]
optimizer = dict(
    optimizer=dict(lr=2e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict({
            'head.empty_bias': dict(decay_mult=0.0, lr_mult=25.0),
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1)
        })))
pc_range = [
    -40.0,
    -40.0,
    -1.0,
    40.0,
    40.0,
    5.4,
]
print_freq = 50
save_every_updates = 128
scheduler_step_before_optimizer = True
seed = 928
strict_load = True
test_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(
        interval=(
            4,
            8,
        ),
        num_cams=6,
        num_history=7,
        test_mode=True,
        to_float32=True,
        type='LoadOPUSV1ImageHistory'),
    dict(
        grid_size=0.4,
        model_coord='ego',
        occ3d_path='data/occ3d/gts',
        pc_range=[
            -40.0,
            -40.0,
            -1.0,
            40.0,
            40.0,
            5.4,
        ],
        semantic=True,
        train_mask_type='none',
        type='LoadOccupancyOcc3D'),
    dict(type='ResizeCropFlipImage'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    dict(type='DefaultFormatBundle'),
    dict(num_cams=48, type='NuScenesAdaptor', use_ego=True),
]
train_dataset_config = dict(
    data_aug_conf=dict(
        H=900,
        W=1600,
        bot_pct_lim=(
            0.0,
            0.0,
        ),
        final_dim=(
            256,
            704,
        ),
        rand_flip=True,
        resize_lim=(
            0.38,
            0.55,
        ),
        rot_lim=(
            0.0,
            0.0,
        )),
    data_root='data/nuscenes/',
    imageset='data/mini_occ3d_4096_1024/train.pkl',
    occ3d=True,
    occ3d_coord='ego',
    pc_range=[
        -40.0,
        -40.0,
        -1.0,
        40.0,
        40.0,
        5.4,
    ],
    phase='train',
    pipeline=[
        dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
        dict(
            interval=(
                4,
                8,
            ),
            num_cams=6,
            num_history=7,
            test_mode=False,
            to_float32=True,
            type='LoadOPUSV1ImageHistory'),
        dict(
            grid_size=0.4,
            model_coord='ego',
            occ3d_path='data/occ3d/gts',
            pc_range=[
                -40.0,
                -40.0,
                -1.0,
                40.0,
                40.0,
                5.4,
            ],
            semantic=True,
            train_mask_type='none',
            type='LoadOccupancyOcc3D'),
        dict(type='ResizeCropFlipImage'),
        dict(type='PhotoMetricDistortionMultiViewImage'),
        dict(
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            to_rgb=True,
            type='NormalizeMultiviewImage'),
        dict(type='DefaultFormatBundle'),
        dict(num_cams=48, type='NuScenesAdaptor', use_ego=True),
    ],
    return_keys=[
        'img',
        'projection_mat',
        'image_wh',
        'occ_label',
        'occ_xyz',
        'occ_cam_mask',
        'occ_lidar_mask',
        'occ_nonempty_mask',
        'occ_loss_mask',
    ],
    type='NuScenesDataset')
train_load_from = 'work_dirs/small_camera_set_full_1ep/epoch_1.pth'
train_load_revise_keys = []
train_load_strict = True
train_loader = dict(batch_size=1, num_workers=3, shuffle=True)
train_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(
        interval=(
            4,
            8,
        ),
        num_cams=6,
        num_history=7,
        test_mode=False,
        to_float32=True,
        type='LoadOPUSV1ImageHistory'),
    dict(
        grid_size=0.4,
        model_coord='ego',
        occ3d_path='data/occ3d/gts',
        pc_range=[
            -40.0,
            -40.0,
            -1.0,
            40.0,
            40.0,
            5.4,
        ],
        semantic=True,
        train_mask_type='none',
        type='LoadOccupancyOcc3D'),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='NormalizeMultiviewImage'),
    dict(type='DefaultFormatBundle'),
    dict(num_cams=48, type='NuScenesAdaptor', use_ego=True),
]
val_dataset_config = dict(
    data_aug_conf=dict(
        H=900,
        W=1600,
        bot_pct_lim=(
            0.0,
            0.0,
        ),
        final_dim=(
            256,
            704,
        ),
        rand_flip=True,
        resize_lim=(
            0.38,
            0.55,
        ),
        rot_lim=(
            0.0,
            0.0,
        )),
    data_root='data/nuscenes/',
    imageset='data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl',
    occ3d=True,
    occ3d_coord='ego',
    pc_range=[
        -40.0,
        -40.0,
        -1.0,
        40.0,
        40.0,
        5.4,
    ],
    phase='val',
    pipeline=[
        dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
        dict(
            interval=(
                4,
                8,
            ),
            num_cams=6,
            num_history=7,
            test_mode=True,
            to_float32=True,
            type='LoadOPUSV1ImageHistory'),
        dict(
            grid_size=0.4,
            model_coord='ego',
            occ3d_path='data/occ3d/gts',
            pc_range=[
                -40.0,
                -40.0,
                -1.0,
                40.0,
                40.0,
                5.4,
            ],
            semantic=True,
            train_mask_type='none',
            type='LoadOccupancyOcc3D'),
        dict(type='ResizeCropFlipImage'),
        dict(
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            to_rgb=True,
            type='NormalizeMultiviewImage'),
        dict(type='DefaultFormatBundle'),
        dict(num_cams=48, type='NuScenesAdaptor', use_ego=True),
    ],
    return_keys=[
        'img',
        'projection_mat',
        'image_wh',
        'occ_label',
        'occ_xyz',
        'occ_cam_mask',
        'occ_lidar_mask',
        'occ_nonempty_mask',
        'occ_loss_mask',
    ],
    type='NuScenesDataset')
val_loader = dict(batch_size=1, num_workers=4)
warmup_iters = 400
warmup_lr_init = 2e-06
warmup_ratio = 0.3333333333333333
work_dir = None
