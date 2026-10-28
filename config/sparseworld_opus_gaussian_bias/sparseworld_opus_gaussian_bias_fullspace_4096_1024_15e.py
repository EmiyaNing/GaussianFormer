amp = False
anno_root = 'data/nuscenes_cam/'
checkpoint_interval_steps = 250
common_pipeline = [
    dict(
        color_type='color',
        to_float32=False,
        type='LoadMultiViewImageFromFiles'),
    dict(
        color_type='color',
        num_cams=6,
        num_history=4,
        pad_history=False,
        to_float32=False,
        type='LoadMultiViewImageHistory'),
    dict(
        grid_size=0.4,
        model_coord='ego',
        occ3d_path='data/occ3d',
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
    dict(occ3d_path='data/occ3d', type='LoadSparseWorldFutureOccupancy'),
    dict(type='ResizeCropFlipImage'),
    dict(type='DefaultFormatBundle'),
    dict(num_cams=30, type='NuScenesAdaptor', use_ego=True),
]
cudnn_benchmark = False
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'experiments.sparseworld_opus_gaussian_bias',
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
dataloader_seed = 17
dataset_name_flag = 'occ3d'
enable_model_set_epoch = True
epoch_diagnostics = dict(attribute_samples=16, gradient_samples=2)
eval_every_epochs = 1
eval_mask_flag = False
evaluation_horizons = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
]
experiment_name = 'sparseworld_opus_gaussian_bias_fullspace'
find_unused_parameters = False
future_queries = [
    60,
    60,
    60,
    60,
    40,
    40,
]
future_steps = 6
grad_max_norm = 5
gradient_accumulation = 2
grid_shape = [
    200,
    200,
    16,
]
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
initialization_scope = 'image_backbone_and_neck_only'
load_from = None
loss = dict(
    current_weight=1.0,
    future_weight=0.5,
    mask_type='none',
    occupancy_loss=dict(
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
    strict_loss=dict(
        chunk_size=1024,
        class_weights=[
            3,
            2,
            3,
            2,
            2,
            3,
            3,
            2,
            3,
            2,
            2,
            1,
            2,
            1,
            1,
            1,
            1,
        ],
        empty_dist_thr=0.2,
        empty_weight=3.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        focal_weight=2.0,
        future_queries=[
            60,
            60,
            60,
            60,
            40,
            40,
        ],
        num_query=720,
        pc_range=[
            -40.0,
            -40.0,
            -1.0,
            40.0,
            40.0,
            5.4,
        ],
        point_weight=0.5,
        smooth_l1_beta=0.2,
        type='SparseWorldStrictLoss',
        voxel_size=0.4),
    type='SparseWorldOPUSGaussianBiasLoss')
loss_input_convertion = dict(
    current_pred_logits='current_pred_logits',
    current_pred_points='current_pred_points',
    future_pred_logits='future_pred_logits',
    future_pred_masks='future_pred_masks',
    future_pred_points='future_pred_points',
    gaussian_logits_by_horizon='gaussian_logits_by_horizon',
    initial_pred_points='initial_pred_points',
    pred_traj='pred_traj',
    query_stamps='query_stamps',
    sparseworld_pretrain='sparseworld_pretrain',
    stamp_statistics='stamp_statistics',
    temporal_pred_logits='temporal_pred_logits',
    temporal_pred_points='temporal_pred_points',
    update_stamp_statistics='update_stamp_statistics')
max_epochs = 15
min_lr_ratio = 0.001
model = dict(
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=dict(
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
            to_rgb=True),
        img_pad_cfg=dict(size_divisor=32)),
    feature_mode='official_auto',
    finetune_epoch=5,
    gaussian_head=dict(
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
        type='SparseWorldBiasRenderer',
        use_localagg_react=False,
        use_localaggprob=False,
        with_empty=True),
    gaussian_loss_horizons=[
        0,
        2,
        4,
        6,
    ],
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
    img_neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    is_pretrain=False,
    out_dim=256,
    pts_bbox_head=dict(
        in_channels=256,
        is_pretrain=False,
        num_classes=17,
        num_fu_frames=6,
        num_fu_query=[
            60,
            60,
            60,
            60,
            40,
            40,
        ],
        num_query=720,
        pc_range=[
            -40.0,
            -40.0,
            -1.0,
            40.0,
            40.0,
            5.4,
        ],
        test_cfg=dict(
            padding=True,
            score_thr=[
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.35,
                0.25,
                0.3,
            ]),
        train_cfg=dict(cls_weights=[
            3,
            2,
            3,
            2,
            2,
            3,
            3,
            2,
            3,
            2,
            2,
            1,
            2,
            1,
            1,
            1,
            1,
        ]),
        transformer=dict(
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
                num_frames=5,
                num_groups=4,
                num_heads=8,
                num_levels=4,
                num_points=4,
                num_refines=[
                    1,
                    4,
                    16,
                    32,
                    64,
                    128,
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
                type='SparseWorldBiasGaussianEncoder'),
            type='SparseWorldBiasGaussianTransformer'),
        type='SparseWorldBiasGaussianPerceptionHead',
        voxel_size=[
            0.4,
            0.4,
            0.4,
        ]),
    type='SparseWorldOPUSGaussianBias')
num_frames = 5
occ3d_eval_mask = 'none'
occ3d_path = 'data/occ3d'
official_checkpoint = '../SparseWorld/ckpts/epoch_56.pth'
optimizer = dict(
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            {
                'gaussian_head.empty_bias': dict(decay_mult=0.0, lr_mult=25.0),
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
print_freq = 25
report_directory = 'analysis/sparseworld_opus_gaussian_bias_fullspace_4096_1024_15e'
return_keys = [
    'img',
    'projection_mat',
    'image_wh',
    'occ_label',
    'occ_xyz',
    'occ_cam_mask',
    'occ_lidar_mask',
    'occ_nonempty_mask',
    'occ_loss_mask',
    'future_occ_labels',
    'future_occ_cam_masks',
    'future_occ_lidar_masks',
    'future_ego_to_current',
    'temporal_trajs',
    'temporal_ego_states',
    'ego2lidar',
    'img_filename',
]
scheduler_step_before_optimizer = True
seed = 17
selection_horizons = [
    0,
    2,
    4,
    6,
]
sparseworld_strict_eval = True
strict_load = True
syncBN = True
train_dataset_config = dict(
    admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
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
    future_steps=6,
    history_frames=4,
    imageset='data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl',
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
        dict(
            color_type='color',
            to_float32=False,
            type='LoadMultiViewImageFromFiles'),
        dict(
            color_type='color',
            num_cams=6,
            num_history=4,
            pad_history=False,
            to_float32=False,
            type='LoadMultiViewImageHistory'),
        dict(
            grid_size=0.4,
            model_coord='ego',
            occ3d_path='data/occ3d',
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
        dict(occ3d_path='data/occ3d', type='LoadSparseWorldFutureOccupancy'),
        dict(type='ResizeCropFlipImage'),
        dict(type='DefaultFormatBundle'),
        dict(num_cams=30, type='NuScenesAdaptor', use_ego=True),
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
        'future_occ_labels',
        'future_occ_cam_masks',
        'future_occ_lidar_masks',
        'future_ego_to_current',
        'temporal_trajs',
        'temporal_ego_states',
        'ego2lidar',
        'img_filename',
    ],
    selection_manifest=
    'data/sparseworld_opus_gaussian_bias_4096_1024/train.json',
    trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl',
    type='SparseWorldBiasSubsetDataset')
train_load_from = '/home/emiya/projects/SparseWorld/ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
train_load_revise_keys = [
    (
        'backbone.',
        'img_backbone.',
    ),
]
train_load_strict = False
train_loader = dict(batch_size=2, num_workers=4, shuffle=True)
val_dataset_config = dict(
    admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
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
    future_steps=6,
    history_frames=4,
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
        dict(
            color_type='color',
            to_float32=False,
            type='LoadMultiViewImageFromFiles'),
        dict(
            color_type='color',
            num_cams=6,
            num_history=4,
            pad_history=False,
            to_float32=False,
            type='LoadMultiViewImageHistory'),
        dict(
            grid_size=0.4,
            model_coord='ego',
            occ3d_path='data/occ3d',
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
        dict(occ3d_path='data/occ3d', type='LoadSparseWorldFutureOccupancy'),
        dict(type='ResizeCropFlipImage'),
        dict(type='DefaultFormatBundle'),
        dict(num_cams=30, type='NuScenesAdaptor', use_ego=True),
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
        'future_occ_labels',
        'future_occ_cam_masks',
        'future_occ_lidar_masks',
        'future_ego_to_current',
        'temporal_trajs',
        'temporal_ego_states',
        'ego2lidar',
        'img_filename',
    ],
    selection_manifest='data/sparseworld_opus_gaussian_bias_4096_1024/val.json',
    trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl',
    type='SparseWorldBiasSubsetDataset')
val_loader = dict(batch_size=1, num_workers=4)
voxel_size = [
    0.4,
    0.4,
    0.4,
]
warmup_iters = 500
warmup_lr_init = 6.666666666666667e-05
