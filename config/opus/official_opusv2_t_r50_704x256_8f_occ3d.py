"""Strict official OPUSv2-T 8-frame Occ3D evaluation configuration."""
_base_ = ['./opusv1_t_r50_704x256_8f_occ3d.py']

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
grid_shape = [200, 200, 16]
num_frames = 8

# OPUSv2 decodes 48 camera images per sample.  Keep images uint8 until the
# resize transform (which emits float32) and omit PrepareOPUSTarget: this head's
# loss consumes the dense occ_xyz/occ_label tensors directly.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=num_frames - 1,
         num_cams=6, to_float32=False, pad_history=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path='data/occ3d/gts', semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6 * num_frames),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=num_frames - 1,
         num_cams=6, to_float32=False, pad_history=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path='data/occ3d/gts', semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6 * num_frames),
]
train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)
train_loader = dict(
    num_workers=8, persistent_workers=True, prefetch_factor=1,
    pin_memory=True)
val_loader = dict(
    num_workers=4, persistent_workers=True, prefetch_factor=1,
    pin_memory=True)
non_blocking_transfer = True
epoch_start_sleep = 0

model = dict(
    _delete_=True,
    type='OfficialOPUSV2Segmentor',
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True, style='pytorch', with_cp=True),
    img_neck=dict(
        type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256,
        start_level=0, num_outs=4),
    head=dict(
        type='OfficialOPUSV2Head', num_classes=17, in_channels=256,
        num_query=600, pc_range=pc_range, voxel_size=[.4, .4, .4],
        pfn_channels=[64, 64], empty_label=17, score_thr=.25,
        transformer=dict(
            embed_dims=256, num_frames=num_frames, num_views=6, num_points=4,
            num_layers=5, num_levels=4, num_groups=4,
            num_refines=[8, 16, 32, 64, 128], num_pt_channels=32,
            scales=[.5], pc_range=pc_range)))

# Official OPUSv2 checkpoint loading is strict and handled by eval.py.
strict_checkpoint = True
checkpoint_mapping = 'official_opusv2'
strict_train_resume = True

# Official OPUSv2 training objective.  This is intentionally isolated from
# the OPUSv1/GaussianFormer losses inherited through the base config.
loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[dict(
        type='OPUSV2Loss', num_classes=17, empty_label=17,
        pc_range=pc_range,
        class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
        focal_gamma=2.0, focal_alpha=0.25, cls_loss_weight=2.0,
        pts_loss_weight=0.5, smooth_l1_beta=0.2,
        empty_dist_thr=0.2, empty_weight=5,
        rare_classes=[0, 2, 5, 8], rare_weight=10,
        chunk_size=1024)])
loss_input_convertion = dict(
    _delete_=True,
    init_points='init_points',
    all_refine_pts='all_refine_pts',
    all_cls_scores='all_cls_scores',
    all_voxel_coors='all_voxel_coors')

# Released 100e checkpoints use AdamW and a total batch of 8.  The runner
# derives the per-rank batch from world_size only for this opt-in config.
global_batch_size = 2
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1)}))
grad_max_norm = 35
amp = True
# spconv 2.x has no suitable FP16 inference kernel for the densifier's sparse
# convolution on this stack.  Training remains AMP; epoch validation matches
# the standalone evaluator and the official OPUS FP32 evaluation boundary.
force_fp32_eval = True
amp_loss_scale = 512.0
amp_growth_interval = 1000000000
fail_on_nonfinite_grad = False

lr_scheduler_type = 'official_opusv2'
warmup_iters = 500
warmup_ratio = 1.0 / 3.0
min_lr_ratio = 1e-3
max_epochs = 100
# Run the built-in validation loop after every completed training epoch.  All
# released S/M/L 8-frame variants inherit this OPUSv2-only setting.
eval_every_epochs = 1

# Use only the image backbone part of the available nuImages-pretrained
# detector, matching the official backbone-only initialization boundary.
load_from = 'ckpts/raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth'
load_only_prefixes = ['img_backbone.']
