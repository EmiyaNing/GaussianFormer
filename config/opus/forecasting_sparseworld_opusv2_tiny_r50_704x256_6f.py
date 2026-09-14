"""ForecastOcc image-state prediction with shared OPUSv2-Tiny occupancy."""
_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

num_history = 5
num_future = 6
num_cams = 6
pc_range = [-40., -40., -1., 40., 40., 5.4]
grid_size = .4
data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
occ3d_path = 'data/occ3d/gts'
input_shape = (704, 256)
data_aug_conf = dict(resize_lim=(.44, .44), final_dim=input_shape[::-1],
                     bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600,
                     rand_flip=True)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    _delete_=True,
    type='ForecastingSparseWorld',
    num_history=num_history, num_future=num_future, num_cams=num_cams,
    freeze_img_backbone=True,
    init_checkpoint='./ckpts/raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth',
    forecast_layers=8, forecast_heads=16,
    # One rotating horizon per iteration is the 24 GB training recipe. At
    # evaluation the model emits all six horizons.
    train_future_horizons_per_step=1,
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=4, norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True, style='pytorch', with_cp=False),
    img_neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
                  out_channels=256, start_level=0, num_outs=4),
    head=dict(
        type='OfficialOPUSV2Head', num_classes=17, in_channels=256,
        num_query=600, pc_range=pc_range, voxel_size=[.4, .4, .4],
        pfn_channels=[64, 64], empty_label=17, score_thr=.25,
        transformer=dict(
            embed_dims=256, num_frames=1, num_views=6, num_points=4,
            num_layers=5, num_levels=4, num_groups=4,
            num_refines=[8, 16, 32, 64, 128], num_pt_channels=32,
            scales=[.5], pc_range=pc_range)))

train_pipeline = [
    # Keep decoded source images uint8. ResizeCropFlipImage converts them to
    # float32 only after the ~8x spatial downsample, avoiding a large transient
    # CPU-memory and copy-bandwidth cost.
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=num_history,
         num_cams=num_cams, to_float32=False, pad_history=False),
    dict(type='LoadMultiViewImageFuture', num_future=num_future,
         num_cams=num_cams, to_float32=False),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='LoadSparseWorldFutureOccupancy', occ3d_path=occ3d_path),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True,
         num_cams=(1 + num_history + num_future) * num_cams),
]
test_pipeline = [item for item in train_pipeline
                 if item.get('type') != 'PhotoMetricDistortionMultiViewImage']
return_keys = [
    'sample_idx', 'scene_token', 'img', 'projection_mat', 'image_wh',
    'occ_label', 'occ_xyz', 'occ_cam_mask', 'occ_lidar_mask',
    'occ_nonempty_mask', 'occ_loss_mask', 'future_occ_labels',
    'future_occ_cam_masks', 'future_occ_lidar_masks', 'future_ego_to_current',
]
dataset_common = dict(
    type='NuScenesSparseWorldTrajectoryDataset', data_root=data_root,
    data_aug_conf=data_aug_conf, pc_range=pc_range, occ3d=True,
    occ3d_coord='ego', admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
    future_steps=num_future, history_frames=num_history,
    history_keyframes_only=True, official_boundary_filter=True,
    return_keys=return_keys)
train_dataset_config = dict(
    **dataset_common, imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
    pipeline=train_pipeline, phase='train',
    trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl')
val_dataset_config = dict(
    **dataset_common, imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
    pipeline=test_pipeline, phase='val',
    trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl')
train_loader = dict(batch_size=1, num_workers=6, shuffle=True,
                    persistent_workers=True, prefetch_factor=2,
                    pin_memory=True)
val_loader = dict(batch_size=1, num_workers=4, persistent_workers=True,
                  prefetch_factor=2, pin_memory=True)

loss = dict(
    _delete_=True, type='ForecastingSparseWorldLoss',
    current_weight=1., future_weight=1.,
    latent_alignment_weight=30., pyramid_alignment_weight=10.,
    occupancy_loss=dict(
        num_classes=17, empty_label=17, pc_range=pc_range,
        class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
        focal_gamma=2., focal_alpha=.25, cls_loss_weight=2.,
        pts_loss_weight=.5, smooth_l1_beta=.2, empty_dist_thr=.2,
        empty_weight=5., rare_classes=[0, 2, 5, 8], rare_weight=10.,
        chunk_size=1024))
loss_input_convertion = dict(_delete_=True,
                             forecasting_output='forecasting_output')

optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.),
        'sampling_offset': dict(lr_mult=.1)}))
amp = True
# The forecasting attention stack occasionally overflows at 512 on long runs.
# Start lower; GradScaler still backs off automatically on isolated spikes.
amp_loss_scale = 128.
amp_growth_interval = 1000000000
recover_amp_overflow = True
max_consecutive_amp_overflows = 3
force_fp32_eval = True
fail_on_nonfinite_grad = True
grad_max_norm = 35
max_epochs = 6
eval_every_epochs = 6
warmup_iters = 50
min_lr_ratio = .01
load_from = None
epoch_start_sleep = 0
non_blocking_transfer = True
occ3d_eval_mask = 'none'
