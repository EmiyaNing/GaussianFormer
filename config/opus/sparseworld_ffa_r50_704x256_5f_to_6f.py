"""24 GiB single-GPU recipe for SparseWorld-FFA at future keyframe h=6."""

_base_ = ['./sparseworld_strict_r50_704x256_5f_occ3d.py']

target_horizon_index = 6
keyframe_interval_seconds = 0.5
target_horizon_seconds = 3.0
ffa_contract_version = 2
future_prediction_coordinate = 'current_ego'
num_student_frames = 5
num_future = 6
num_cams = 6

teacher_config = (
    'config/opus/'
    'opusv1_t_r50_704x256_8f_720q48p_nomask_occ3d_20e.py')
teacher_checkpoint = 'ckpts/opusv1_720query.pth'
teacher_checkpoint_sha256 = (
    '1d85e429f83792ba7b2a84d8a37446a0736124859bd55d79400d1ecf53132fce')
teacher_prefer_ema = True
# FP32 no-grad teacher inference is the conservative default.  It is executed
# before the student and retains only the small final sparse state.
teacher_amp = False

student_init_checkpoint = (
    'out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth')
student_init_checkpoint_sha256 = (
    'fb9f955f15362cb23244fd551145e499b6a28b1a46e45849b129d9b60a879828')

model = dict(
    type='SparseWorldFFASegmentor',
    target_horizon_index=target_horizon_index,
    num_student_frames=num_student_frames,
    freeze_student_perception=True,
    # The original SparseWorld pretrain curriculum is not used: this model
    # starts from the completed epoch-56 SparseWorld baseline.
    finetune_epoch=0,
    ffa=dict(
        embed_dims=256,
        num_points=48,
        num_layers=3,
        num_heads=8,
        ffn_dims=512,
        dropout=0.1,
        alignment_dims=128,
        max_translation_m=12.0,
        max_deformation_m=2.0,
        checkpoint_layers=True,
        # Let centers/features stabilize briefly, then enable local point-set
        # deformation early enough to train it in the 16-epoch schedule.
        shape_start_epoch=2,
        shape_ramp_epochs=4))

pc_range = [-40., -40., -1., 40., 40., 5.4]
grid_size = .4
data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
occ3d_path = 'data/occ3d/gts'
input_shape = (704, 256)
data_aug_conf = dict(
    resize_lim=(.38, .55), final_dim=input_shape[::-1],
    bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600,
    rand_flip=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=num_student_frames - 1,
         num_cams=num_cams, to_float32=False, pad_history=False),
    dict(type='LoadMultiViewImageFuture', num_future=num_future,
         num_cams=num_cams, to_float32=False),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='LoadSparseWorldFutureOccupancy', occ3d_path=occ3d_path),
    dict(type='PrepareOPUSTarget', max_gt_points=76800, empty_label=17),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True,
         num_cams=(num_student_frames + num_future) * num_cams),
]

# Validation never loads future RGB.  It retains future occupancy labels and
# ego transforms, which are evaluation targets rather than model inputs.
val_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=num_student_frames - 1,
         num_cams=num_cams, to_float32=False, pad_history=False),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='LoadSparseWorldFutureOccupancy', occ3d_path=occ3d_path),
    dict(type='PrepareOPUSTarget', max_gt_points=76800, empty_label=17),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True,
         num_cams=num_student_frames * num_cams),
]

return_keys = [
    'sample_idx', 'scene_token', 'img', 'projection_mat', 'image_wh',
    'occ_label', 'occ_xyz', 'occ_cam_mask', 'occ_lidar_mask',
    'occ_nonempty_mask', 'occ_loss_mask', 'future_occ_labels',
    'future_occ_cam_masks', 'future_occ_lidar_masks',
    'future_ego_to_current', 'temporal_trajs', 'temporal_ego_states',
]
dataset_common = dict(
    type='NuScenesSparseWorldTrajectoryDataset', data_root=data_root,
    data_aug_conf=data_aug_conf, pc_range=pc_range, occ3d=True,
    occ3d_coord='ego', admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
    future_steps=num_future, history_frames=num_student_frames - 1,
    history_keyframes_only=True, official_boundary_filter=True,
    return_keys=return_keys)
train_dataset_config = dict(
    **dataset_common,
    imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
    pipeline=train_pipeline, phase='train',
    trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl')
val_dataset_config = dict(
    **dataset_common,
    imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
    pipeline=val_pipeline, phase='val',
    trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl')

# This is the tested memory-bounded contract for one 24 GiB device.
train_loader = dict(
    batch_size=2, num_workers=4, shuffle=True, persistent_workers=True,
    prefetch_factor=2, pin_memory=True)
val_loader = dict(
    batch_size=1, num_workers=4, shuffle=False, persistent_workers=True,
    prefetch_factor=2, pin_memory=True)
gradient_accumulation = 2
max_cuda_memory_gb = 23.5

loss = dict(
    _delete_=True,
    type='SparseWorldFFALoss',
    target_horizon_index=target_horizon_index,
    occupancy_weight=1.0,
    # Occupancy remains the primary task. Teacher objectives are auxiliary
    # and must not dominate early optimization.
    geometry_weight=0.5,
    feature_weight=0.25,
    semantic_weight=0.25,
    residual_weight=1e-3,
    geometry_warmup_iters=10000,
    matcher=dict(position_weight=2.0, semantic_weight=1.0,
                 feature_weight=0.0, max_cost=20.0),
    future_occupancy_loss=dict(
        stage_weights=[1.0], loss_mode='official_v1', lambda_cls=2.0,
        focal_gamma=2.0, focal_alpha=0.25, smooth_l1_beta=0.2,
        lambda_pts=0.5, empty_dist_thr=0.2, empty_weight=5.0,
        rare_classes=[0, 2, 5, 8], rare_weight=10.0,
        class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10,
                       5, 5, 1, 5, 1, 1, 2, 1],
        pc_range=pc_range, chunk_size=1024,
        use_camera_mask_weights=False))

optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=.01),
    paramwise_cfg=dict(custom_keys={
        'semantic': dict(lr_mult=.1),
        'ffa.student_projection': dict(weight_decay_mult=0.0)}))

amp = True
amp_loss_scale = 128.0
amp_growth_interval = 2000
# FP16 loss scale can occasionally overflow on a difficult sample after it has
# grown. Skip that optimizer update and let GradScaler lower the scale. A long
# consecutive streak still indicates a real numerical problem and aborts.
recover_amp_overflow = True
max_consecutive_amp_overflows = 8
fail_on_nonfinite_grad = True
grad_max_norm = 5.0
max_epochs = 16
eval_every_epochs = 1
warmup_iters = 500
min_lr_ratio = .05
epoch_start_sleep = 0
print_freq = 20
occ3d_eval_mask = 'none'

# Generic train.py must not accidentally initialize this composite model.
load_from = None
