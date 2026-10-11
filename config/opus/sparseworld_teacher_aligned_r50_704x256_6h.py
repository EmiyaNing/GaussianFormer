"""Six-horizon frozen-OPUS distillation on one 24 GiB GPU."""

_base_ = ['./sparseworld_strict_r50_704x256_5f_occ3d.py']

alignment_contract_version = 2
num_cams = 6
student_frames = 5
teacher_history_frames = 6
num_future = 6
step_seconds = .5

teacher_config = (
    'config/opus/'
    'opusv1_t_r50_704x256_8f_720q48p_nomask_occ3d_20e.py')
teacher_checkpoint = 'ckpts/opusv1_720query.pth'
teacher_checkpoint_sha256 = (
    '1d85e429f83792ba7b2a84d8a37446a0736124859bd55d79400d1ecf53132fce')
teacher_prefer_ema = True
teacher_amp = True

student_init_checkpoint = (
    'out/sparseworld_uncertainty/T00/official_epoch56_mapped.pth')
student_init_checkpoint_sha256 = (
    'fb9f955f15362cb23244fd551145e499b6a28b1a46e45849b129d9b60a879828')

model = dict(
    type='SparseWorldTeacherAlignedSegmentor',
    feature_mlp_dims=512,
    feature_mlp_dropout=.1,
    alignment_dims=256,
    freeze_perception=True,
    finetune_epoch=0)

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

# Training carries two extra history keyframes solely for the eight-frame
# teacher windows.  SparseWorld itself still consumes exactly t ... t-4.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory',
         num_history=teacher_history_frames, num_cams=num_cams,
         to_float32=False, pad_history=False),
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
         num_cams=(1 + teacher_history_frames + num_future) * num_cams),
]

val_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=student_frames - 1,
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
         num_cams=student_frames * num_cams),
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
    future_steps=num_future, history_keyframes_only=True,
    official_boundary_filter=True, return_keys=return_keys)
train_dataset_config = dict(
    **dataset_common, history_frames=teacher_history_frames,
    imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
    pipeline=train_pipeline, phase='train',
    trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl')
val_dataset_config = dict(
    **dataset_common, history_frames=student_frames - 1,
    imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
    pipeline=val_pipeline, phase='val',
    trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl')

# Batch 1 plus accumulation 2 is the conservative 24 GiB contract. Teacher
# horizons execute serially and keep only detached sparse query tensors.
train_loader = dict(
    batch_size=1, num_workers=4, shuffle=True, persistent_workers=True,
    prefetch_factor=2, pin_memory=True)
val_loader = dict(
    batch_size=1, num_workers=4, shuffle=False, persistent_workers=True,
    prefetch_factor=2, pin_memory=True)
gradient_accumulation = 2
max_cuda_memory_gb = 23.5

base_sparseworld_loss = dict(
    type='SparseWorldStrictLoss',
    current_loss=dict(
        type='OPUSSetLoss', stage_weights=[1.] * 6,
        loss_mode='official_v1', lambda_cls=2., focal_gamma=2.,
        smooth_l1_beta=.2, lambda_pts=.5, empty_dist_thr=.2,
        empty_weight=5., rare_classes=[0, 2, 5, 8], rare_weight=10.,
        class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10,
                       5, 5, 1, 5, 1, 1, 2, 1], pc_range=pc_range),
    pc_range=pc_range, grid_size=grid_size, lambda_future_cls=2.,
    lambda_future_pts=.5, lambda_traj=1., chunk_size=1024)
loss = dict(
    _delete_=True, type='SparseWorldTeacherAlignmentLoss',
    base_loss=base_sparseworld_loss,
    horizon_weights=[.5, .5, .75, .75, 1., 1.],
    geometry_weight=.25, feature_weight=.1, semantic_weight=.1,
    residual_weight=1e-3, alignment_warmup_iters=2000, future_only=True,
    matcher=dict(position_weight=2., semantic_weight=1.,
                 feature_weight=0., max_cost=20.))

# Existing future heads receive a smaller learning rate; the new residual MLP
# and alignment projection learn at the base rate.
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=.01),
    existing_future_lr=2e-5)
amp = True
amp_loss_scale = 64.
amp_growth_interval = 2000
recover_amp_overflow = True
max_consecutive_amp_overflows = 8
grad_max_norm = 5.
max_epochs = 16
eval_every_epochs = 1
warmup_iters = 500
min_lr_ratio = .05
print_freq = 20
occ3d_eval_mask = 'none'
sparseworld_future_eval = dict(
    enabled=True, horizons=[2, 4, 6], step_seconds=.5,
    report_occ_iou=True)
load_from = None
