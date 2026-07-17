_base_ = ['../_base_/misc.py']

dataset_name_flag = 'occ3d'
occ3d_eval_mask = 'camera'
data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
occ3d_path = 'data/occ3d/gts'
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
grid_shape = [200, 200, 16]
input_shape = (704, 256)
batch_size = 4
max_gt_points = 76800

data_aug_conf = dict(
    resize_lim=(0.44, 0.44), final_dim=input_shape[::-1],
    bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0), H=900, W=1600,
    rand_flip=True)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
occ3d_return_keys = [
    'img', 'projection_mat', 'image_wh', 'occ_label', 'occ_xyz',
    'occ_cam_mask', 'occ_lidar_mask', 'occ_nonempty_mask', 'occ_loss_mask',
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]
train_dataset_config = dict(
    type='NuScenesDataset', data_root=data_root,
    imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
    data_aug_conf=data_aug_conf, pipeline=train_pipeline, pc_range=pc_range,
    occ3d=True, occ3d_coord='ego', phase='train', return_keys=occ3d_return_keys)
val_dataset_config = dict(
    type='NuScenesDataset', data_root=data_root,
    imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
    data_aug_conf=data_aug_conf, pipeline=test_pipeline, pc_range=pc_range,
    occ3d=True, occ3d_coord='ego', phase='val', return_keys=occ3d_return_keys)
train_loader = dict(batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=4)

model = dict(
    type='OPUSSegmentor', img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(type='ResNet', depth=50, num_stages=4,
        out_indices=(0, 1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True), norm_eval=True,
        style='pytorch', with_cp=True),
    img_neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
        out_channels=256, start_level=0, num_outs=4),
    lifter=dict(type='OPUSQueryLifter', num_queries=600, embed_dims=256,
        learnable_features=False, reference_mode='direct'),
    encoder=dict(type='StrictOPUSV1Encoder', embed_dims=256, num_decoder=6,
        num_frames=1, num_views=6, num_points=4, num_levels=4, num_groups=4,
        num_heads=8, feedforward_channels=512, dropout=0.1, num_classes=17,
        num_refines=[1, 4, 16, 32, 64, 128], scales=[0.5], pc_range=pc_range),
    head=dict(type='OPUSHead', embed_dims=256, num_classes=17,
        point_multipliers=[1, 4, 16, 32, 64, 128], pc_range=pc_range,
        grid_size=grid_size, grid_shape=grid_shape, empty_label=17,
        score_threshold=0.5, center_distance_threshold=3.0, padding=True,
        decoder_outputs_logits=True),
)

loss = dict(type='MultiLoss', loss_cfgs=[
    dict(type='OPUSSetLoss', stage_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
         loss_mode='official_v1', lambda_cls=2.0, focal_gamma=2.0,
         smooth_l1_beta=0.2, lambda_pts=0.5, empty_dist_thr=0.2,
         empty_weight=5.0, rare_classes=[0, 2, 5, 8], rare_weight=10.0,
         class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
         pc_range=pc_range, chunk_size=1024),
])
loss_input_convertion = dict(
    opus_pred_points='opus_pred_points', opus_pred_logits='opus_pred_logits')

optimizer = dict(
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1), 'sampling_offset': dict(lr_mult=0.1)}))
# The 1-frame configuration is a memory/debug baseline, not the official
# 8-frame FP16 recipe.  Keep it in FP32: the MSMV extension's backward is
# FP32 and global AMP can overflow its first backbone update on this path.
amp = False
fail_on_nonfinite_grad = True
grad_max_norm = 35
max_epochs = 100
eval_every_epochs = 1
warmup_iters = 500
load_from = 'ckpts/raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth'
