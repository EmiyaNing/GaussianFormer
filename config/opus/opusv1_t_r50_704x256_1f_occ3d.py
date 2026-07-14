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
batch_size = 1
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
    'opus_gt_points', 'opus_gt_labels', 'opus_gt_valid',
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='PrepareOPUSTarget', max_gt_points=max_gt_points, empty_label=17),
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
    dict(type='PrepareOPUSTarget', max_gt_points=max_gt_points, empty_label=17),
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
train_loader = dict(batch_size=batch_size, num_workers=2, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=2)

model = dict(
    type='OPUSSegmentor', img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(type='ResNet', depth=50, num_stages=4,
        out_indices=(0, 1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False), norm_eval=True,
        style='caffe', with_cp=True),
    img_neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
        out_channels=128, start_level=0, num_outs=4,
        add_extra_convs='on_output', relu_before_extra_convs=True),
    lifter=dict(type='OPUSQueryLifter', num_queries=600, embed_dims=128),
    encoder=dict(type='OPUSEncoder', embed_dims=128, num_decoder=6,
        num_heads=8, feedforward_channels=512, dropout=0.1, point_step=0.08),
    head=dict(type='OPUSHead', embed_dims=128, num_classes=17,
        point_multipliers=[1, 4, 16, 32, 64, 128], pc_range=pc_range,
        grid_size=grid_size, grid_shape=grid_shape, empty_label=17,
        score_threshold=0.0),
)

loss = dict(type='MultiLoss', loss_cfgs=[
    dict(type='OPUSSetLoss', stage_weights=[0.25, 0.35, 0.5, 0.7, 0.85, 1.0],
         lambda_cd=5.0, lambda_cls=1.0, focal_gamma=2.0,
         pc_range=pc_range, chunk_size=1024, max_match_points=8192),
])
loss_input_convertion = dict(
    opus_pred_points='opus_pred_points', opus_pred_logits='opus_pred_logits')

optimizer = dict(
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}))
# The sparse set loss and six-stage decoder are numerically sensitive during
# cold start. Keep the reference baseline in FP32; enable AMP only after a
# stable checkpoint and a separately tuned GradScaler configuration exist.
amp = False
fail_on_nonfinite_grad = True
grad_max_norm = 35
max_epochs = 100
eval_every_epochs = 1
warmup_iters = 500
load_from = 'ckpts/raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth'
