"""Official OPUS-V1-T-equivalent temporal configuration.

This configuration intentionally keeps the repository's current dataset and
runner APIs, while matching the official V1-T model capacity, 8-frame camera
input, point schedule and optimisation policy.  It is the reproduction target;
the 1-frame config is only a debugging baseline and is not comparable to the
official 33.2 mIoU report.
"""

_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

num_frames = 8
batch_size = 8
input_shape = (704, 256)
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
occ3d_path = 'data/occ3d/gts'
max_gt_points = 76800
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)

data_aug_conf = dict(
    resize_lim=(0.38, 0.55), final_dim=input_shape[::-1],
    bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0), H=900, W=1600,
    rand_flip=True)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageHistory', num_history=num_frames - 1,
         num_cams=6, to_float32=True, pad_history=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='PrepareOPUSTarget', max_gt_points=max_gt_points, empty_label=17),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6 * num_frames),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageHistory', num_history=num_frames - 1,
         num_cams=6, to_float32=True, pad_history=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='PrepareOPUSTarget', max_gt_points=max_gt_points, empty_label=17),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6 * num_frames),
]

train_dataset_config = dict(data_aug_conf=data_aug_conf, pipeline=train_pipeline)
val_dataset_config = dict(data_aug_conf=data_aug_conf, pipeline=test_pipeline)
train_loader = dict(batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=4)

model = dict(
    img_backbone=dict(
        norm_cfg=dict(type='BN2d', requires_grad=True), norm_eval=True,
        style='pytorch'),
    encoder=dict(num_frames=num_frames),
)

amp = True
amp_loss_scale = 512.0
amp_growth_interval = 2147483647
min_lr_ratio = 1e-3
max_epochs = 100
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
