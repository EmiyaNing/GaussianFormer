"""Strict official OPUSv2-T future-15-frame Occ3D configuration."""
_base_ = ['./official_opusv2_t_r50_704x256_8f_occ3d.py']

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ3d_path = 'data/occ3d/gts'
grid_size = 0.4
max_gt_points = 76800
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=7, num_future=7,
         num_cams=6, to_float32=False, pad_history=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=90),
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadMultiViewImageHistory', num_history=7, num_future=7,
         num_cams=6, to_float32=False, pad_history=True),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=grid_size, model_coord='ego',
         train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=90),
]

train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)
model = dict(head=dict(transformer=dict(num_frames=15)))
