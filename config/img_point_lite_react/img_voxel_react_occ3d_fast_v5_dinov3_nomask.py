"""Fast v5 + DINOv3 ConvNeXt-Small backbone with no camera mask anywhere.

Derived from ``img_voxel_react_occ3d_fast_v5_dinov3.py``; only the two mask
switches change:

* training supervises the full 200 x 200 x 16 grid (``train_mask_type='none'``,
  so ``occ_loss_mask`` is all True) instead of ``mask_camera``;
* evaluation scores the full grid (``occ3d_eval_mask='none'``) instead of
  camera-visible voxels.

Backbone weights, 256 x 704 input, tiled React renderer, BF16 image path,
four workers, the ten LiDAR frames (``num_lidar_history=9`` inherited from the
parent dataset configs) and the loss are inherited unchanged.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5_dinov3.py']

train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline in (train_pipeline, test_pipeline):
    for transform in pipeline:
        if transform['type'] == 'LoadOccupancyOcc3D':
            transform['train_mask_type'] = 'none'
train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)

occ3d_eval_mask = 'none'
