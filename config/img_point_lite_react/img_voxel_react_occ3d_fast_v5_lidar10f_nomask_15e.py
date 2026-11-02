"""Fast v5 + nine LiDAR history frames, no camera mask in train or eval, 15 epochs.

Both mask decisions are made explicit because the repository defaults are easy
to misread:

* Training already supervises the full 200 x 200 x 16 grid: the Occ3D loader
  runs with ``train_mask_type='none'``, so ``occ_loss_mask`` is all True. It is
  pinned to ``'none'`` below so a future parent change cannot silently switch
  the loss to camera-visible voxels (``'camera'`` would shrink it to ~24.5 %).
* Evaluation would normally score only camera-visible voxels because
  ``img_voxel_react_occ3d.py`` sets ``occ3d_eval_mask='camera'``. Setting it to
  ``'none'`` makes both the Occ3D ``Metric_mIoU`` in eval.py and the
  in-training ``MeanIoU`` count every voxel in the grid.

Everything else is inherited from ``img_voxel_react_occ3d_fast_v5_lidar10f.py``
(256 x 704 images, tiled React renderer, BF16 image path, four workers, ten
LiDAR frames, losses). Only 15 epochs are trained, and the OneCycle schedule is
built from ``max_epochs`` so it spans those 15 epochs instead of 20.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5_lidar10f.py']

# ---- training: full-grid supervision, no camera mask ----
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline in (train_pipeline, test_pipeline):
    for transform in pipeline:
        if transform['type'] == 'LoadOccupancyOcc3D':
            transform['train_mask_type'] = 'none'
train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)

# ---- evaluation: full-grid mIoU, no camera mask ----
# 'camera' would filter to mask_camera; 'none' scores the whole grid. Leaving
# the key unset is NOT equivalent, because it falls back to eval_mask_flag=True.
occ3d_eval_mask = 'none'

# ---- schedule ----
max_epochs = 15
