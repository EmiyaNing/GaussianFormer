"""Occ3D GaussianFormer-React with five LiDAR frames at 256 x 704.

Two deliberate differences from ``img_voxel_react_occ3d.py``:

1. LiDAR history.  ``num_lidar_history = 4`` makes ``NuScenesDataset`` fuse
   the current frame's ``LIDAR_TOP`` scan with the four previous ones (five
   LiDAR frames in total, roughly 2 s at nuScenes' 2 Hz keyframe rate; scene
   starts simply provide fewer frames).  In the ``*_sweeps_occ.pkl``
   annotations only keyframes carry ``LIDAR_TOP``, so the collected history is
   the previous keyframe scans rather than the 10 Hz LiDAR sweeps.  Each scan
   is transformed into the current LiDAR frame before concatenation, so
   ``GaussianVoxelLearnear`` initialises Gaussians from the fused cloud.  The
   camera branch is untouched and still consumes only the current frame's six
   views; nothing about the image pipeline changes for history.
2. 256 x 704 image input.  NuScenes frames are 900 x 1600, so
   ``resize_lim = (0.44, 0.44)`` resizes them to 704 x 396 and the pipeline
   crops the bottom 256 rows.  ``H``/``W`` stay at the real source resolution
   (900/1600) so ``ResizeCropFlipImage`` builds the correct homography;
   overriding ``H``/``W`` with 704/256 instead would resize both axes by
   different factors while the calibration is only scaled by ``resize``.
"""

_base_ = ['./img_voxel_react_occ3d.py']

# Five LiDAR frames: the current keyframe scan plus four previous ones.
num_lidar_history = 4

# Image input is 256 (H) x 704 (W); ``input_shape`` is (W, H) like the
# existing configs, and ``final_dim`` is derived as (H, W).
input_shape = (704, 256)
data_aug_conf = {
    # 0.44 * 1600 = 704 exactly; 0.44 * 900 = 396 -> bottom crop of 256 rows.
    # The same resize/crop is used in train and val, matching the official
    # 704x256 Occ3D recipe.
    "resize_lim": (0.44, 0.44),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_dataset_config = dict(
    data_aug_conf=data_aug_conf,
    num_lidar_history=num_lidar_history)

val_dataset_config = dict(
    data_aug_conf=data_aug_conf,
    num_lidar_history=num_lidar_history)

# Five LiDAR scans per sample make loading noticeably heavier than the
# single-scan recipe, which inherited ``num_workers=0`` from
# ``_base_/surroundocc.py``.
train_loader = dict(num_workers=4)
