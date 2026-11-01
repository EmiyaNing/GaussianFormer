"""v5_mem perception with 704x256 images and nine historical LiDAR frames.

History is aligned and fused before the existing voxelization/perception path.
Scene starts use the available history, following the original dataset policy.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v5_mem.py']

input_shape = (704, 256)
# Resize before cropping in training, just as in validation. A full-size crop
# would change both the apparent object scale and the camera field of view.
data_aug_conf = dict(final_dim=input_shape[::-1], resize_lim=(.44, .44))
train_dataset_config = dict(num_lidar_history=9, data_aug_conf=data_aug_conf)
val_dataset_config = dict(num_lidar_history=9, data_aug_conf=data_aug_conf)
