"""Cloud-checkpoint perception: 704x256 images and nine historical LiDAR frames."""
_base_ = ['./img_voxel_react_occ3d_fast_v5.py']

# input_shape is (width, height); the augmentation expects (height, width).
# Match the validation scale: 704 / 1600 = .44. Keeping resize_lim=(1,1)
# would train on narrow crops of the original image, despite the same final size.
input_shape = (704, 256)
data_aug_conf = dict(final_dim=input_shape[::-1], resize_lim=(.44, .44))
train_dataset_config = dict(num_lidar_history=9, data_aug_conf=data_aug_conf)
val_dataset_config = dict(num_lidar_history=9, data_aug_conf=data_aug_conf)
