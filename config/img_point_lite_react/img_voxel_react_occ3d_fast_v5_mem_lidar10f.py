"""Match the four-GPU 704x256 recipe with nine historical LiDAR frames."""

_base_ = ['./img_voxel_react_occ3d_fast_v5_mem.py']

# These checkpoints were trained at 704x256. Image resolution/cropping is not
# recorded in state_dict and must match the saved training configuration.
input_shape = (704, 256)
data_aug_conf = dict(
    H=900, W=1600, resize_lim=(0.44, 0.44), final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0), rot_lim=(0.0, 0.0), rand_flip=True,
)
# Match the fusion path recorded in the four-GPU run; recomputation inherited
# from v5_mem is active only during training.
model = dict(encoder=dict(deformable_model=dict(fuse_normalized_sampling=False)))
train_dataset_config = dict(num_lidar_history=9, data_aug_conf=data_aug_conf)
val_dataset_config = dict(num_lidar_history=9, data_aug_conf=data_aug_conf)
occ3d_eval_mask = 'none'
