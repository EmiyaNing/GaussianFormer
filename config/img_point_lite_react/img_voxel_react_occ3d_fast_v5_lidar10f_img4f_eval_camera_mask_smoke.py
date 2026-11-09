"""Two-sample smoke check for the official Occ3D eval config."""

_base_ = ['./img_voxel_react_occ3d_fast_v5_lidar10f_img4f_eval_camera_mask.py']

# Use samples away from scene starts so the history window is complete.
val_dataset_config = dict(vis_indices=[136, 264])
val_loader = dict(num_workers=0)
print_freq = 1
