"""Two training samples and one validation sample; execution checks only."""
_base_ = ['./img_voxel_react_occ3d_world_v1.py']

max_epochs = 1
# Use samples away from scene starts so all nine historical LiDAR frames exist.
train_dataset_config = dict(vis_indices=[136, 264])
val_dataset_config = dict(vis_indices=[136])
train_loader = dict(num_workers=0)
val_loader = dict(num_workers=0)
print_freq = 1
cuda_memory_log_interval = 1
