"""Two training samples and one validation sample; execution checks only."""
_base_ = ['./react_occ3d_world_perception_finetune_img4f.py']

max_epochs = 1
# Use samples away from scene starts so all history frames exist.
train_dataset_config = dict(vis_indices=[136, 264])
val_dataset_config = dict(vis_indices=[136])
train_loader = dict(num_workers=0)
val_loader = dict(num_workers=0)
print_freq = 1
cuda_memory_log_interval = 1
