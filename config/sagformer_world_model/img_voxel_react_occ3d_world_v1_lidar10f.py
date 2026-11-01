"""Local world-model training with the cloud-trained ten-frame LiDAR prior.

The frozen checkpoint uses current LiDAR plus nine historical frames. Cloud
metrics below are user-reported full-space results, not a local reproduction.
"""
_base_ = ['./img_voxel_react_occ3d_world_v1.py']

perception_checkpoint = 'ckpts/img_voxel_react_occ3d_fastv5_lidar10f.pth'
model = dict(perception_checkpoint=perception_checkpoint, freeze_perception=True)

seed = 17
dataloader_seed = 17
print_freq = 20
cuda_memory_log_interval = 100
cloud_perception_baseline = dict(
    source='user_reported_cloud_evaluation', camera_mask=False,
    mIoU=33.0, IoU=51.0, approximate=True, num_lidar_history=9,
    image_width_height=(704, 256),
    checkpoint=perception_checkpoint)
