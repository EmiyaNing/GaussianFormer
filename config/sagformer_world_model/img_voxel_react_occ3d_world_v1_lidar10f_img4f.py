"""Local world-model training with the locally trained four-frame perceiver.

The frozen perceiver is the img4f checkpoint at epoch 17 of
``out/img_voxel_react_img_history_aggregation``: four image frames (current
plus three past), ten LiDAR frames, 704x256, no camera mask. It was copied to
``ckpts/img_voxel_react_occ3d_fast_v5_lidar10f_img4f.pth`` with
sha256 ``f2cf67421965a5276e47c56a48b318cc6094ac0542ceae410ef86135cbe511d7``.

The numbers below are that run's full 6019-frame validation metrics
(``occ3d_eval_mask=none``), not a world-model result.
"""
_base_ = ['./img_voxel_react_occ3d_world_v1_img4f.py']

perception_checkpoint = 'ckpts/img_voxel_react_occ3d_fast_v5_lidar10f_img4f.pth'
model = dict(perception_checkpoint=perception_checkpoint, freeze_perception=True)

seed = 17
dataloader_seed = 17
print_freq = 20
cuda_memory_log_interval = 100
perception_baseline = dict(
    source='local full validation, epoch 17', camera_mask=False,
    mIoU=34.49, IoU=51.58, approximate=False, num_lidar_history=9,
    num_image_frames=4, image_width_height=(704, 256), epoch=17,
    checkpoint=perception_checkpoint,
    sha256='f2cf67421965a5276e47c56a48b318cc6094ac0542ceae410ef86135cbe511d7')
