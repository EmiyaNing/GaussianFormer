"""Main voxel-FPS OPUS Fusion recipe: 8 image frames and 10 LiDAR frames.

Image history follows the existing OPUS temporal pipeline. Point clouds are
aligned into the current ego frame by NuScenesDataset before voxelization.
Only available past LiDAR frames are used at the start of each scene.
"""

_base_ = ['./opusv1_t_r50_704x256_8f_occ3d.py']

batch_size = 1
num_lidar_history = 9
occ3d_return_keys = [
    'img', 'projection_mat', 'image_wh', 'lidar_points', 'occ_label', 'occ_xyz',
    'occ_cam_mask', 'occ_lidar_mask', 'occ_nonempty_mask', 'occ_loss_mask',
]
train_dataset_config = dict(return_keys=occ3d_return_keys, num_lidar_history=num_lidar_history)
val_dataset_config = dict(return_keys=occ3d_return_keys, num_lidar_history=num_lidar_history)
train_loader = dict(batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=4)

model = dict(
    type='OPUSTemporalSegmentor', image_chunk_size=6, image_checkpoint=True,
    image_amp_dtype='bfloat16',
    lifter=dict(
        _delete_=True, type='FixedVoxelOPUSLifter', num_queries=600, embed_dims=256,
        pc_range=[-40., -40., -1., 40., 40., 5.4], voxel_size=0.4,
        max_num_points_per_voxel=5, max_num_voxels=1600000),
    encoder=dict(
        _delete_=True, type='OPUSVoxelFusionEncoder', embed_dims=256,
        num_decoder=6, num_frames=8, num_views=6, num_points=4, num_levels=4,
        num_groups=4, num_heads=8, feedforward_channels=512, dropout=0.1,
        num_classes=17, num_refines=[1, 4, 16, 32, 64, 128], scales=[0.5],
        pc_range=[-40., -40., -1., 40., 40., 5.4], min_sampling_scale=0.4,
        voxel_neighborhood=1, voxel_kernel_sigma=1.0, voxel_sample_chunk_size=1024,
        decoder_checkpoint=True),
)

loss_input_convertion = dict(
    opus_pred_points='opus_pred_points', opus_pred_logits='opus_pred_logits',
    opus_point_valid_masks='opus_point_valid_masks')
amp = False
fail_on_nonfinite_grad = True
