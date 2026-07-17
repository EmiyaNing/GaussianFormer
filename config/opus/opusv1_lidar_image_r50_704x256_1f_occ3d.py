_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

# Keep LiDAR samples ragged through collation. MultiModalOPUSLifter voxelizes
# each sample independently before constructing a padded query batch.
occ3d_return_keys = [
    'img', 'projection_mat', 'image_wh', 'lidar_points', 'occ_label', 'occ_xyz',
    'occ_cam_mask', 'occ_lidar_mask', 'occ_nonempty_mask', 'occ_loss_mask',
]

train_dataset_config = dict(return_keys=occ3d_return_keys)
val_dataset_config = dict(return_keys=occ3d_return_keys)

model = dict(
    lifter=dict(
        type='MultiModalOPUSLifter', embed_dims=256,
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        voxel_size=0.4, query_stride=16, query_cap=4096,
        max_num_points_per_voxel=5, max_num_voxels=1600000,
        num_fallback_queries=32,
    ),
    encoder=dict(
        type='OPUSEncoder', embed_dims=256, num_decoder=6, num_heads=8,
        feedforward_channels=512, dropout=0.1, point_step=0.08,
        lidar_ball_radius=2.4, lidar_ball_k=16,
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
    ),
)

loss_input_convertion = dict(
    opus_pred_points='opus_pred_points',
    opus_pred_logits='opus_pred_logits',
    opus_point_valid_masks='opus_point_valid_masks',
)
