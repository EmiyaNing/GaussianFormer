"""Frozen four-frame React perception plus reg/cls/vel prediction and planning.

Same prediction/planning head, losses, datasets and readout protocol as
``img_voxel_react_occ3d_world_v1.py``. Only the perceiver changes: the current
camera frame plus three past frames, six cameras each, fused by
``TemporalDeformableFeatureAggregation`` on top of the ten-frame LiDAR branch.

Set ``REACT_PERCEPTION_CHECKPOINT`` to an img4f perception checkpoint before
fresh training, or use
``img_voxel_react_occ3d_world_v1_lidar10f_img4f.py``, which pins the locally
trained epoch-17 weights. A resumed world checkpoint already contains the
perception state.
"""
import os

_base_ = ['../img_point_lite_react/img_voxel_react_occ3d_fast_v5_lidar10f_img4f.py']

future_steps = 6
perception_checkpoint = os.environ.get('REACT_PERCEPTION_CHECKPOINT')
model = dict(
    type='ReactTemporalGaussianWorld', freeze_perception=True,
    perception_checkpoint=perception_checkpoint,
    world_decoder=dict(type='ReactWorldDecoder', embed_dims=128, ego_state_dim=21,
                       future_steps=future_steps, step_seconds=.5,
                       reg_limit=.2, displacement_limit=10., pc_range=_base_.pc_range))

# Keep every perception transform and setting from the four-frame preset.
# Only target loading and metadata selection are appended.
world_return_keys = list(_base_.occ3d_return_keys) + [
    'future_occ_labels', 'future_occ_cam_masks', 'future_occ_lidar_masks',
    'future_ego_to_current', 'temporal_ego_states', 'temporal_trajs',
    'ego2lidar', 'scene_token', 'sample_idx', 'timestamp']
future_labels = dict(type='LoadSparseWorldFutureOccupancy', occ3d_path='data/occ3d')
train_dataset_config = dict(
    type='NuScenesReactWorldDataset', history_frames=0, future_steps=future_steps,
    admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
    trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl',
    pipeline=list(_base_.train_pipeline) + [future_labels], return_keys=world_return_keys)
val_dataset_config = dict(
    type='NuScenesReactWorldDataset', history_frames=0, future_steps=future_steps,
    admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
    trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl',
    pipeline=list(_base_.test_pipeline) + [future_labels], return_keys=world_return_keys)

loss = dict(
    _delete_=True, type='ReactWorldLoss',
    occupancy_loss=_base_.loss.loss_cfgs[0],
    occupancy_weight=1., plan_weight=1., reg_weight=.01,
    horizon_weights=[1.] * future_steps)
loss_input_convertion = dict(
    _delete_=True, future_occ_logits='future_occ_logits',
    future_occ_frame='future_occ_frame', pred_traj='pred_traj',
    forecast_reg_offsets='forecast_reg_offsets')

optimizer = dict(_delete_=True, optimizer=dict(type='AdamW', lr=2e-4, weight_decay=.01))
onecyclelr = False
max_epochs = 24
grad_max_norm = 5
fail_on_nonfinite_grad = True
eval_every_epochs = 1
react_world_eval = True
occ3d_eval_mask = 'none'
# The perception has fixed BN statistics; new world modules use LayerNorm.
syncBN = False
load_from = None
train_load_from = None
