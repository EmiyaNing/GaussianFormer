"""704x256 React: current + 3 past images, current + 9 past LiDAR scans.

History images use past sensor poses in the current ego coordinate system.
Only current image features retain a backbone/FPN gradient by default.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5_mem_lidar10f.py']

num_image_frames = 4
num_lidar_history = 9
occ3d_eval_mask = 'none'

model = dict(
    type='ReactTemporalSegmentor', num_image_frames=num_image_frames,
    num_cameras=6, detach_history=True, image_checkpoint=True, image_cache_size=8,
    encoder=dict(
        checkpoint_ffn=True, checkpoint_deformable=True,
        deformable_model=dict(
            type='TemporalDeformableFeatureAggregation',
            num_image_frames=num_image_frames, num_cameras=6, num_cams=6 * num_image_frames,
            initial_history_logit_bias=-2.0, fuse_normalized_sampling=True)))

# Warm-start all existing weights; only the two small temporal bias vectors
# are new. Override load_from to train from a different initialization.
load_from = 'ckpts/img_voxel_react_occ3d_fast_v5_mem.pth'
optimizer = dict(optimizer=dict(lr=2e-5))
onecyclelr = False
warmup_iters = 200
warmup_lr_init = 2e-6

train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline, test_mode in [(train_pipeline, False), (test_pipeline, True)]:
    pipeline.insert(1, dict(type='LoadReactImageHistory', num_history=num_image_frames - 1,
                           num_cams=6, to_float32=True, test_mode=test_mode, interval=(4, 8)))
    for transform in pipeline:
        if transform['type'] == 'NuScenesAdaptor':
            transform['num_cams'] = 6 * num_image_frames

occ3d_return_keys = list(_base_.occ3d_return_keys)
occ3d_return_keys += ['image_frame_indices', 'image_time_offsets', 'image_frame_valid',
                      'image_frame_keys', 'image_aug_key', 'scene_token']
train_dataset_config = dict(pipeline=train_pipeline, return_keys=occ3d_return_keys,
                            num_lidar_history=num_lidar_history)
val_dataset_config = dict(pipeline=test_pipeline, return_keys=occ3d_return_keys,
                          num_lidar_history=num_lidar_history)
