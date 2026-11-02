"""Ten image times (current + nine past), ten LiDAR scans; CUDA memory mode.

Rebuild temporal_ops after extending its frame-pointer capacity. For this
9-point/4-group/4-level model the 48-KiB row limit permits up to 11 frames.
The older four-frame history-bias vectors require explicit warm-start mapping.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda.py']

num_image_frames = 10
model = dict(num_image_frames=num_image_frames, image_cache_size=12,
             encoder=dict(deformable_model=dict(
                 num_image_frames=num_image_frames, num_cams=6 * num_image_frames)))

train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline in (train_pipeline, test_pipeline):
    for transform in pipeline:
        if transform['type'] == 'LoadReactImageHistory':
            transform['num_history'] = num_image_frames - 1
        elif transform['type'] == 'NuScenesAdaptor':
            transform['num_cams'] = 6 * num_image_frames

train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)
