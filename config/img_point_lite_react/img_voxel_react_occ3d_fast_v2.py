"""Second execution round: worker voxelization, lean batches and fused DAF.

Same history, precision, model budget and objective as fast.py. Requires a DAF
extension rebuild; workers use the original CPU voxelization algorithm.
"""
_base_ = ['./img_voxel_react_occ3d_fast.py']

model = dict(encoder=dict(deformable_model=dict(fuse_point_sum=True)))
non_blocking_transfer = True

# Two workers per rank; prefetch one large multi-camera sample per worker.
train_loader = dict(num_workers=2, persistent_workers=True, prefetch_factor=1)
val_loader = dict(num_workers=2, persistent_workers=True, prefetch_factor=1)

occ3d_return_keys = [key for key in _base_.occ3d_return_keys
                    if key not in ('ori_img', 'lidar_points')]
occ3d_return_keys.append('react_voxels')
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
train_pipeline[0]['keep_original'] = False
test_pipeline[0]['keep_original'] = False
train_pipeline.append(dict(type='PrepareReactVoxels', pc_range=_base_.pc_range,
                           voxel_size=_base_.grid_size))
test_pipeline.append(dict(type='PrepareReactVoxels', pc_range=_base_.pc_range,
                          voxel_size=_base_.grid_size))
train_dataset_config = dict(pipeline=train_pipeline, return_keys=occ3d_return_keys)
val_dataset_config = dict(pipeline=test_pipeline, return_keys=occ3d_return_keys)
