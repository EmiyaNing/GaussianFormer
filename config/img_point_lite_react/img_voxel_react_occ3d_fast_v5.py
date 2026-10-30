"""Round five: image execution + worker BF16 transport and static grid.

History/queries/labels/losses are inherited unchanged. Four workers per rank
are measured separately from the image changes. Rebuild the DAF extension.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v5_image.py']
train_loader = dict(num_workers=4, persistent_workers=True, prefetch_factor=1)
val_loader = dict(num_workers=4, persistent_workers=True, prefetch_factor=1)
occ3d_return_keys = [key for key in _base_.occ3d_return_keys if key != 'occ_xyz']
occ3d_return_keys.append('static_occ_grid')
train_pipeline = _base_.train_pipeline
test_pipeline = _base_.test_pipeline
for pipeline in [train_pipeline, test_pipeline]:
    pipeline[1]['defer_xyz'] = True
    pipeline.append(dict(type='CastReactImage'))
train_dataset_config = dict(pipeline=train_pipeline, return_keys=occ3d_return_keys)
val_dataset_config = dict(pipeline=test_pipeline, return_keys=occ3d_return_keys)
