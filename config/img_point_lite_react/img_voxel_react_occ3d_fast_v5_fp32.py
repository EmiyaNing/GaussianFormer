"""FP32 alternative: fixed BN, data workers/static grid and fused DAF.

Keep image input/backbone FP32 to isolate speed changes from BF16 rounding.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v5_fused.py']
model = dict(image_backbone_bf16=False)
train_pipeline = [step for step in _base_.train_pipeline if step['type'] != 'CastReactImage']
test_pipeline = [step for step in _base_.test_pipeline if step['type'] != 'CastReactImage']
train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)
