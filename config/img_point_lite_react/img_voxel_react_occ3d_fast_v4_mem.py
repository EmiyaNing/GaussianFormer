"""Decoder memory round with DAF/FFN activation recomputation."""
_base_ = ['./img_voxel_react_occ3d_fast_v4.py']

model = dict(encoder=dict(checkpoint_ffn=True, checkpoint_deformable=True))
