"""Lower-memory variant: full v5 plus decoder recomputation."""
_base_ = ['./img_voxel_react_occ3d_fast_v5_fused.py']
model = dict(encoder=dict(checkpoint_ffn=True, checkpoint_deformable=True))
