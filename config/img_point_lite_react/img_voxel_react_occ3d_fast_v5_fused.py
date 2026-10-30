"""Fourth optimization step: no global DAF probabilities; recompute in backward."""
_base_ = ['./img_voxel_react_occ3d_fast_v5.py']
model = dict(encoder=dict(deformable_model=dict(fuse_normalized_sampling=True)))
