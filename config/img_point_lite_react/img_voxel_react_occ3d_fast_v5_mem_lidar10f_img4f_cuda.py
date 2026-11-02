"""Exact four-frame fusion with packed FPN/dropout and fused CUDA geometry.

No parameter or checkpoint-key changes. Build the independent temporal_ops
extension; existing DAF and renderer binaries do not need rebuilding.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f.py']
model = dict(pack_temporal_features=True,
             encoder=dict(deformable_model=dict(fuse_temporal_sampling=True)))
