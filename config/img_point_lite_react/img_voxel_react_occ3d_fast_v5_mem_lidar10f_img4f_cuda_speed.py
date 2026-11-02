"""Optional training speed tradeoff: keep FFN/image checkpoint, retain DAF graph.

This saves another DAF recomputation but uses more memory than img4f_cuda.
"""

_base_ = ['./img_voxel_react_occ3d_fast_v5_mem_lidar10f_img4f_cuda.py']
model = dict(encoder=dict(checkpoint_deformable=False))
