"""Decoder memory round: same FP32 model/budget, compact projection/weights.

Rebuild DAF for the fused masked softmax. Checkpointing stays optional; the
fast_v4_mem.py derivative enables DAF and FFN recomputation for comparison.
"""
_base_ = ['./img_voxel_react_occ3d_fast_v3.py']

model = dict(encoder=dict(deformable_model=dict(
    batched_projection=True, factor_camera_weights=True,
    compact_weight_mask=True, fused_weight_softmax=True)))
