"""Strict OPUSv2-L matching the official L config and checkpoint."""
_base_ = ['./official_opusv2_t_r50_704x256_8f_occ3d.py']

model = dict(head=dict(
    num_query=4800,
    transformer=dict(num_points=2, num_refines=[1, 2, 4, 8, 16])))
