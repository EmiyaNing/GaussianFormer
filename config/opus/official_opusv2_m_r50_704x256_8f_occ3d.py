"""Strict OPUSv2-M inferred from the released official M checkpoint."""
_base_ = ['./official_opusv2_t_r50_704x256_8f_occ3d.py']

model = dict(head=dict(
    num_query=2400,
    transformer=dict(num_points=2, num_refines=[2, 4, 8, 16, 32])))

