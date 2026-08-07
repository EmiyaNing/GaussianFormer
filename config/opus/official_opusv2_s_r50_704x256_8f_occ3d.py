"""Strict OPUSv2-S inferred from the released official S checkpoint."""
_base_ = ['./official_opusv2_t_r50_704x256_8f_occ3d.py']

model = dict(head=dict(
    num_query=1200,
    transformer=dict(num_points=2, num_refines=[4, 8, 16, 32, 64])))

