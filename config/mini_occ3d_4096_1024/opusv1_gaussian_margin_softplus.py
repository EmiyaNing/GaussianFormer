"""Five-epoch ablation: direct geometry margin plus positive semantics."""
_base_ = ['./opusv1_gaussian_margin.py']

model = dict(
    encoder=dict(type='OPUSV1GaussianSoftplusEncoder'),
    head=dict(empty_logit_offset=0.0),
)
