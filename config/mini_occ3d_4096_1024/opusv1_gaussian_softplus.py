"""Five-epoch ablation: softplus OPUS logits only for Gaussian rendering."""
_base_ = ['./opusv1_gaussian_sparse_render.py']

model = dict(
    encoder=dict(type='OPUSV1GaussianSoftplusEncoder'),
    head=dict(empty_logit_offset=0.0),
)
