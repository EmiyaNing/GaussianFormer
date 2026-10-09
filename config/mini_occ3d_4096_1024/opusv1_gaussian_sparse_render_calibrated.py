"""Inference-only empty-class calibration selected on 256 training samples.

Use the epoch-5 checkpoint from opusv1_gaussian_sparse_render.py.  This
configuration changes only the empty-vs-occupied decision after GaussianFormer
local_aggregate; the Gaussian renderer and learned weights stay identical.
"""
_base_ = ['./opusv1_gaussian_sparse_render.py']

model = dict(head=dict(empty_logit_offset=4.0))
