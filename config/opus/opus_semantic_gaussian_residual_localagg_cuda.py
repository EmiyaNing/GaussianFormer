"""Phase-B fused CUDA local aggregation; Phase-A config remains unchanged."""
_base_ = ['./opus_semantic_gaussian_residual.py']

model = dict(
    head=dict(
        type='OPUSSemanticGaussianLocalAggCUDAHead',
        # With scale_max=.6 and support_sigma=3, a 1.8m cell means each
        # query scans its own and the 26 adjacent cells in the common case.
        gaussian_cell_size=1.8,
        gaussian_support_sigma=3.0,
        gaussian_scale_eps=1e-4,
        gaussian_denom_eps=1e-6,
        gaussian_include_self=True,
        eval_mode='ensemble',
    ))
