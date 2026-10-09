"""P3 control: dense G2V with the S2GO-style 4x4x4 block kernel.

Identical to ``p3_dense_g2v.py`` except that the dense renderer is the
block-shared implementation (``model/ops/block_g2v``) instead of the naive
``LocalAggregator``.  This is the baseline that P4-3 showed to be 8.14x faster
and 6.4x lighter at the operator level, and it exists to re-measure the
*training-level* memory comparison that previously used the naive operator.
"""
_base_ = ['./p3_dense_g2v.py']

model = dict(
    gaussian_horizons=[0],
    semantic_gaussian=dict(dense_op='block'))
