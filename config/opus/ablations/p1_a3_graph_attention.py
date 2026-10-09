"""P1/A3: k-NN graph attention - the "graph aggregation is already enough" control.

Same k-NN graph and the same residual mechanism as A2, but the message is a
scaled dot-product attention over neighbour features (the GraphGSOcc-style
alternative the plan asks to rule out) instead of a plain mean/max pooling.

Parameter count is 215,202 versus the Gaussian adapter's 180,501 (+19%), so
this control is biased *in its own favour*; a Gaussian win here is therefore a
conservative result, and the report states the mismatch explicitly.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=8, include_self=True, mode='attention',
        query_chunk_size=256))
