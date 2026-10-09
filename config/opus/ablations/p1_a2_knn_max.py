"""P1/A2-max: KNN-8 max residual - the second neighbourhood-smoothing control.

Same as ``p1_a2_knn_mean.py`` but aggregates with a per-neighbour maximum
instead of a mean, which is the other cheap non-parametric alternative the
plan asks to rule out.
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=8, include_self=True, mode='max',
        query_chunk_size=256))
