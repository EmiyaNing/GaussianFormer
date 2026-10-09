"""P1/A1: point-wise MLP residual with NO neighbourhood aggregation.

This is the "extra capacity, not aggregation" control.  It inherits the A5
recipe (frozen backbone, warm start, reduced split, identical budget) and
replaces only the aggregation module with a pure per-point residual, so if A1
matches A2/A5 then the gain comes from adding a residual branch at all rather
than from any spatial aggregation.

    python3.10 train.py --py-config config/opus/ablations/p1_a1_pointwise.py \
      --work-dir out/p1_a1_pointwise --seed 42
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=8, include_self=True, mode="self",
        query_chunk_size=256))
