"""P1/A2: KNN-8 mean residual - the "plain neighbourhood smoothing" control.

Inherits the A5 recipe (frozen backbone, warm start, reduced split, identical
budget) and replaces only the aggregation module, so any difference between
A2 and A5 is attributable to the Gaussian parameterisation rather than to
data, schedule or warm start.

    python3.10 train.py --py-config config/opus/ablations/p1_a2_knn_mean.py \
      --work-dir out/p1_a2_knn_mean --seed 42
"""
_base_ = ['./p1_a5_gaussian.py']

model = dict(
    semantic_gaussian=dict(
        _delete_=True,
        type='KNNResidualAggregator',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        num_neighbors=8, include_self=True, mode='mean',
        query_chunk_size=256))
