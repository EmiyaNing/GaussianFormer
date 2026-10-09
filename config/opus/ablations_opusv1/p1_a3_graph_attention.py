"""P1/A3: OPUS-v1 KNN graph-attention residual control."""
_base_ = ['./p1_a1_pointwise.py']

model = dict(head=dict(gaussian_aggregator=dict(type='KNNResidualAggregator',
                                                 mode='attention')))
