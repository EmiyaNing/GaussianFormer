"""P1/A2: OPUS-v1 fixed KNN-8 mean residual control."""
_base_ = ['./p1_a1_pointwise.py']

model = dict(head=dict(gaussian_aggregator=dict(type='KNNResidualAggregator',
                                                 mode='mean')))
