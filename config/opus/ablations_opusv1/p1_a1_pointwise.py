"""P1/A1: OPUS-v1 point-wise residual control; no neighbour aggregation."""
_base_ = ['./p1_adapter_base.py']

pc_range = [-40., -40., -1., 40., 40., 5.4]
model = dict(head=dict(
    _delete_=True, type='OPUSGaussianResidualHead', embed_dims=256,
    num_classes=17, point_multipliers=[1, 4, 16, 32, 64, 128],
    pc_range=pc_range, grid_size=.4, grid_shape=[200, 200, 16],
    empty_label=17, score_threshold=.5, center_distance_threshold=3.,
    padding=True, decoder_outputs_logits=True, eval_mode='ensemble',
    gaussian_aggregator=dict(type='KNNResidualAggregator', mode='self')))
