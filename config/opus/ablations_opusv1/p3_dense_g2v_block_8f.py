"""P3 OPUS-v1-8f dense block-G2V control; only the aggregation operator differs."""
_base_ = ['./p3_ours_8f.py']

pc_range = [-40., -40., -1., 40., 40., 5.4]
model = dict(head=dict(
    _delete_=True, type='OPUSGaussianResidualHead', embed_dims=256,
    num_classes=17, point_multipliers=[1, 4, 16, 32, 64, 128],
    pc_range=pc_range, grid_size=.4, grid_shape=[200, 200, 16],
    empty_label=17, score_threshold=.5, center_distance_threshold=3.,
    padding=True, decoder_outputs_logits=True, eval_mode='ensemble',
    gaussian_scale_range=[[.1, .1, .1], [.6, .6, .6]],
    gaussian_initial_scale=.35, gaussian_initial_opacity=.02,
    gaussian_num_neighbors=8, gaussian_include_self=True,
    gaussian_query_chunk_size=256,
    gaussian_aggregator=dict(type='DenseG2VResidualAggregator', dense_op='block',
        grid_shape=[200, 200, 16], grid_size=.4, scale_multiplier=3.,
        render_channels=18)))
