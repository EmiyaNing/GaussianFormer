"""Efficient final-point Semantic Gaussian residual on top of strict OPUS."""
_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

# Child configs are evaluated before base variables are merged.
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
grid_shape = [200, 200, 16]

model = dict(
    head=dict(
        _delete_=True,
        type='OPUSGaussianResidualHead',
        embed_dims=256, num_classes=17,
        point_multipliers=[1, 4, 16, 32, 64, 128],
        pc_range=pc_range, grid_size=grid_size, grid_shape=grid_shape,
        empty_label=17, score_threshold=0.5, center_distance_threshold=3.0,
        padding=True, decoder_outputs_logits=True,
        gaussian_scale_range=[[.1, .1, .1], [.6, .6, .6]],
        gaussian_initial_scale=.35, gaussian_initial_opacity=.02,
        gaussian_num_neighbors=8, gaussian_include_self=True,
        gaussian_query_chunk_size=256,
        ensemble_gamma=1.0,
        # Switch to 'opus' or 'gaussian_refined' for the two diagnostic modes.
        eval_mode='ensemble'),
)

loss = dict(
    _delete_=True,
    type='MultiLoss', loss_cfgs=[
        dict(type='OPUSSetLoss', stage_weights=[1., 1., 1., 1., 1., 1.],
             loss_mode='official_v1', lambda_cls=2.0, focal_gamma=2.0,
             smooth_l1_beta=.2, lambda_pts=.5, empty_dist_thr=.2,
             empty_weight=5.0, rare_classes=[0, 2, 5, 8], rare_weight=10.0,
             class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
             pc_range=pc_range, chunk_size=1024),
        dict(type='GaussianPointOccupancyLoss', weight=1.0, empty_label=17,
             focal_gamma=2.0, focal_alpha=.25, warmup_iters=5000,
             chunk_size=1024,
             class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1]),
    ])

loss_input_convertion = dict(
    opus_pred_points='opus_pred_points', opus_pred_logits='opus_pred_logits',
    gaussian_point_logits='gaussian_point_logits',
    gaussian_point_points='gaussian_point_points',
    gaussian_point_valid_mask='gaussian_point_valid_mask')
