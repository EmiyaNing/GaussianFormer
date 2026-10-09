"""P3 OPUS-v1-8f end-to-end sparse Gaussian system experiment."""
_base_ = ['../opusv1_t_r50_704x256_8f_occ3d.py']

pc_range = [-40., -40., -1., 40., 40., 5.4]
model = dict(head=dict(
    _delete_=True, type='OPUSSemanticGaussianLocalAggCUDAHead',
    embed_dims=256, num_classes=17, point_multipliers=[1, 4, 16, 32, 64, 128],
    pc_range=pc_range, grid_size=.4, grid_shape=[200, 200, 16],
    empty_label=17, score_threshold=.5, center_distance_threshold=3.,
    padding=True, decoder_outputs_logits=True, eval_mode='ensemble',
    gaussian_scale_range=[[.1, .1, .1], [.6, .6, .6]],
    gaussian_initial_scale=.35, gaussian_initial_opacity=.02,
    gaussian_num_neighbors=8, gaussian_include_self=True,
    gaussian_query_chunk_size=256, gaussian_cell_size=1.8,
    gaussian_support_sigma=3., gaussian_scale_eps=1e-4,
    gaussian_denom_eps=1e-6, gaussian_max_neighbours=32))

loss = dict(_delete_=True, type='MultiLoss', loss_cfgs=[
    dict(type='OPUSSetLoss', stage_weights=[1., 1., 1., 1., 1., 1.],
         loss_mode='official_v1', lambda_cls=2., focal_gamma=2.,
         smooth_l1_beta=.2, lambda_pts=.5, empty_dist_thr=.2, empty_weight=5.,
         rare_classes=[0, 2, 5, 8], rare_weight=10.,
         class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
         pc_range=pc_range, chunk_size=1024),
    dict(type='GaussianPointOccupancyLoss', weight=1., empty_label=17,
         focal_gamma=2., focal_alpha=.25, warmup_iters=5000, chunk_size=1024,
         class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1]),
])
loss_input_convertion = dict(opus_pred_points='opus_pred_points',
    opus_pred_logits='opus_pred_logits', gaussian_point_logits='gaussian_point_logits',
    gaussian_point_points='gaussian_point_points',
    gaussian_point_valid_mask='gaussian_point_valid_mask')
