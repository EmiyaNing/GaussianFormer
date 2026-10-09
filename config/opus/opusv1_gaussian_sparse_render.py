"""OPUS-V1 child Gaussians rendered with GaussianFormer local_aggregate."""
_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

model = dict(
    encoder=dict(
        _delete_=True,
        type='OPUSV1GaussianEncoder', embed_dims=256, num_decoder=6,
        num_frames=1, num_views=6, num_points=4, num_levels=4,
        num_groups=4, num_heads=8, feedforward_channels=512,
        dropout=0.1, num_classes=17,
        num_refines=[1, 4, 16, 32, 64, 128], scales=[0.5],
        pc_range=pc_range,
        scale_min=0.08, scale_max=0.40, initial_scale=0.20,
        initial_opacity=0.10, attribute_stages=[5]),
    head=dict(
        _delete_=True, type='OPUSV1GaussianHead',
        apply_loss_type='fixed_5', max_render_scale=0.40,
        num_classes=18, empty_label=17, with_empty=True,
        use_localaggprob=False, use_localagg_react=False,
        dataset_type='nusc',
        empty_args=dict(mean=[0.0, 0.0, -1.0],
                        scale=[100.0, 100.0, 8.0]),
        cuda_kwargs=dict(scale_multiplier=3, H=200, W=200, D=16,
                         pc_min=pc_range[:3], grid_size=0.4)),
)

loss = dict(type='MultiLoss', loss_cfgs=[
    dict(type='OPUSSetLoss', stage_weights=[1.0] * 6,
         loss_mode='official_v1', lambda_cls=2.0, focal_gamma=2.0,
         smooth_l1_beta=0.2, lambda_pts=0.5, empty_dist_thr=0.2,
         empty_weight=5.0, rare_classes=[0, 2, 5, 8], rare_weight=10.0,
         class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
         pc_range=pc_range, chunk_size=1024),
    dict(type='OccupancyLoss', weight=1.0, empty_label=17, num_classes=18,
         use_focal_loss=False, use_dice_loss=False, balance_cls_weight=True,
         multi_loss_weights=dict(loss_voxel_ce_weight=10.0,
                                 loss_voxel_lovasz_weight=1.0),
         use_sem_geo_scal_loss=False, use_lovasz_loss=True, lovasz_ignore=17,
         manual_class_weight=[
             1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502,
             1.10087012, 1.26960524, 1.06258364, 1.189019, 1.06217292,
             1.00595144, 0.85706115, 1.03923299, 0.90867526, 0.8936431,
             0.85486129, 0.8527829, 0.5]),
])
loss_input_convertion = dict(
    _delete_=True,
    opus_pred_points='opus_pred_points',
    opus_pred_logits='opus_pred_logits',
    opus_point_valid_masks='opus_point_valid_masks',
    pred_occ='pred_occ', sampled_xyz='sampled_xyz',
    sampled_label='sampled_label', occ_mask='occ_loss_mask')
