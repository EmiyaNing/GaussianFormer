"""Readable template-sampling Gaussian-OPUS Version 2."""
_base_ = ['./gaussian_opusv1_t_r50_704x256_1f_occ3d.py']

semantic_dim = 17
num_templates = 600
num_refines = [2, 16, 32, 64, 128]
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

batch_size = 2
train_loader = dict(batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=4)

model = dict(
    encoder=dict(
        _delete_=True,
        type='GaussianOPUSV2Encoder',
        embed_dims=256,
        num_decoder=5,
        num_frames=1,
        num_views=6,
        num_levels=4,
        num_groups=4,
        num_heads=8,
        feedforward_channels=512,
        dropout=0.1,
        semantic_dim=semantic_dim,
        num_refines=num_refines,
        # Controls the fixed Gaussian sampling template size in each stage.
        stage_steps=[4.0, 3.6, 3.2, 2.8, 2.4],
        sampling_template=[
            [0., 0., 0.], [.45, 0., 0.], [-.45, 0., 0.],
            [0., .45, 0.], [0., -.45, 0.], [0., 0., .45], [0., 0., -.45],
        ],
        # Start from the deterministic template; enable learnable points after
        # the fixed-template run is stable.
        num_learnable_pts=6,
        learnable_fixed_scale=1.0,
        scale_range=[[.08, .08, .08], [.6, .6, .6]],
        center_step_multiplier=1.0,
        attn_drop=0.0,
        cross_stage_geometry_grad=True,
        pc_range=pc_range),
    head=dict(
        # Every V2 stage owns a semantic branch, so supervise each of them.
        apply_loss_type='fixed_0_2_4'),
)

# V2 has five decoder stages, so the centre supervision must use five weights.
loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[
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
        dict(type='GaussianCenterChamferLoss',
             stage_weights=[0.3, 0.42, 0.56, 0.72, 1.0],
             lambda_center=0.5, smooth_l1_beta=0.2, empty_label=17,
             chunk_size=1024, use_occ_loss_mask=False),
    ])
