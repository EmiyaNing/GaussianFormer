"""Stage-one sparse latent query forecasting with a frozen OPUSv2 teacher."""
_base_ = ['./forecasting_sparseworld_opusv2_tiny_r50_704x256_6f.py']

model = dict(
    _delete_=True,
    type='SparseLatentWorld', num_history=5, num_future=6, num_cams=6,
    current_occupancy_checkpoint='out/work_dirs/opusv2_t/official_mapped.pth',
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=4, norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True, style='pytorch', with_cp=False),
    img_neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
                  out_channels=256, start_level=0, num_outs=4),
    head=dict(
        type='OfficialOPUSV2Head', num_classes=17, in_channels=256,
        num_query=600, pc_range=[-40., -40., -1., 40., 40., 5.4],
        voxel_size=[.4, .4, .4], pfn_channels=[64, 64],
        empty_label=17, score_thr=.25,
        transformer=dict(
            embed_dims=256, num_frames=1, num_views=6, num_points=4,
            num_layers=5, num_levels=4, num_groups=4,
            num_refines=[8, 16, 32, 64, 128], num_pt_channels=32,
            scales=[.5], pc_range=[-40., -40., -1., 40., 40., 5.4])),
    predictor=dict(
        embed_dims=256, point_dims=32, num_points=128, num_classes=17,
        num_horizons=6, num_layers=3, num_heads=8,
        max_position_residual=12.),
    matcher=dict(position_weight=2., semantic_weight=1.,
                 feature_weight=.25, max_cost=20.))

loss = dict(
    _delete_=True, type='SparseLatentWorldLoss',
    occupancy_weight=1., position_weight=2., feature_weight=1.,
    semantic_weight=1., existence_weight=.25,
    occupancy_loss=dict(
        num_classes=17, empty_label=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
        focal_gamma=2., focal_alpha=.25, cls_loss_weight=2.,
        pts_loss_weight=.5, smooth_l1_beta=.2, empty_dist_thr=.2,
        empty_weight=5., rare_classes=[0, 2, 5, 8], rare_weight=10.,
        chunk_size=1024))
loss_input_convertion = dict(_delete_=True, stage1_output='stage1_output')

# Only the dynamics module is trainable; the checkpoint is the fixed teacher
# and the shared frozen decoder.
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=.01))
load_from = None
amp = False
grad_max_norm = 5
max_epochs = 36
eval_every_epochs = 6
warmup_iters = 500
min_lr_ratio = .05
enable_model_set_epoch = False
enable_dataset_set_epoch = True
train_dataset_config = dict(
    single_future_training=True,
    forecast_curriculum_start_epoch=0,
    forecast_curriculum_epochs_per_horizon=2)
