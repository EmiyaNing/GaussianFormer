"""Phase-A Gaussian-OPUS: fixed-cardinality Gaussian templates + localagg."""
_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

semantic_dim = 17
num_templates = 600
num_refines = [1, 4, 8, 32, 16, 32]
# MMEngine resolves a child config in its own Python scope; base-config
# variables are merged afterwards and are therefore not names usable here.
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
# GaussianOPUSHead supports B>1 by sequentially invoking the single-scene
# localagg kernel.  Keep B=1 by default because renderer memory still scales
# approximately linearly with B; users may set this to 2 after profiling.
batch_size = 2
train_loader = dict(batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = dict(batch_size=batch_size, num_workers=4)

model = dict(
    lifter=dict(
        _delete_=True, type='SemanticGaussianTemplateLifter', num_templates=num_templates,
        embed_dims=256, semantic_dim=semantic_dim, pc_range=pc_range,
        scale_range=[[0.15, 0.15, 0.15], [8.0, 8.0, 4.0]],
        initial_scale=[1.6, 1.6, 1.0], initial_opacity=0.1,
        query_grad=True, feature_grad=True),
    encoder=dict(
        _delete_=True,
        type='GaussianOPUSEncoder', 
        embed_dims=256, 
        num_decoder=6,
        num_frames=1, 
        num_views=6, 
        num_points=4, 
        num_levels=4, 
        num_groups=4,
        num_heads=8, 
        feedforward_channels=512, 
        dropout=0.1,
        semantic_dim=semantic_dim, 
        num_refines=num_refines,
        stage_steps=[1.0, 0.85, 0.7, 0.55, 0.4, 0.3],
        # Preserve the means/scales graph between decoder stages: later
        # occupancy/CD losses directly refine earlier Gaussian geometry.
        cross_stage_geometry_grad=True,
        pc_range=pc_range),
    head=dict(
        _delete_=True,
        type='GaussianOPUSHead', 
        # Render only the final stage during Phase-A training to constrain
        # localagg activation memory.  GaussianHead expects an underscore.
        apply_loss_type='random_1',
        num_classes=18,
        empty_label=17, 
        with_empty=True, 
        use_localaggprob=False,
        use_localagg_react=False, 
        dataset_type='nusc',
        empty_args=dict(mean=[0.0, 0.0, -1.0], scale=[100.0, 100.0, 8.0]),
        cuda_kwargs=dict(
            scale_multiplier=3, 
            H=200, W=200, D=16,
            pc_min=pc_range[:3], 
            grid_size=grid_size)),
)

# Intermediate semantic/rotation/opacity heads are not consumed by the
# Phase-A cross-stage recurrence. Keep DDP robust while those branches remain
# intentionally unsupervised. This is consumed by train.py, not model builder.
find_unused_parameters = True

loss = dict(type='MultiLoss', loss_cfgs=[
    # Preserve the GaussianFormer occupancy objective and its CE/Lovasz
    # implementation; Gaussian-OPUS only changes where pred_occ comes from.
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
    # OPUS-style symmetric KNN assignment + Smooth-L1 centre Chamfer term.
    dict(type='GaussianCenterChamferLoss',
         stage_weights=[0.2, 0.3, 0.42, 0.56, 0.72, 1.0],
         lambda_center=0.5, smooth_l1_beta=0.2, empty_label=17,
         chunk_size=1024, use_occ_loss_mask=False),
])

loss_input_convertion = dict(_delete_=True, pred_occ='pred_occ', sampled_xyz='sampled_xyz',
                             sampled_label='sampled_label', occ_mask='occ_loss_mask',
                             gaussians='gaussians')
