_base_ = ['./adaptive_allocationv5_lite.py']

# V6 keeps the four concepts as training-time proxy targets rather than adding
# explicit diagnostic heads.  The auxiliary loss trains risk ranking, operation
# routing, branch outcomes, and the renderer-cost budget.
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0,
            ),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735,
                0.94637502, 1.10087012, 1.26960524, 1.06258364,
                1.189019, 1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431, 0.85486129,
                0.8527829, 0.5,
            ],
        ),
        dict(
            type='AdaptiveAllocationV6Loss',
            weight=1.0,
            empty_label=17,
            semantic_dim=17,
            support_k=27,
            support_radius=2.0,
            coverage_k=8,
            coverage_radius=3.0,
            max_coverage_voxels=65536,
            risk_bce_weight=0.1,
            risk_ranking_weight=0.1,
            operation_kl_weight=0.02,
            budget_weight=0.05,
            outcome_weight=0.02,
        ),
    ],
)

loss_input_convertion = dict(
    pred_occ='pred_occ',
    gaussian='gaussian',
    sampled_xyz='sampled_xyz',
    sampled_label='sampled_label',
    occ_mask='occ_mask',
    allocation_aux='allocation_aux',
)

model = dict(
    encoder=dict(
        densify_layer=dict(
            type='AdaptiveAllocationV6',
            feat_embed_dim=128,
            semantic_dim=17,
            pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
            scale_range=[0.17, 0.64],
            unit_xyz=[4.0, 4.0, 1.0],
            allocation_ratio=0.15,
            router_temperature=1.0,
            selector_temperature=0.1,
            risk_floor=0.1,
            neighbor_k=8,
            neighbor_radius=2.0,
            cost_grid_size=0.5,
            cost_scale_multiplier=3.0,
            cost_budget_ratio=1.3,
            cost_priority_power=0.5,
            cache_router_stats=False,
        ),
    ),
)
