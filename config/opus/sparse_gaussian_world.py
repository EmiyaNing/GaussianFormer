"""Five-epoch SparseWorld + point-anchored Semantic Gaussian fine-tuning."""
_base_ = ['./sparseworld_strict.py']

pc_range = [-40., -40., -1., 40., 40., 5.4]
future_queries = [60, 60, 60, 60, 40, 40]

model = dict(
    type='SparseGaussianWorld',
    # The warm start is already the released epoch-56 fine-tuned model.
    finetune_epoch=0,
    freeze_sparseworld=False,
    semantic_gaussian=dict(
        type='PointAnchoredSemanticGaussianCUDA',
        embed_dims=256, num_children=48, num_classes=17,
        pc_range=pc_range,
        scale_range=[[.1, .1, .1], [.6, .6, .6]],
        initial_scale=.35, initial_opacity=.02,
        num_neighbors=8, include_self=True, query_chunk_size=256,
        cell_size=1.8, support_sigma=3., scale_eps=1e-4,
        denom_eps=1e-6),
    gaussian_horizons=[0, 2, 4, 6],
    eval_mode='ensemble', ensemble_gamma=1.)

loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='SparseWorldStrictLoss', pc_range=pc_range,
            voxel_size=.4, num_query=720, future_queries=future_queries,
            class_weights=[3, 2, 3, 2, 2, 3, 3, 2, 3,
                           2, 2, 1, 2, 1, 1, 1, 1],
            focal_alpha=.25, focal_gamma=2., focal_weight=2.,
            smooth_l1_beta=.2, point_weight=.5, empty_dist_thr=.2,
            empty_weight=3., chunk_size=1024),
        dict(
            type='SparseWorldGaussianPointOccupancyLoss', weight=1.,
            empty_label=17, focal_gamma=2., focal_alpha=.25,
            warmup_iters=1000, chunk_size=1024,
            horizon_weights=[1., .5, .5, .5, .5, .5, .5],
            enable_during_pretrain=False,
            class_weights=[3, 2, 3, 2, 2, 3, 3, 2, 3,
                           2, 2, 1, 2, 1, 1, 1, 1])])

loss_input_convertion = dict(
    _delete_=True,
    initial_pred_points='initial_pred_points',
    temporal_pred_points='temporal_pred_points',
    temporal_pred_logits='temporal_pred_logits',
    current_pred_points='current_pred_points',
    current_pred_logits='current_pred_logits',
    future_pred_points='future_pred_points',
    future_pred_logits='future_pred_logits',
    future_pred_masks='future_pred_masks',
    pred_traj='pred_traj', sparseworld_pretrain='sparseworld_pretrain',
    query_stamps='query_stamps', stamp_statistics='stamp_statistics',
    update_stamp_statistics='update_stamp_statistics',
    current_gaussian_point_points='current_gaussian_point_points',
    current_gaussian_point_logits='current_gaussian_point_logits',
    current_gaussian_point_valid_mask='current_gaussian_point_valid_mask',
    future_gaussian_point_points='future_gaussian_point_points',
    future_gaussian_point_logits='future_gaussian_point_logits',
    future_gaussian_point_valid_masks='future_gaussian_point_valid_masks',
    future_gaussian_horizons='future_gaussian_horizons',
    gaussian_scales='gaussian_scales',
    gaussian_opacities='gaussian_opacities',
    gaussian_residual='gaussian_residual')

# Measured at the largest epoch-5 training horizon: B=3 uses 18.4 GiB on a
# 24-GiB RTX 4090, leaving margin for the allocator and dataloader variance.
# Validation follows the strict B=1 protocol.
train_loader = dict(batch_size=2, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)

max_epochs = 50
eval_every_epochs = 1
amp = False
find_unused_parameters = False
strict_load = True
sparse_gaussian_warmstart = True
#train_load_from = '../SparseWorld/ckpts/epoch_56.pth'
train_load_revise_keys = []
train_load_strict = False
load_from = None
