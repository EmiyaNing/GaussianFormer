"""Six learned GS heads; each layer has full original Set and occupancy loss."""
_base_ = ['./small_baseline_1ep.py']
model = dict(encoder=dict(attribute_stages=[0, 1, 2, 3, 4, 5]),
             head=dict(apply_loss_type='all'))
loss = dict(loss_cfgs=[
    dict(type='OPUSSetLoss', stage_weights=[1.] * 6, loss_mode='official_v1',
         lambda_cls=2., focal_gamma=2., smooth_l1_beta=.2, lambda_pts=.5,
         empty_dist_thr=.2, empty_weight=5., rare_classes=[0, 2, 5, 8], rare_weight=10.,
         class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
         pc_range=[-40., -40., -1., 40., 40., 5.4], chunk_size=1024),
    dict(type='AllLayerOccupancyLoss', weight=1., stage_weights=[1.] * 6,
         empty_label=17, num_classes=18, use_focal_loss=False, use_dice_loss=False,
         balance_cls_weight=True,
         multi_loss_weights=dict(loss_voxel_ce_weight=10., loss_voxel_lovasz_weight=1.),
         use_sem_geo_scal_loss=False, use_lovasz_loss=True, lovasz_ignore=17,
         manual_class_weight=[1.01552756, 1.06897009, 1.30013094, 1.07253735,
             .94637502, 1.10087012, 1.26960524, 1.06258364, 1.189019,
             1.06217292, 1.00595144, .85706115, 1.03923299, .90867526,
             .8936431, .85486129, .8527829, .5]),
])
