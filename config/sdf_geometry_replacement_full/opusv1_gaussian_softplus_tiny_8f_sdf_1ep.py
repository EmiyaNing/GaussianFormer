"""One complete epoch: replace all Set geometry with final-layer SDF losses."""
_base_ = ['../full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py']
custom_imports = dict(imports=['loss.gaussian_sdf_geometry_loss'], allow_failed_imports=False)
max_epochs = 1
seed = 928
load_from = 'ckpts/opus_gaussian_8f_tiny.pth'
train_load_from = load_from
gradient_accumulation = 4
finetune_base_lr = 2e-5
finetune_min_ratio = .1
finetune_warmup_updates = 100
sdf_cache_root = 'work_dirs/sdf_geometry_replacement_full/cache'

loss = dict(type='MultiLoss', loss_cfgs=[
    dict(type='OPUSSetClassificationLoss', stage_weights=[1.] * 6,
         loss_mode='official_v1', lambda_cls=2., focal_gamma=2., focal_alpha=.25,
         smooth_l1_beta=.2, lambda_pts=0., empty_dist_thr=.2, empty_weight=5.,
         rare_classes=[0, 2, 5, 8], rare_weight=10.,
         class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1],
         pc_range=[-40., -40., -1., 40., 40., 5.4], chunk_size=1024),
    dict(type='OccupancyLoss', weight=1., empty_label=17, num_classes=18,
         use_focal_loss=False, use_dice_loss=False, balance_cls_weight=True,
         multi_loss_weights=dict(loss_voxel_ce_weight=10., loss_voxel_lovasz_weight=1.),
         use_sem_geo_scal_loss=False, use_lovasz_loss=True, lovasz_ignore=17,
         manual_class_weight=[1.01552756, 1.06897009, 1.30013094, 1.07253735,
             .94637502, 1.10087012, 1.26960524, 1.06258364, 1.189019,
             1.06217292, 1.00595144, .85706115, 1.03923299, .90867526,
             .8936431, .85486129, .8527829, .5]),
    dict(type='GaussianSDFGeometryLoss', weight_center=.5, weight_shape=.5,
         delta=0., rho=.1, pc_range=[-40., -40., -1., 40., 40., 5.4]),
])
loss_input_convertion = dict(gaussian='gaussian')
