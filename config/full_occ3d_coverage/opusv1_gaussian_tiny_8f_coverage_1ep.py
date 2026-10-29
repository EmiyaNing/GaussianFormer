"""Original complete objective + center-only core-coverage auxiliary, 1 epoch.

Run with tools/train_coverage_finetune.py; it handles accumulation tails and
restores original AdamW moments while starting a new fine-tuning schedule.
"""
_base_ = ['../full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py']
custom_imports = dict(imports=['loss.gaussian_core_coverage_loss'], allow_failed_imports=False)

load_from = 'ckpts/opus_gaussian_8f_tiny.pth'
train_load_from = load_from
train_load_revise_keys = []
train_load_strict = True
max_epochs = 1
seed = 928
gradient_accumulation = 4
optimizer = dict(optimizer=dict(lr=2e-5))
finetune_base_lr = 2e-5
source_optimizer_base_lr = 2e-4
finetune_min_ratio = .1
finetune_warmup_updates = 100
save_every_updates = 500
expected_train_frames = 28130
expected_train_scenes = 700
expected_val_frames = 6019
expected_val_scenes = 150

loss = dict(type='MultiLoss', loss_cfgs=[
    dict(type='OPUSSetLoss', stage_weights=[1.] * 6,
         loss_mode='official_v1', lambda_cls=2., focal_gamma=2.,
         smooth_l1_beta=.2, lambda_pts=.5, empty_dist_thr=.2, empty_weight=5.,
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
    dict(type='GaussianCoreCoverageLoss', weight=.1, empty_label=17,
         grid_shape=[200, 200, 16], pc_range=[-40., -40., -1., 40., 40., 5.4],
         grid_size=.4, max_scale=.4, chunk_size=2048, knn_chunk_size=1024),
])
loss_input_convertion = dict(gaussian='gaussian')
