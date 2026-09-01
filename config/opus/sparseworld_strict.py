"""Native SparseWorldStrict reproduction in the GaussianFormer environment."""

seed = 17
dataloader_seed = 17
dataset_name_flag = 'occ3d'
sparseworld_strict_eval = True
strict_load = True
enable_model_set_epoch = True
official_checkpoint = '../SparseWorld/ckpts/epoch_56.pth'

data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
occ3d_path = 'data/occ3d'
pc_range = [-40., -40., -1., 40., 40., 5.4]
voxel_size = [.4, .4, .4]
grid_shape = [200, 200, 16]
num_frames = 5
future_steps = 6
future_queries = [60, 60, 60, 60, 40, 40]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375], to_rgb=True)
data_aug_conf = dict(
    resize_lim=(.38, .55), final_dim=(256, 704),
    bot_pct_lim=(0., 0.), rot_lim=(0., 0.), H=900, W=1600,
    rand_flip=True)

model = dict(
    type='SparseWorldStrict', out_dim=256, finetune_epoch=5,
    is_pretrain=False, feature_mode='official_auto',
    data_aug=dict(img_color_aug=True, img_norm_cfg=img_norm_cfg,
                  img_pad_cfg=dict(size_divisor=32)),
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4,
        out_indices=(0, 1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True), norm_eval=True,
        style='pytorch', with_cp=True),
    img_neck=dict(
        type='FPN', in_channels=[256, 512, 1024, 2048],
        out_channels=256, num_outs=4),
    pts_bbox_head=dict(
        type='SparseWorldStrictHead', num_classes=17, in_channels=256,
        num_query=720, num_fu_frames=future_steps,
        num_fu_query=future_queries, pc_range=pc_range,
        voxel_size=voxel_size, is_pretrain=False,
        transformer=dict(
            type='SparseWorldTransformer', embed_dims=256,
            num_frames=num_frames, num_views=6, num_points=4,
            num_layers=6, num_levels=4, num_classes=17, num_groups=4,
            num_refines=[1, 4, 16, 24, 32, 48], scales=[.5],
            pc_range=pc_range),
        train_cfg=dict(cls_weights=[3, 2, 3, 2, 2, 3, 3, 2, 3,
                                    2, 2, 1, 2, 1, 1, 1, 1]),
        test_cfg=dict(score_thr=[.35] * 15 + [.25, .3], padding=True)))

return_keys = [
    'img', 'projection_mat', 'image_wh', 'occ_label', 'occ_xyz',
    'occ_cam_mask', 'occ_lidar_mask', 'occ_nonempty_mask', 'occ_loss_mask',
    'future_occ_labels', 'future_occ_cam_masks', 'future_occ_lidar_masks',
    'future_ego_to_current', 'temporal_trajs', 'temporal_ego_states',
    'ego2lidar', 'img_filename']

common_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False,
         color_type='color'),
    dict(type='LoadMultiViewImageHistory', num_history=4, num_cams=6,
         to_float32=False, color_type='color', pad_history=False),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
         pc_range=pc_range, grid_size=.4, model_coord='ego',
         train_mask_type='none'),
    dict(type='LoadSparseWorldFutureOccupancy', occ3d_path=occ3d_path),
    dict(type='ResizeCropFlipImage'),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=30)]

train_dataset_config = dict(
    type='NuScenesSparseWorldStrictDataset', data_root=data_root,
    imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
    data_aug_conf=data_aug_conf, pipeline=common_pipeline, pc_range=pc_range,
    occ3d=True, occ3d_coord='ego', phase='train', future_steps=future_steps,
    history_frames=4, admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
    trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl',
    return_keys=return_keys)
val_dataset_config = dict(
    type='NuScenesSparseWorldStrictDataset', data_root=data_root,
    imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
    data_aug_conf=data_aug_conf, pipeline=common_pipeline, pc_range=pc_range,
    occ3d=True, occ3d_coord='ego', phase='val', future_steps=future_steps,
    history_frames=4, admlp_state_path='admlp/stp3_val/data_nuscene.pkl',
    trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl',
    return_keys=return_keys)
train_loader = dict(batch_size=2, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)

loss = dict(
    type='SparseWorldStrictLoss', pc_range=pc_range, voxel_size=.4,
    num_query=720, future_queries=future_queries,
    class_weights=[3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1],
    focal_alpha=.25, focal_gamma=2., focal_weight=2.,
    smooth_l1_beta=.2, point_weight=.5, empty_dist_thr=.2,
    empty_weight=3., chunk_size=1024)
loss_input_convertion = dict(
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
    update_stamp_statistics='update_stamp_statistics')

optimizer = dict(optimizer=dict(type='AdamW', lr=2e-4, weight_decay=.01))
grad_max_norm = 5
max_epochs = 64
warmup_iters = 500
warmup_lr_init = 2e-4 / 3
min_lr_ratio = 1e-3
scheduler_step_before_optimizer = True
amp = False
eval_every_epochs = 24
print_freq = 50
syncBN = True
find_unused_parameters = False
load_from = None
train_load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
train_load_revise_keys = [('backbone.', 'img_backbone.')]
train_load_strict = False
