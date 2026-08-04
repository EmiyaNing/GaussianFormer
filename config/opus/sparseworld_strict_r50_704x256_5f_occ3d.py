"""Isolated reproduction of the public SparseWorld trajectory configuration.

This config intentionally shares only data/backbone definitions with the local
OPUS setup. It registers no behavioral change for existing SparseWorld configs.
"""
_base_ = ['./opusv1_t_r50_704x256_1f_occ3d.py']

num_frames = 5
future_steps = 6
future_queries = [60, 60, 60, 60, 40, 40]
pc_range = [-40., -40., -1., 40., 40., 5.4]
grid_size, grid_shape, max_gt_points = .4, [200, 200, 16], 76800
data_root, anno_root, occ3d_path = 'data/nuscenes/', 'data/nuscenes_cam/', 'data/occ3d/gts'
input_shape = (704, 256)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_aug_conf = dict(resize_lim=(.38,.55), final_dim=input_shape[::-1], bot_pct_lim=(0.,0.), rot_lim=(0.,0.), H=900, W=1600, rand_flip=True)

model = dict(
    type='SparseWorldStrictSegmentor',
    lifter=dict(_delete_=True, type='SparseWorldStrictQueryLifter', num_queries=720, future_queries=future_queries, embed_dims=256),
    encoder=dict(_delete_=True, type='SparseWorldStrictEncoder', embed_dims=256, num_decoder=6, num_frames=num_frames, num_views=6, num_points=4, num_levels=4, num_groups=4, num_heads=8, feedforward_channels=512, dropout=.1, num_classes=17, num_refines=[1,4,16,24,32,48], scales=[.5], pc_range=pc_range),
    head=dict(_delete_=True, type='SparseWorldStrictHead', num_classes=17, point_multipliers=[1,4,16,24,32,48], pc_range=pc_range, grid_size=grid_size, grid_shape=grid_shape, empty_label=17, score_threshold=[.35]*15+[.25,.3], center_distance_threshold=3., padding=True),
    future_queries=future_queries, finetune_epoch=5, embed_dims=256, num_refines=48, pc_range=pc_range, ego_state_dim=21)

enable_model_set_epoch = True
sparseworld_future_eval = dict(enabled=True, horizons=[1,2,3,4,5,6], report_occ_iou=True)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiViewImageHistory', num_history=num_frames-1, num_cams=6, to_float32=True, pad_history=False),
    dict(type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True, pc_range=pc_range, grid_size=grid_size, model_coord='ego', train_mask_type='none'),
    dict(type='LoadSparseWorldFutureOccupancy', occ3d_path=occ3d_path), dict(type='PrepareOPUSTarget', max_gt_points=max_gt_points, empty_label=17),
    dict(type='ResizeCropFlipImage'), dict(type='PhotoMetricDistortionMultiViewImage'), dict(type='NormalizeMultiviewImage', **img_norm_cfg), dict(type='DefaultFormatBundle'), dict(type='NuScenesAdaptor', use_ego=True, num_cams=30)]
trajectory_return_keys = ['img','projection_mat','image_wh','occ_label','occ_xyz','occ_cam_mask','occ_lidar_mask','occ_nonempty_mask','occ_loss_mask','future_occ_labels','future_occ_cam_masks','future_occ_lidar_masks','future_ego_to_current','temporal_trajs','temporal_ego_states']
train_dataset_config = dict(type='NuScenesSparseWorldTrajectoryDataset', data_root=data_root, imageset=anno_root+'nuscenes_infos_train_sweeps_occ.pkl', data_aug_conf=data_aug_conf, pipeline=train_pipeline, pc_range=pc_range, occ3d=True, occ3d_coord='ego', phase='train', admlp_state_path='admlp/stp3_val/data_nuscene.pkl', trajectory_path='occworld/nuscenes_infos_train_temporal_v3_scene.pkl', return_keys=trajectory_return_keys)
test_pipeline = [v for v in train_pipeline if v.get('type') != 'PhotoMetricDistortionMultiViewImage']
val_dataset_config = dict(type='NuScenesSparseWorldTrajectoryDataset', data_root=data_root, imageset=anno_root+'nuscenes_infos_val_sweeps_occ.pkl', data_aug_conf=data_aug_conf, pipeline=test_pipeline, pc_range=pc_range, occ3d=True, occ3d_coord='ego', phase='val', admlp_state_path='admlp/stp3_val/data_nuscene.pkl', trajectory_path='occworld/nuscenes_infos_val_temporal_v3_scene.pkl', return_keys=trajectory_return_keys)
train_loader = dict(batch_size=2,num_workers=4,shuffle=True); val_loader=dict(batch_size=2,num_workers=4)
loss = dict(_delete_=True, type='SparseWorldStrictLoss', current_loss=dict(type='OPUSSetLoss',stage_weights=[1.]*6,loss_mode='official_v1',lambda_cls=2.,focal_gamma=2.,smooth_l1_beta=.2,lambda_pts=.5,empty_dist_thr=.2,empty_weight=5.,rare_classes=[0,2,5,8],rare_weight=10.,class_weights=[10,5,10,5,5,10,10,5,10,5,5,1,5,1,1,2,1],pc_range=pc_range), pc_range=pc_range)
loss_input_convertion = dict(opus_pred_points='opus_pred_points',opus_pred_logits='opus_pred_logits',future_pred_points='future_pred_points',future_pred_logits='future_pred_logits',pred_traj='pred_traj',temporal_pred_points='temporal_pred_points',temporal_pred_logits='temporal_pred_logits',sparseworld_pretrain='sparseworld_pretrain')
optimizer = dict(optimizer=dict(type='AdamW',lr=2e-4,weight_decay=.01), paramwise_cfg=dict(custom_keys={'img_backbone':dict(lr_mult=.1),'sampling_offset':dict(lr_mult=.1)})); grad_max_norm=5; max_epochs=64
load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
