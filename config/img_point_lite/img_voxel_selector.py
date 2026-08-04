"""图像/点云历史帧由组合熵增益动态选择的训练与评估配置。"""

_base_ = ['./img_voxel_history.py']

data_root = 'data/nuscenes/'
occ_path = 'data/surroundocc/samples'
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
input_shape = (1200, 648)
data_aug_conf = {
    'resize_lim': (1.0, 1.0),
    'final_dim': input_shape[::-1],
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900,
    'W': 1600,
    'rand_flip': True,
}

# 融合后的原始点云由 lifter 在 CPU 侧体素化，不做无效的 GPU 往返。
keep_lidar_points_cpu = True


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

entropy_selector = dict(
    type='EntropyBasedHistoryPointLoader',
    max_window=15,
    min_window=0,
    # e=0.05 表示累计收益达到完整窗口收益的 95% 后停止。
    entropy_gain_ratio_threshold=0.05,
    data_root=data_root,
    pc_range=pc_range,
    voxel_size=(0.5, 0.5, 0.5),
    max_points_per_voxel=20,
    max_voxels=1600000,
    deci_batch_size=65536,
    deci_topk=64,
    point_cache_size=64,
    log_selection=False,
)

selector_return_keys = [
    'img',
    'projection_mat',
    'image_wh',
    'occ_label',
    'occ_xyz',
    'occ_cam_mask',
    'ori_img',
    'cam_positions',
    'focal_positions',
    'lidar_points',
    'lidar_pose',
    'ego_pose',
    'num_lidar_history_frame',
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    entropy_selector,
    dict(
        type='LoadOccupancySurroundOcc',
        occ_path=occ_path,
        semantic=True,
        use_ego=False,
    ),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    entropy_selector,
    dict(
        type='LoadOccupancySurroundOcc',
        occ_path=occ_path,
        semantic=True,
        use_ego=False,
    ),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=False, num_cams=6),
]

# Dataset 本身不提前融合固定窗口；所有历史点云均由 selector 决定。
train_dataset_config = dict(
    pipeline=train_pipeline,
    data_aug_conf=data_aug_conf,
    num_lidar_history=0,
    return_keys=selector_return_keys,
    phase='train',
)

val_dataset_config = dict(
    pipeline=test_pipeline,
    data_aug_conf=data_aug_conf,
    num_lidar_history=0,
    return_keys=selector_return_keys,
    phase='val',
)

# Selector 是 CPU 密集型 Pipeline，使用持久 worker 与预取隐藏数据准备耗时。
train_loader = dict(
    batch_size=1,
    num_workers=4,
    shuffle=True,
    persistent_workers=True,
    prefetch_factor=2,
    limit_worker_threads=True,
)

val_loader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
    limit_worker_threads=True,
)

model = dict(
    freeze_img_backbone=True,
    freeze_img_neck=True,
    encoder=dict(      
        history_attn=None,
    ),
)
