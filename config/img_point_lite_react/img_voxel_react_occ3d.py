_base_ = [
    './img_voxel_react.py'
]

# =========== Occ3D ego data config ==============
dataset_name_flag = 'occ3d'
occ3d_eval_mask = 'camera'  # 'camera' | 'none' | 'lidar' | 'nonempty'

data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ3d_path = "data/occ3d/gts"

pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
grid_size = 0.4
input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

occ3d_return_keys = [
    'img',
    'projection_mat',
    'image_wh',
    'occ_label',
    'occ_xyz',
    'occ_cam_mask',
    'occ_lidar_mask',
    'occ_nonempty_mask',
    'occ_loss_mask',
    'ori_img',
    'cam_positions',
    'focal_positions',
    'lidar_points',
    'lidar_pose',
    'ego_pose',
]

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadOccupancyOcc3D",
        occ3d_path=occ3d_path,
        semantic=True,
        pc_range=pc_range,
        grid_size=grid_size,
        model_coord='ego',
        train_mask_type='none'),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadOccupancyOcc3D",
        occ3d_path=occ3d_path,
        semantic=True,
        pc_range=pc_range,
        grid_size=grid_size,
        model_coord='ego',
        train_mask_type='none'),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=True, num_cams=6),
]

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    pc_range=pc_range,
    occ3d=True,
    occ3d_coord='ego',
    phase='train',
    return_keys=occ3d_return_keys)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    pc_range=pc_range,
    occ3d=True,
    occ3d_coord='ego',
    phase='val',
    return_keys=occ3d_return_keys)

loss_input_convertion = dict(
    pred_occ="pred_occ",
    gaussian="gaussian",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_loss_mask",
)

# ========= Occ3D ego model-space overrides ===============
scale_range = [0.08, 0.64]
semantic_dim = 17

model = dict(
    lifter=dict(
        pc_range=pc_range,
        voxel_size=grid_size),
    encoder=dict(
        deformable_model=dict(
            kps_generator=dict(
                pc_range=pc_range,
                scale_range=scale_range)),
        refine_layer=dict(
            pc_range=pc_range,
            scale_range=scale_range),
        densify_layer=dict(
            pc_range=pc_range,
            scale_range=scale_range),
        spconv_layer=dict(
            pc_range=pc_range,
            grid_size=[grid_size, grid_size, grid_size])),
    head=dict(
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, 2.2],
            scale=[80, 80, 6.4]),
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=3,
            H=200,
            W=200,
            D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=grid_size)))
