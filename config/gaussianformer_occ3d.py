_base_ = [
    './_base_/misc.py',
    './_base_/model.py',
    './_base_/surroundocc.py'
]

# =========== Occ3D data config (ego frame) ==============
# Occ3D labels are defined in the ego frame on a 200 x 200 x 16 grid.
dataset_name_flag = 'occ3d'
occ3d_eval_mask = 'camera'  # 'camera' | 'none' | 'lidar' | 'nonempty'

data_root = 'data/nuscenes/'
anno_root = 'data/nuscenes_cam/'
occ3d_path = 'data/occ3d/gts'
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
    'img', 'projection_mat', 'image_wh', 'occ_label', 'occ_xyz',
    'occ_cam_mask', 'occ_lidar_mask', 'occ_nonempty_mask', 'occ_loss_mask',
    'ori_img', 'cam_positions', 'focal_positions', 'lidar_points',
    'lidar_pose', 'ego_pose',
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
        pc_range=pc_range, grid_size=grid_size, model_coord='ego',
        train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadOccupancyOcc3D', occ3d_path=occ3d_path, semantic=True,
        pc_range=pc_range, grid_size=grid_size, model_coord='ego',
        train_mask_type='none'),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
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
    imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    pc_range=pc_range,
    occ3d=True,
    occ3d_coord='ego',
    phase='val',
    return_keys=occ3d_return_keys)
# =========== misc config ==============
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=2e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
# ========= model config ===============
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
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ])
        ])

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_loss_mask"
)
# ========= model config ===============
embed_dims = 128
num_decoder = 4
num_single_frame_decoder = 1
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = 'ckpts/raydn_r50_flash_704_bs2_seq_428q_nui_60e.pth'
semantics = True
semantic_dim = 17

model = dict(
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True),
    img_neck=dict(
        start_level=1),
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        phi_activation=phi_activation,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=2,
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.5, 0.5, 0.5],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=pc_range[:3],
            grid_size=grid_size),
    )
)
