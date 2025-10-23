_base_ = [
    '../_base_/misc.py',
    '../_base_/model.py',
    '../_base_/surroundocc.py'
]

# =========== data config ==============
# SemanticKITTI图像尺寸
input_shape = (1224, 370)
data_aug_conf = {
    "resize_lim": (0.8, 1.2),
    "final_dim": (320, 1216),  # 调整后的尺寸
    "bot_pct_lim": (0.0, 0.2),
    "rot_lim": (-5.4, 5.4),
    "H": 370,
    "W": 1224,
    "rand_flip": True,
}

# SemanticKITTI数据集配置
dataset_type = 'SemanticKITTIDataset'
data_root = 'data/semanticKITTI/dataset/'
occ_path = data_root  # 使用相同路径

# 训练和验证序列划分
train_sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
val_sequences = ['08']

# 训练数据流水线 - 与surroundocc.py保持相同的顺序和结构
train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, color_type='unchanged'),
    dict(type="LoadOccupancySemanticKITTI", occ_path=occ_path, semantic=True, training=True, use_fov_filter=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),  # 添加PhotoMetricDistortion
    dict(type="NormalizeMultiviewImage", 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_rgb=True),
    dict(type="DefaultFormatBundle"),
    dict(type="SemanticKITTIAdaptor", use_ego=False),  # 只使用1个相机
    dict(type='LoadKITTILiDARFromFile', pc_range=[0.0, -25.6, -2.0, 51.2, 25.6, 4.4], filter_points=True),
]

# 验证数据流水线
val_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True, color_type='unchanged'),
    dict(type="LoadOccupancySemanticKITTI", occ_path=occ_path, semantic=True, training=False, use_fov_filter=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_rgb=True),
    dict(type="DefaultFormatBundle"),
    dict(type="SemanticKITTIAdaptor", use_ego=False),  # 只使用1个相机
    dict(type='LoadKITTILiDARFromFile', pc_range=[0.0, -25.6, -2.0, 51.2, 25.6, 4.4], filter_points=True),
]

train_dataset_config = dict(
    _delete_=True, 
    type=dataset_type,
    data_root=data_root,
    sequences=train_sequences,
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
    use_fov_filter=True  # 启用FOV过滤
)

val_dataset_config = dict(
    _delete_=True, 
    type=dataset_type,
    data_root=data_root,
    sequences=val_sequences,
    data_aug_conf=data_aug_conf,
    pipeline=val_pipeline,
    phase='val',
    use_fov_filter=True  # 启用FOV过滤
)

# 数据加载器配置 - 与surroundocc.py保持一致
batch_size = 1

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2
)

# =========== misc config ==============
# 与nuscenes_gs6400.py完全一致
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=4e-4, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35

# ========= model config ===============
# 与nuscenes_gs6400.py完全一致的模型配置
embed_dims = 128
num_decoder = 4
# SemanticKITTI的坐标范围
pc_range = [0.0, -25.6, -2.0, 51.2, 25.6, 4.4]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 19  # SemanticKITTI有19个语义类别（不包括unlabeled）

# 损失函数配置 - 与nuscenes_gs6400.py结构相同，但调整类别数
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=0,  # SemanticKITTI中unlabeled是0
            num_classes=20,  # 包括unlabeled共20个类别
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=0,  # 忽略unlabeled类别
            manual_class_weight=None,
            ignore_empty=False,  # 与nuscenes_gs6400.py保持一致
            lovasz_use_softmax=False),
        dict(
            type="PixelDistributionLoss",
            weight=1.0,
            use_sigmoid=False),
    ])

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
    bin_logits="bin_logits",
    density="density",
    pixel_logits="pixel_logits",
    pixel_gt="pixel_gt"
)

# 模型配置 - 与nuscenes_gs6400.py完全一致，只调整输出维度和坐标范围
model = dict(
    freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        start_level=1),
    lifter=dict(
        type='GaussianLifterV2',
        num_anchor=6400,
        embed_dims=embed_dims,
        anchor_grad=False,
        feat_grad=False,
        semantics=semantics,
        semantic_dim=semantic_dim,  # 修改为19
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        pretrained_path="out/prob/init/init.pth",
        deterministic=False,
        random_samples=0),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim  # 修改为19
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            _delete_=True,
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            num_cams=1,
            residual_mode="none",
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,  # 使用笛卡尔坐标
                num_learnable_pts=6,
                pc_range=pc_range,  # 使用SemanticKITTI范围
                scale_range=scale_range,
                learnable_fixed_scale=6.0,
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,  # 使用SemanticKITTI范围
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,  # 修改为19
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,  # 使用笛卡尔坐标
            semantics_activation='identity',
        ),
        spconv_layer=dict(
            _delete_=True,
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,  # 使用SemanticKITTI范围
            grid_size=[1.0, 1.0, 1.0],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,  # 使用笛卡尔坐标
            use_out_proj=True,
            use_multi_layer=True,
        ),
        num_decoder=num_decoder,
        operation_order=[
            "identity",
            "deformable",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",

            "identity",
            "spconv",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",
            
            "refine",
        ] * num_decoder,
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,  # 20个类别（包括unlabeled）
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=4,
            H=256, W=256, D=32,  # 调整网格尺寸以匹配SemanticKITTI的256x256x32
            pc_min=[0.0, -25.6, -2.0],
            grid_size=0.2),  # SemanticKITTI的体素分辨率是0.2m
    )
)