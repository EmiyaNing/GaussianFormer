"""Strict port of DAOcc's official SurroundOcc experiment.

Reference:
configs/nuscenes/surroundocc/daocc_surroundocc.yaml
commit 6846432bbc1eff9d2a601900d21f1031df170a8c
"""

dataset_name_flag = 'surroundocc'
eval_mask_flag = True
occ3d_eval_mask = 'camera'

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]

train_dataset_config = dict(
    type='DAOccSurroundOccDataset',
    ann_file='data/nuscenes_infos_train_w_3occ.pkl',
    data_root='data/nuscenes',
    occ_root='data/surroundocc/samples',
    phase='train',
    num_sweeps=9,
    use_valid_flag=True,
    cbgs=True)
val_dataset_config = dict(
    type='DAOccSurroundOccDataset',
    ann_file='data/nuscenes_infos_val_w_3occ.pkl',
    data_root='data/nuscenes',
    occ_root='data/surroundocc/samples',
    phase='val',
    num_sweeps=9,
    use_valid_flag=True,
    cbgs=False)

train_loader = dict(
    batch_size=4,
    num_workers=4,
    shuffle=True,
    persistent_workers=True,
    prefetch_factor=2)
val_loader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2)

lidar_encoder = dict(
    type='SparseEncoder',
    in_channels=5,
    # MMDetection3D 1.1.1/spconv 2.x uses z-y-x sparse shape order.
    sparse_shape=[41, 1440, 1440],
    output_channels=128,
    order=('conv', 'norm', 'act'),
    encoder_channels=(
        (16, 16, 32),
        (32, 32, 64),
        (64, 64, 128),
        (128, 128)),
    encoder_paddings=(
        (0, 0, 1),
        (0, 0, 1),
        (0, 0, (0, 1, 1)),
        (0, 0)),
    block_type='basicblock')

detection_head = dict(
    type='CenterHead',
    in_channels=512,
    tasks=[
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ],
    common_heads=dict(
        reg=(2, 2), height=(1, 2), dim=(3, 2),
        rot=(2, 2), vel=(2, 2)),
    share_conv_channel=64,
    bbox_coder=dict(
        type='CenterPointBBoxCoder',
        pc_range=point_cloud_range[:2],
        post_center_range=[
            -61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=9),
    separate_head=dict(
        type='SeparateHead', init_bias=-2.19, final_kernel=3),
    loss_cls=dict(
        type='mmdet.GaussianFocalLoss', reduction='mean'),
    loss_bbox=dict(
        type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    norm_bbox=True,
    train_cfg=dict(
        point_cloud_range=point_cloud_range,
        grid_size=[1440, 1440, 41],
        voxel_size=voxel_size,
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 0.2, 0.2]),
    test_cfg=dict(
        post_center_limit_range=[
            -61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2))

model = dict(
    type='DAOccSegmentor',
    img_backbone=dict(
        type='DAOccResNetReLU6',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.pytorch.org/models/resnet50-0676ba61.pth')),
    lidar_encoder=lidar_encoder,
    detection_head=detection_head,
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    max_num_points=10,
    max_voxels=(120000, 160000),
    detection_loss_weight=0.01)

loss = dict(type='MultiLoss', loss_cfgs=[dict(type='DAOccLoss')])
loss_input_convertion = dict(loss_total='loss_total')

optimizer = dict(
    optimizer=dict(type='AdamW', lr=2.0e-4, weight_decay=0.01))
max_epochs = 6
warmup_iters = 500
min_lr_ratio = 1.0e-3
grad_max_norm = 35
# The official DAOcc runner enables fp16 through Fp16OptimizerHook.  Use
# native PyTorch AMP in the GaussianFormer runner for the equivalent path.
amp = True
syncBN = True
find_unused_parameters = False
print_freq = 50
eval_every_epochs = 2
load_from = None
