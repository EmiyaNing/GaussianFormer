"""Single-frame OPUS-V2 sparse-voxel discretization on Occ3D mini split."""
_base_ = ['../opus/opusv1_t_r50_704x256_1f_occ3d.py']

seed = 928
max_epochs = 5
eval_every_epochs = 1
train_dataset_config = dict(imageset='data/mini_occ3d_4096_1024/train.pkl')
val_dataset_config = dict(imageset='data/mini_occ3d_4096_1024/val.pkl')
train_loader = dict(batch_size=1, num_workers=4, shuffle=True)
val_loader = dict(batch_size=1, num_workers=4)

model = dict(
    encoder=dict(
        _delete_=True, type='OPUSV2SparseEncoder',
        embed_dims=256, num_decoder=5,
        num_frames=1, num_views=6, num_points=4, num_levels=4,
        num_groups=4, num_heads=8, feedforward_channels=512,
        dropout=0.1, point_feature_dims=32,
        num_refines=[8, 16, 32, 64, 128], scales=[0.5],
        pc_range=[-40., -40., -1., 40., 40., 5.4]),
    head=dict(
        _delete_=True, type='OPUSV2SparseHead',
        num_stages=5, point_feature_dims=32, num_classes=17,
        pc_range=[-40., -40., -1., 40., 40., 5.4],
        grid_shape=[200, 200, 16], empty_label=17,
        score_threshold=0.25))

loss = dict(_delete_=True, type='MultiLoss', loss_cfgs=[
    dict(type='OPUSV2SparseLoss', num_classes=17, empty_label=17,
         lambda_cls=2.0, lambda_pts=0.5, beta=0.2,
         focal_alpha=0.25, focal_gamma=2.0, chunk_size=1024)
])
loss_input_convertion = dict(
    _delete_=True,
    v2_pred_points='v2_pred_points',
    v2_sparse_logits='v2_sparse_logits',
    v2_sparse_coords='v2_sparse_coords',
    v2_init_points='v2_init_points')
