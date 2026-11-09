"""Five-epoch joint training of the img4f perceiver and the world predictor.

Starts from the locally trained four-frame perception checkpoint
(``ckpts/img_voxel_react_occ3d_fast_v5_lidar10f_img4f.pth``, epoch 17). The
predictor is re-initialized and the previous frozen-world optimizer state is
not reused. Images keep the perceiver's 704x256 recipe, ten LiDAR frames and
the no-camera-mask validation protocol.

The perceiver is unfrozen and the ResNet ``frozen_stages`` is opened to -1 so
the stem and first stage can adapt as well; the v5 fixed image BatchNorm keeps
its frozen statistics and affine parameters.
"""
_base_ = ['./img_voxel_react_occ3d_world_v1_lidar10f_img4f.py']

model = dict(
    freeze_perception=False,
    img_backbone=dict(frozen_stages=-1))

# Longest matching custom key wins: world_decoder must also override 'head'
# for predictor names such as world_decoder.traj_head.
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=.1),
        'img_neck': dict(lr_mult=.1),
        'lifter': dict(lr_mult=.1),
        'encoder': dict(lr_mult=.1),
        'head': dict(lr_mult=.1),
        'world_decoder': dict(lr_mult=1.),
    }))
# Apply one shared cosine/warm-up curve with fixed parameter-group ratios.
proportional_lr_groups = True
log_lr_groups = True
log_input_shapes = True
scheduler_step_before_optimizer = True
warmup_iters = 500
warmup_lr_init = 1e-6
min_lr_ratio = .1

# Supervise the current perception as well as the six future states and plan.
loss = dict(current_occupancy_weight=1.)
loss_input_convertion = dict(
    pred_occ='pred_occ', sampled_label='sampled_label',
    occ_loss_mask='occ_loss_mask')

max_epochs = 5
eval_every_epochs = 1
cuda_memory_log_interval = 100
# Avoid large cuDNN search workspaces during full perception backpropagation.
cudnn_benchmark = False
