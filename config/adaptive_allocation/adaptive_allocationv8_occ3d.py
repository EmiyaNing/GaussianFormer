"""Joint training: baseline child generator plus a detached parent ranker.

Same v5_mem execution path, 9 historical LiDAR frames, fixed +2560 children.
No allocator checkpoint or freezing is required. load_from retains the
baseline image pretraining. Durations count training forwards per GPU.
"""

_base_ = ["../img_point_lite_react/img_voxel_react_occ3d_fast_v5_mem.py"]
separate_allocation_grad_clip = True
model = dict(
    freeze_img_backbone=False,
    freeze_img_neck=False,
    freeze_lifter=False,
    freeze_gaussian_head=False,
    freeze_encoder_except_densify=False,
    freeze_encoder_anchor_encoder=False,
    encoder=dict(
        densify_layer=dict(
            _delete_=True,
            type="AdaptiveAllocationV8",
            feat_embed_dim=128,
            semantic_dim=17,
            hidden_dim=128,
            topk_count=2560,
            pc_range=_base_.pc_range,
            scale_range=_base_.scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            grid_size=_base_.grid_size,
            probe_count=128,
            selection_mode="learned",
            selection_warmup_steps=1000,
            selection_transition_steps=1000,
        ),
    ),
)
loss = dict(
    type="MultiLoss",
    loss_cfgs=[
        _base_.loss.loss_cfgs[0],
        dict(
            type="AdaptiveAllocationV8Loss",
            weight=0.1,
            teacher_radius=2,
            class_weight=_base_.loss.loss_cfgs[0].manual_class_weight,
        ),
    ],
)
loss_input_convertion = dict(allocation_aux="allocation_aux")
train_dataset_config = dict(num_lidar_history=9)
val_dataset_config = dict(num_lidar_history=9)
