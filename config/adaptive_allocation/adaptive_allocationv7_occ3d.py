"""Terminal shared growth on the FP32 React v5 execution path.

Warm-start with a checkpoint matching this Occ3D backbone/layout. The experiment
tool records an explicit old-checkpoint remapping; no unrelated RayDN weights
are loaded by default. Training here freezes the trunk to isolate allocation.
"""

_base_ = ["../img_point_lite_react/img_voxel_react_occ3d_fast_v5_fp32.py"]
load_from = "work_dirs/adaptive_allocation_v7_pilot/warmstart_occ3d.pth"
model = dict(
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_lifter=True,
    frozen_modules_eval=True,
    freeze_gaussian_head=True,
    freeze_encoder_except_densify=True,
    freeze_encoder_anchor_encoder=True,
    encoder=dict(
        densify_layer=dict(
            _delete_=True,
            type="AdaptiveAllocationV7",
            feat_embed_dim=128,
            semantic_dim=17,
            hidden_dim=128,
            pc_range=_base_.pc_range,
            scale_range=_base_.scale_range,
            grid_size=_base_.grid_size,
            allocation_ratio=0.15,
            probe_count=128,
            exploration_fraction=0.1,
            exploration_steps=1000,
            update_parent=True,
            child_opacity_init=0.02,
        )
    ),
)
loss = dict(
    type="MultiLoss",
    loss_cfgs=[
        _base_.loss.loss_cfgs[0],
        dict(
            type="AdaptiveAllocationV7Loss",
            weight=0.1,
            teacher_radius=2,
            stability_weight=0.05,
            stability_steps=200,
            class_weight=_base_.loss.loss_cfgs[0].manual_class_weight,
        ),
    ],
)
loss_input_convertion = dict(allocation_aux="allocation_aux")
train_dataset_config = dict(num_lidar_history=9)
val_dataset_config = dict(num_lidar_history=9)
