"""OPUS-V1-T teacher aligned to SparseWorld's 720-query/48-point state.

The model follows the existing strict 8-frame OPUS-V1-T recipe, while using
SparseWorld's final sparse-state cardinality and a genuine full-volume NoMask
training objective.  It is intended to become the frozen teacher checkpoint
for sparse latent world stage one after passing current-occupancy validation.
"""
_base_ = ['./opusv1_t_r50_704x256_8f_occ3d.py']

num_queries = 720
num_refines = [1, 4, 16, 24, 32, 48]

model = dict(
    lifter=dict(num_queries=num_queries),
    encoder=dict(num_refines=num_refines),
    head=dict(point_multipliers=num_refines))

# All non-empty voxels are already geometric targets under official_v1 mode.
# This switch additionally prevents camera visibility from changing their
# hard-target weights, making the objective genuinely camera-mask independent.
loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[dict(
        type='OPUSSetLoss',
        stage_weights=[1.0] * 6,
        loss_mode='official_v1',
        lambda_cls=2.0,
        focal_gamma=2.0,
        focal_alpha=0.25,
        smooth_l1_beta=0.2,
        lambda_pts=0.5,
        empty_dist_thr=0.2,
        empty_weight=5.0,
        rare_classes=[0, 2, 5, 8],
        rare_weight=10.0,
        class_weights=[10, 5, 10, 5, 5, 10, 10, 5, 10,
                       5, 5, 1, 5, 1, 1, 2, 1],
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        chunk_size=1024,
        use_camera_mask_weights=False)])

# Full-volume is the primary teacher-selection protocol. Camera-visible
# metrics can still be run as an additional evaluation by overriding this.
occ3d_eval_mask = 'none'
max_epochs = 20
eval_every_epochs = 1

# The 720x48 full-volume objective overflows inside the FP16 backward path
# even after GradScaler backs off below 1. FP32 backward is stable at batch
# size 2 on the target 24 GiB GPU, including the EMA shadow parameters.
amp = False
fail_on_nonfinite_grad = True

# Keep a teacher-quality shadow model. EMA is updated only after successful
# optimizer steps, persisted in every checkpoint, and used for validation.
model_ema = dict(
    enabled=True,
    decay=0.9998,
    update_buffers=True,
    eval_with_ema=True)

# ``eval.py`` prefers ema_state_dict when it is present and falls back to the
# regular state_dict for the official initialization checkpoint.
load_ema_checkpoint = True
