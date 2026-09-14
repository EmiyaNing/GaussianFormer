"""A1: deterministic 32-sample t+1 forecasting overfit test."""
_base_ = ['./forecasting_sparseworld_opusv2_tiny_r50_704x256_6f.py']

model = dict(
    init_checkpoint=None,
    current_occupancy_checkpoint='out/work_dirs/opusv2_t/official_mapped.pth',
    freeze_img_backbone=True,
    freeze_img_neck=True,
    freeze_occupancy_head=True,
    forecast_curriculum_start_epoch=0,
    forecast_curriculum_epochs_per_horizon=1000,
    train_future_horizons_per_step=1)

train_dataset_config = dict(max_samples=32)
val_dataset_config = dict(max_samples=1)
train_loader = dict(batch_size=1, num_workers=2, shuffle=False,
                    persistent_workers=True, prefetch_factor=1)
val_loader = dict(batch_size=1, num_workers=1, persistent_workers=True,
                  prefetch_factor=1)

loss = dict(current_weight=0., future_weight=1.,
            latent_alignment_weight=30., pyramid_alignment_weight=0.)
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.),
        'img_neck': dict(lr_mult=0.),
        'head': dict(lr_mult=0.),
        'sampling_offset': dict(lr_mult=.1)}))
grad_max_norm = 5
max_epochs = 12
eval_every_epochs = 12
warmup_iters = 64
min_lr_ratio = .01
print_freq = 16
epoch_start_sleep = 0
