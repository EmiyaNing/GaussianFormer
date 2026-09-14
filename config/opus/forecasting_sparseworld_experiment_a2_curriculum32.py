"""A2: full warm-start, differential LR, and progressive horizons."""
_base_ = ['./forecasting_sparseworld_opusv2_tiny_r50_704x256_6f.py']

enable_model_set_epoch = True
model = dict(
    init_checkpoint=None,
    current_occupancy_checkpoint='out/work_dirs/opusv2_t/official_mapped.pth',
    freeze_img_backbone=True,
    freeze_img_neck=False,
    freeze_occupancy_head=False,
    forecast_curriculum_start_epoch=0,
    forecast_curriculum_epochs_per_horizon=2,
    train_future_horizons_per_step=1)

train_dataset_config = dict(max_samples=32)
val_dataset_config = dict(max_samples=8)

loss = dict(current_weight=1., future_weight=1.,
            latent_alignment_weight=30., pyramid_alignment_weight=0.)
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.),
        'scale_projections': dict(lr_mult=100.),
        'forecast': dict(lr_mult=100.),
        'pyramid_decoder': dict(lr_mult=100.),
        'sampling_offset': dict(lr_mult=.1)}))
grad_max_norm = 5
max_epochs = 12
eval_every_epochs = 12
warmup_iters = 64
min_lr_ratio = .01
print_freq = 16
epoch_start_sleep = 0
