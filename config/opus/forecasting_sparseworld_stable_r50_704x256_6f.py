"""Recommended full-data recipe derived from the A0--A3 diagnostics.

The six horizons are introduced every two epochs.  With 36 epochs, horizon 6
enters at epoch 10 while the cosine LR is still high and then receives 26
epochs of optimization.  One horizon per iteration keeps the 24 GB memory
footprint of the original forecasting config.
"""
_base_ = ['./forecasting_sparseworld_opusv2_tiny_r50_704x256_6f.py']

enable_model_set_epoch = True
enable_dataset_set_epoch = True
model = dict(
    init_checkpoint=None,
    current_occupancy_checkpoint='out/work_dirs/opusv2_t/official_mapped.pth',
    freeze_img_backbone=True,
    freeze_img_neck=False,
    freeze_occupancy_head=False,
    forecast_curriculum_start_epoch=0,
    forecast_curriculum_epochs_per_horizon=2,
    observed_backbone_chunk_frames=3,
    train_future_horizons_per_step=1)

# Training loads only the selected future frame/occupancy target. Validation
# retains all six horizons for comparable metrics.
train_dataset_config = dict(
    single_future_training=True,
    forecast_curriculum_start_epoch=0,
    forecast_curriculum_epochs_per_horizon=2)

# A3 showed that this term prevents the learned FPN from drifting without
# degrading future occupancy loss.  Keep the stronger latent target as the
# primary forecasting supervision and use pyramid alignment as a regularizer.
loss = dict(current_weight=1., future_weight=1.,
            latent_alignment_weight=30., pyramid_alignment_weight=3.)

# Preserve the pretrained occupancy stack with a small LR; let the new
# forecasting/projection/decoder modules learn 100x faster.
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
amp_loss_scale = 128.
max_epochs = 36
eval_every_epochs = 6
warmup_iters = 500
min_lr_ratio = .05
epoch_start_sleep = 0
