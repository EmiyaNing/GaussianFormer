"""Required S1-C control: deterministic 32-sample, horizon-1 overfit."""
_base_ = ['./sparse_latent_world_stage1_r50_704x256_6f.py']

train_dataset_config = dict(
    max_samples=32, single_future_training=True,
    forecast_curriculum_start_epoch=0,
    forecast_curriculum_epochs_per_horizon=1000)
val_dataset_config = dict(max_samples=8)
train_loader = dict(batch_size=1, num_workers=2, shuffle=False,
                    persistent_workers=True, prefetch_factor=1)
val_loader = dict(batch_size=1, num_workers=1, persistent_workers=True,
                  prefetch_factor=1)
optimizer = dict(
    _delete_=True,
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=.01))
max_epochs = 20
eval_every_epochs = 20
warmup_iters = 64
