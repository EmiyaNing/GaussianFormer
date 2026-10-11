"""Experiment-local S0 continuation: factorial task-training controls."""
_base_ = ['./future_distillation_causal.py']
protocol_root = 'out/future_distillation_causal/20261001_training_protocol'
protocol_train_samples = 128
protocol_val_samples = 64
protocol_updates = 256
protocol_eval_steps = [32, 128, 256]
protocol_seeds = [20260930]
# Training and feature audits use zero workers in this experiment's loader;
# native occupancy evaluation inherits full_tune_workers=4.
protocol_workers = 0
# Keep the first 256 optimizer learning rates identical to the earlier run;
# do not compress its five-epoch cosine schedule into this short screen.
protocol_schedule_steps = 4096 * 5
protocol_arms = [
    'full_bn_train', 'full_bn_fixed',
    'future_bn_train', 'future_bn_fixed',
    'full_bn_fixed_detached_geom', 'future_bn_fixed_detached_geom',
]
