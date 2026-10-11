"""Approved S3 development replication; NOT independent-scene confirmation."""
_base_ = ['./future_distillation_training_protocol.py']
confirmation_root = 'out/future_distillation_causal/20261008_development_confirmation'
confirmation_train_samples = 4096
confirmation_val_samples = 1024
confirmation_epochs = 5
confirmation_seeds = [20260930, 20261001, 20261002]
confirmation_supervisions = ['task', 'teacher', 'shuffled']
confirmation_workers = 4
confirmation_checkpoint_every = 256
confirmation_log_every = 32
confirmation_bootstrap_repetitions = 1000
confirmation_alignment_samples = 128
# Check epoch 1 for protocol stability; epoch 5 remains the fixed endpoint.
confirmation_evaluation_epochs = [1, 5]
confirmation_development_only = True
