"""Isolated staged causal diagnostics; does not change existing trainers."""
_base_ = ['./future_predictable_full_finetune.py']
causal_root = 'out/future_distillation_causal/20260930'
causal_reference_root = 'out/future2current/predictable_full_finetune_4096x1024'
causal_train_samples = 256
causal_val_samples = 128
causal_audit_samples = 64
causal_gradient_samples = 16
causal_updates = 512
causal_seeds = [20260930, 20261001, 20261002]
causal_feature_weight = .1
causal_gradient_cap = .1
causal_lr = 2e-5
causal_workers = 0
causal_mix = [0., .1, .25, .5, 1.]
