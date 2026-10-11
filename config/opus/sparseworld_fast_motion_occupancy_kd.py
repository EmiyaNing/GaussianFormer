"""Paired 2048/512 fast-motion occupancy-only KD, native architecture."""
_base_=['./future2current_feature_identity.py']
fast_root='out/sparseworld_fast_motion_distillation/20261008_2048x512'
fast_samples_train=2048
fast_samples_val=512
fast_epochs=1
fast_seed=20261008
fast_workers=4
fast_kd_weight=.1
fast_lr=2e-5
fast_backbone_lr=2e-6
fast_warmup=128
fast_arms=['frozen_task','frozen_fast_kd','full_task','full_fast_kd']
