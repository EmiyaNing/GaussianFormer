"""Isolated, cached Future2Current predictability experiment.

This config does not register a new segmentor or alter SparseWorld weights.
"""

_base_ = ['./future2current_feature_identity.py']

predictability_root = 'out/future2current/predictable_information_1024x256'
predictability_identity_root = 'out/future2current/feature_identity_2048x512'
predictability_pilot_root = 'out/future2current/pilot256x128'
predictability_train_samples = 1024
predictability_val_samples = 256
predictability_updates = 512
predictability_seeds = [20260930, 20261001, 20261002]
predictability_lr = 1e-4
predictability_feature_weight = 0.1
predictability_max_points = 4096
predictability_workers = 0
predictability_arms = ['task_only', 'teacher_all', 'teacher_shuffled']
