"""Remove OPUS set supervision at every stage; retain the matched voxel loss.

The reference is contribution-1 a0_original, not a5 (which retains early set
losses). Model, pretraining, learned empty bias and optimizer remain identical.
"""
_base_ = ['../center_support_contribution1/a0_original.py']

loss = _base_.loss
loss.loss_cfgs = [settings for settings in loss.loss_cfgs
                 if settings.type == 'OccupancyLoss']
