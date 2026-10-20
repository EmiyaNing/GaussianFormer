"""Retain final geometry, voxel loss, and every early-stage anchor."""
_base_ = ['./base.py']
loss = _base_.loss
loss.loss_cfgs[0].final_classification_weight = 0.0
