"""Retain final classification, voxel loss, and every early-stage anchor."""
_base_ = ['./base.py']
loss = _base_.loss
loss.loss_cfgs[0].final_geometry_weight = 0.0
