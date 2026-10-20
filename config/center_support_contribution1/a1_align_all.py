"""Separate mask control: align all stages, retaining all early loss terms."""
_base_ = ['./base.py']
loss = _base_.loss
loss.loss_cfgs[0].gt_domain = 'camera'
