"""Historical-domain anchor, using the parity-tested diagnostic loss."""
_base_ = ['./base.py']
loss = _base_.loss
loss.loss_cfgs[0].final_gt_domain = 'all'
