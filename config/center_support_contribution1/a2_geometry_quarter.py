"""Weight-only geometry control after final-stage domain alignment."""
_base_ = ['./base.py']
loss = _base_.loss
loss.loss_cfgs[0].final_geometry_weight = 0.25
