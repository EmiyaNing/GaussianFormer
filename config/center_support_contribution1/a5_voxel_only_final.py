"""Only the final set terms are removed; early point supervision remains."""
_base_ = ['./base.py']
loss = _base_.loss
loss.loss_cfgs[0].final_geometry_weight = 0.0
loss.loss_cfgs[0].final_classification_weight = 0.0
