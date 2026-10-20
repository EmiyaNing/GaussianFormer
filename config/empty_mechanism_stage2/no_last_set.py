"""Control: zero only the last-stage classification multiplier; retain geometry."""
_base_ = ["../center_support_contribution1/a0_original.py"]
loss = _base_.loss
loss.loss_cfgs[0].final_classification_weight = 0.0
