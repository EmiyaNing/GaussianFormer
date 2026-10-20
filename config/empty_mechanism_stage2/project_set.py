"""Five-epoch mini candidate; identical forward model and original training loop."""
_base_ = ["../center_support_contribution1/a0_original.py"]
custom_imports = dict(imports=["loss.amplitude_projected_set_loss"], allow_failed_imports=False)
loss = _base_.loss
loss.loss_cfgs[0].type = "AmplitudeProjectedSetLoss"
