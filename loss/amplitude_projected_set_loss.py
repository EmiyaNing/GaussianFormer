"""Training-only projection of the final Set-classification gradient."""
import torch
from . import OPENOCC_LOSS
from .center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss


class _ClassificationAmplitudeProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        ctx.save_for_backward(logits)
        return logits

    @staticmethod
    def backward(ctx, gradient):
        (logits,) = ctx.saved_tensors
        d = logits.double().sigmoid()
        g = gradient.double()
        denominator = d.square().sum(-1, keepdim=True)
        if torch.any((denominator <= 1e-30) & (g.square().sum(-1, keepdim=True) > 0)):
            raise FloatingPointError("Nonzero classification gradient at a vanishing amplitude derivative")
        result = g - d * (d*g).sum(-1, keepdim=True) / denominator.clamp_min(1e-30)
        return result.to(gradient)


@OPENOCC_LOSS.register_module()
class AmplitudeProjectedSetLoss(CenterSupportDiagnosticSetLoss):
    """Same loss value and assignment; project only the final classification path.

    Renderer logits retain their original graph. No occupancy gradient passes
    through this identity, and geometry uses the original prediction points.
    Output orthogonality does not imply amplitude preservation in shared params.
    """
    def components(self, inputs):
        modified = dict(inputs)
        logits = list(inputs["opus_pred_logits"])
        logits[-1] = _ClassificationAmplitudeProjection.apply(logits[-1])
        modified["opus_pred_logits"] = logits
        return super().components(modified)
