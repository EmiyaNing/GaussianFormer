import torch.nn as nn

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class DAOccLoss(nn.Module):
    """Expose the loss already computed by the coupled DAOcc heads."""

    def forward(self, inputs):
        return inputs['loss_total']
