"""Native A0 architecture with all six forecast steps trained from epoch one."""

from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from .sparse_opus_gausisan_world import SparseOpusGaussianWorld


@MODELS.register_module()
@SEGMENTORS.register_module()
class SparseOpusFullHorizonBaseline(SparseOpusGaussianWorld):
    """Change rollout scheduling only; retain A0 parameters and propagation.

    The inherited rollout computes its length from ``curr_epoch``. Temporarily
    advance that scheduling input during training, then restore the actual
    epoch. This enables six steps without copying or changing native rollout
    code, timestamp assignment, point updates, or checkpoint parameter names.
    """

    def __init__(self, **kwargs):
        if kwargs.get("finetune_epoch", 0) != 0:
            raise ValueError("the full-horizon baseline requires finetune_epoch=0")
        super().__init__(**kwargs)

    def _forecast(self, outputs, ego_feature, metas):
        actual_epoch = self.curr_epoch
        if self.training:
            self.curr_epoch = max(actual_epoch, self.num_fu_frames - 1)
        try:
            return super()._forecast(outputs, ego_feature, metas)
        finally:
            self.curr_epoch = actual_epoch
