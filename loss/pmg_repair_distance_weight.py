"""Stop the future focal distance-weight position gradient without changing values."""

from . import OPENOCC_LOSS
from .sparseworld_strict_loss import SparseWorldStrictLoss


@OPENOCC_LOSS.register_module()
class PMGRepairStrictLoss(SparseWorldStrictLoss):
    """Retain the original set objective; detach only future focal weights.

    The native forward supplies prediction_mask exclusively to future set
    terms. All weight factors are constants apart from inverse point distance,
    so detaching weights removes precisely that route to position. Logit
    gradients and the point regression graph are preserved. Enforce complete
    future masks so a caller cannot silently omit this control.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._future_focal = False

    def forward(self, inputs):
        if not inputs["sparseworld_pretrain"]:
            if len(inputs.get("future_pred_masks", ())) != len(inputs["future_pred_points"]):
                raise ValueError("M-repair requires an explicit mask for every future step")
        return super().forward(inputs)

    def _set_loss(self, points, logits, gt_points, gt_labels, gt_stamps=None,
                  stamp_statistics=None, query_count=None, temporal_reweight=False,
                  prediction_mask=None):
        previous = self._future_focal
        self._future_focal = prediction_mask is not None
        try:
            return super()._set_loss(points, logits, gt_points, gt_labels,
                gt_stamps=gt_stamps, stamp_statistics=stamp_statistics,
                query_count=query_count, temporal_reweight=temporal_reweight,
                prediction_mask=prediction_mask)
        finally:
            self._future_focal = previous

    def _focal(self, logits, labels, weights):
        if self._future_focal:
            weights = weights.detach()
        return super()._focal(logits, labels, weights)
