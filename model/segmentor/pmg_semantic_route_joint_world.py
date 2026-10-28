"""End-to-end semantic Gaussian routing with two trainable forecast paths.

The ungated path supplies non-movable-class children; the gated path supplies
the eight movable classes. Both paths and their shared perception are trained.
Routing labels are detached predictions, but neither Gaussian path is detached.
"""

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from .pmg_query_update_gate_world import PMGQueryUpdateGateWorld
from .pmg_semantic_gaussian_route_world import (
    PMGSemanticGaussianRouteWorld,
    semantic_mask,
    select_children,
)
from .sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGSemanticRouteJointWorld(PMGSemanticGaussianRouteWorld):
    """Share perception/attributes/trajectory head and train both rollouts.

    ``reference_forecast_modules`` keeps the earlier checkpoint vocabulary,
    but now denotes a trainable ungated branch, not a frozen baseline. There
    is one supervised trajectory output, so its head is shared between paths.
    Native point/attention detaches remain part of the baseline recurrence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # An extra trajectory head would have no supervised output. Reuse the
        # candidate trajectory head instead of leaving an unused trainable copy.
        del self.reference_forecast_modules["traj_head"]
        self.audit_route_fields = False
        self.last_reference_requires_grad = []

    def train(self, mode=True):
        # Bypass the old adaptation-only override: decoder and forecast dropout
        # must follow the actual training flag. ResNet norm_eval remains its
        # standard baseline convention; its affine weights are still trainable.
        nn.Module.train(self, mode)
        return self

    @contextmanager
    def reference_modules(self):
        original = {
            name: getattr(self, name) for name in self.reference_forecast_modules
        }
        try:
            for name, module in self.reference_forecast_modules.items():
                setattr(self, name, module)
            yield
        finally:
            for name, module in original.items():
                setattr(self, name, module)

    def set_epoch(self, epoch):
        # Child-slot routing assumes a common timestamp allocation. Preserve
        # that allocation while training all associated features and weights.
        head = self.pts_bbox_head
        stamps = head.ind_stamps_all
        statistics = head.num_stamps_all.clone()
        super().set_epoch(epoch)
        head.num_stamps_all.copy_(statistics)
        if stamps is not None:
            head.ind_stamps_all = stamps
            head.reset_mask()

    def _forecast(self, outputs, ego_feature, metas):
        with self.reference_modules():
            reference = SparseOpusFullHorizonBaseline._forecast(
                self, outputs, ego_feature, metas
            )
        candidate = PMGQueryUpdateGateWorld._forecast(self, outputs, ego_feature, metas)
        if len(reference[0]) != 6 or len(candidate[0]) != 6:
            raise ValueError("All six forecast steps must be active")
        self._reference_forecast = reference
        self._candidate_forecast = candidate
        self._route_masks = [
            semantic_mask(logits, self.route_class_ids) for logits in reference[1]
        ]
        # Candidate point-set/trajectory supervision is unchanged. The ungated
        # branch receives voxel gradients through its non-selected children.
        return candidate

    def _render_occupancy(
        self,
        outputs,
        forecast_points,
        forecast_classes,
        forecast_features,
        current_queries,
        sampled_xyz,
        batch_size,
        render_all=True,
    ):
        gaussian = outputs["representation"][-1]["gaussian"]
        current = type(gaussian)(
            *(
                value[:, current_queries] if value is not None else None
                for value in gaussian
            )
        )
        logits_by_horizon = {0: self.gaussian_head.render_state(current, sampled_xyz)}
        grids = [logits_by_horizon[0].argmax(1).view(batch_size, 200, 200, 16)]
        encoder = self.pts_bbox_head.transformer.encoder
        self.last_route_audit = []
        self.last_reference_requires_grad = []
        try:
            for index, mask in enumerate(self._route_masks):
                horizon = index + 1
                supervised = self.training and horizon in self.gaussian_loss_horizons
                if not render_all and not supervised:
                    continue
                with nullcontext() if supervised else torch.no_grad():
                    packs = []
                    for path in (self._reference_forecast, self._candidate_forecast):
                        points = path[0][index]
                        classes = path[1][index]
                        features = path[2][index]
                        valid = torch.ones(
                            features.shape[:2], dtype=torch.bool, device=features.device
                        )
                        packs.append(
                            encoder._make_gaussian(
                                features,
                                points,
                                classes,
                                valid,
                                len(encoder.layers) - 1,
                            )
                        )
                    reference, candidate = packs
                    selected = select_children(reference, candidate, mask)
                    self.last_reference_requires_grad.append(
                        any(
                            value.requires_grad
                            for value in reference
                            if value is not None
                        )
                    )
                    if self.audit_route_fields:
                        self.last_route_audit.append(
                            dict(
                                retained_ungated_exact=all(
                                    torch.equal(x[~mask], r[~mask])
                                    for x, r in zip(selected, reference)
                                    if x is not None
                                ),
                                selected_gated_exact=all(
                                    torch.equal(x[mask], c[mask])
                                    for x, c in zip(selected, candidate)
                                    if x is not None
                                ),
                            )
                        )
                    self._record_route(horizon, reference, candidate, mask)
                    logits = self.gaussian_head.render_state(selected, sampled_xyz)
                    if supervised:
                        logits_by_horizon[horizon] = logits
                    if render_all:
                        grids.append(logits.argmax(1).view(batch_size, 200, 200, 16))
        finally:
            del self._reference_forecast
            del self._candidate_forecast
        return logits_by_horizon, grids
