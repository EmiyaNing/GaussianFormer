"""Trainable semantic routing with a fixed native future reference.

Perception is shared and frozen. The original six forecast modules plus the
query gate form the trainable candidate. A separate frozen native rollout
supplies both routing labels and all unselected child attributes. Candidate
set/trajectory losses retain their native definition; voxel losses render the
routed Gaussian packs. Routed states never feed either rollout.
"""

from contextlib import nullcontext

import torch
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from .pmg_query_update_gate_world import PMGQueryUpdateGateWorld
from .pmg_semantic_gaussian_route_world import (
    PMGSemanticGaussianRouteWorld, semantic_mask, select_children,
)
from .sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGSemanticRouteTrainWorld(PMGSemanticGaussianRouteWorld):
    """Use the same selected future Gaussians for training and evaluation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture_candidate_gradients = False
        self.candidate_gaussian_gradient_packs = []
        self.last_reference_requires_grad = []
        self.capture_reference_grids = False
        self.last_reference_grids = []

    def train(self, mode=True):
        # Match G1 adaptation: deterministic frozen perception and native
        # dropout/BN, with the outer training flag retaining losses/range masks.
        super().train(mode)
        for child in self.children():
            child.eval()
        return self

    def set_epoch(self, epoch):
        # Parent set_epoch(force=True) reassigns query timestamps. Preserve the
        # source allocation and statistics throughout reference adaptation.
        head = self.pts_bbox_head
        stamps = head.ind_stamps_all
        statistics = head.num_stamps_all.clone()
        super().set_epoch(epoch)
        head.num_stamps_all.copy_(statistics)
        if stamps is not None:
            head.ind_stamps_all = stamps
            head.reset_mask()

    def _forecast(self, outputs, ego_feature, metas):
        with self.reference_modules(), torch.no_grad():
            reference = SparseOpusFullHorizonBaseline._forecast(self, outputs, ego_feature, metas)
        candidate = PMGQueryUpdateGateWorld._forecast(self, outputs, ego_feature, metas)
        if len(reference[0]) != 6 or len(candidate[0]) != 6:
            raise ValueError('All six steps must be active from the first epoch')
        if self.capture_reference_grids:
            self.last_reference_pred_traj = torch.cat(reference[3],1)
        self._reference_forecast = reference
        self._candidate_forecast = candidate
        self._route_masks = [semantic_mask(logits, self.route_class_ids) for logits in reference[1]]
        # These are explicitly the candidate states for native set and
        # trajectory supervision, not a surrogate feature for mixed children.
        return candidate

    def _render_occupancy(self, outputs, forecast_points, forecast_classes,
                          forecast_features, current_queries, sampled_xyz,
                          batch_size, render_all=True):
        gaussian = outputs['representation'][-1]['gaussian']
        current = type(gaussian)(*(value[:, current_queries] if value is not None else None for value in gaussian))
        logits_by_horizon = {0:self.gaussian_head.render_state(current, sampled_xyz)}
        grids = [logits_by_horizon[0].argmax(1).view(batch_size,200,200,16)]
        self.last_reference_grids = [grids[0]] if self.capture_reference_grids else []
        encoder = self.pts_bbox_head.transformer.encoder
        self.last_route_audit = []
        self.candidate_gaussian_gradient_packs = []
        self.last_reference_requires_grad = []
        try:
            for index, mask in enumerate(self._route_masks):
                horizon = index+1
                supervised = self.training and horizon in self.gaussian_loss_horizons
                if not render_all and not supervised:
                    continue
                with nullcontext() if supervised else torch.no_grad():
                    with torch.no_grad():
                        path = self._reference_forecast
                        points,classes,features = path[0][index],path[1][index],path[2][index]
                        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                        reference = encoder._make_gaussian(features,points,classes,valid,len(encoder.layers)-1)
                    path = self._candidate_forecast
                    points,classes,features = path[0][index],path[1][index],path[2][index]
                    valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
                    candidate = encoder._make_gaussian(features,points,classes,valid,len(encoder.layers)-1)
                    selected = select_children(reference,candidate,mask)
                    if self.capture_candidate_gradients:
                        for value in candidate:
                            if value is not None and value.requires_grad:
                                value.retain_grad()
                        self.candidate_gaussian_gradient_packs.append(candidate)
                        self.last_reference_requires_grad.append(any(value.requires_grad for value in reference if value is not None))
                    if getattr(self,'audit_route_fields',False):
                        self.last_route_audit.append(dict(
                            retained_reference_exact=all(torch.equal(x[~mask],r[~mask]) for x,r in zip(selected,reference) if x is not None),
                            selected_candidate_exact=all(torch.equal(x[mask],c[mask]) for x,c in zip(selected,candidate) if x is not None)))
                    self._record_route(horizon,reference,candidate,mask)
                    if self.capture_reference_grids:
                        with torch.no_grad():
                            reference_logits = self.gaussian_head.render_state(reference,sampled_xyz)
                            self.last_reference_grids.append(reference_logits.argmax(1).view(batch_size,200,200,16))
                    logits = self.gaussian_head.render_state(selected,sampled_xyz)
                    if supervised:
                        logits_by_horizon[horizon] = logits
                    if render_all:
                        grids.append(logits.argmax(1).view(batch_size,200,200,16))
        finally:
            # Do not retain a previous minibatch's candidate autograd graph.
            del self._reference_forecast
            del self._candidate_forecast
        return logits_by_horizon,grids
