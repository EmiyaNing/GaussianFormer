"""Frozen A0 rollout with a shared, explicitly child-conditioned XY repair head."""

import torch
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .pmg_sparse_opus_world import PMGSparseOpusWorld


class ChildStateRepairHead(nn.Module):
    """Project each query once, then combine with each child's 29 local values.

    Local values are absolute normalized position (3), query-relative metric
    position (3), normalized positive semantics (17), log semantic mass (1),
    scales (3), ROI-masked opacity (1), and forecast time (1). No GT is read.
    """

    local_dimensions = 29

    def __init__(self, embed_dims=256, hidden_dims=64, radius_m=2.):
        super().__init__()
        self.radius_m = float(radius_m)
        if self.radius_m <= 0:
            raise ValueError("repair radius must be positive")
        self.context = nn.Sequential(nn.Linear(embed_dims * 2, hidden_dims), nn.ReLU())
        self.child = nn.Sequential(
            nn.Linear(hidden_dims + self.local_dimensions, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, 3),
        )
        nn.init.zeros_(self.child[-1].weight)
        nn.init.zeros_(self.child[-1].bias)

    def forward(self, future_features, current_features, local_state):
        context = self.context(torch.cat((future_features, current_features), -1))
        context = context.unsqueeze(-2).expand(*local_state.shape[:-1], context.shape[-1])
        raw = self.child(torch.cat((context, local_state), -1))
        xy = self.radius_m * torch.tanh(raw[..., :2] / self.radius_m)
        xy = xy / (xy.norm(dim=-1, keepdim=True) / self.radius_m).clamp_min(1.)
        xy = xy * raw[..., 2:].sigmoid()
        return torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGChildStateRepairWorld(PMGSparseOpusWorld):
    """Modify inherited output centers after the complete native rollout.

    Native features, semantics, shape, births and recurrent state are retained.
    The intended runner freezes native parameters and keeps their dropout off.
    """

    def __init__(self, correction_radius_m=2., repair_hidden_dims=64,
                 repair_scale_normalizer=.4, **kwargs):
        super().__init__(**kwargs)
        if self.geometry_mode != "baseline" or self.persistent_content:
            raise ValueError("M-repair requires native A0 propagation and content")
        self.correction_radius_m = float(correction_radius_m)
        self.repair_scale_normalizer = float(repair_scale_normalizer)
        if self.repair_scale_normalizer <= 0:
            raise ValueError("scale normalizer must be positive")
        with torch.random.fork_rng(devices=[]):
            self.motion_correction = ChildStateRepairHead(
                hidden_dims=repair_hidden_dims, radius_m=self.correction_radius_m)
        # Used only by engineering validation; no branch is selected using GT.
        self.repair_disabled = False
        self.repair_diagnostics = []

    def _child_state(self, points, features, classes, horizon):
        encoder = self.pts_bbox_head.transformer.encoder
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        with torch.no_grad():
            state = encoder._make_gaussian(features, points, classes, valid,
                                           len(encoder.layers) - 1)
            extent = self.pc_range[3:] - self.pc_range[:3]
            relative = (points - points.mean(-2, keepdim=True)) * extent
            mass = state.semantics.sum(-1, keepdim=True).clamp_min(1e-12)
            local = torch.cat((points * 2. - 1., relative / self.correction_radius_m,
                               state.semantics / mass, mass.log1p(),
                               state.scales / self.repair_scale_normalizer,
                               state.opacities, points.new_full((*points.shape[:-1], 1),
                                                               horizon / self.num_fu_frames)), -1)
        return local.detach()

    def _range_masks(self, points, metas):
        """Use the unchanged R2 corrected-center training-mask convention."""
        masks = []
        for interval, state in enumerate(points):
            world = _opus_decode_points(state, self.pc_range)
            ego2lidar = metas["ego2lidar"].to(state)
            lidar = world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
            lidar = lidar + ego2lidar[:, None, None, :3, 3]
            trajectory = metas["temporal_trajs"][:, interval].to(state)
            lidar = lidar + torch.cat((-trajectory, trajectory.new_zeros(state.shape[0], 1)),
                                      -1)[:, None, None]
            inverse = torch.linalg.inv(ego2lidar)
            shifted = lidar @ inverse[:, None, :3, :3].transpose(-1, -2)
            shifted = shifted + inverse[:, None, None, :3, 3]
            masks.append(_opus_encode_points(shifted, self.pc_range)[..., 0] >= 0)
        return masks

    def _forecast(self, outputs, ego_feature, metas):
        epoch = self.curr_epoch
        if self.training:
            self.curr_epoch = max(epoch, self.finetune_epoch + self.num_fu_frames - 1)
        try:
            points, classes, features, trajectories, masks = super()._forecast(
                outputs, ego_feature, metas)
        finally:
            self.curr_epoch = epoch
        current = self.pts_bbox_head.ind_stamps_all == 0
        current_features = outputs["query_feat"][:, current].detach()
        if current_features.shape[1] != self.num_query:
            raise ValueError("native current-query assignment changed")
        extent = self.pc_range[3:] - self.pc_range[:3]
        corrected = []
        self.repair_diagnostics = []
        for horizon, (original, feature, logits) in enumerate(zip(points, features, classes), 1):
            inherited = original[:, :self.num_query]
            if self.repair_disabled:
                delta = torch.zeros_like(inherited)
            else:
                local = self._child_state(inherited.detach(), feature[:, :self.num_query].detach(),
                                          logits[:, :self.num_query].detach(), horizon)
                delta = self.motion_correction(feature[:, :self.num_query].detach(),
                                                current_features, local)
            corrected.append(torch.cat((inherited + delta / extent,
                                         original[:, self.num_query:]), 1))
            with torch.no_grad():
                size = delta[..., :2].norm(dim=-1)
                self.repair_diagnostics.append(dict(seconds=horizon * .5,
                    mean_xy_m=float(size.mean()), max_xy_m=float(size.max()),
                    fraction_gt_1cm=float((size > .01).float().mean())))
        if self.training:
            masks = self._range_masks(corrected, metas)
        return corrected, classes, features, trajectories, masks
