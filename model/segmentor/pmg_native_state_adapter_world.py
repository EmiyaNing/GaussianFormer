"""A bounded vector residual inside the otherwise native forecast recurrence.

The dedicated experiment runner freezes native parameters. This module does
not freeze state evolution: adapted features drive all native output branches
and the next forecast step. Frozen weights therefore do not imply unchanged
future geometry, semantics, Gaussian attributes, or voxel predictions.
"""

import torch
import torch.nn.functional as F
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline


class NativeStateAdapter(nn.Module):
    """Add a query-wise vector innovation bounded relative to the native input.

    The 513 -> 32 -> 256 default MLP has 24,896 parameters. Its zero output
    projection recovers native recurrence exactly at initialization. The RMS
    scale is detached, so its gradient cannot reward changing the reference
    increment's norm merely to amplify a correction. The input features remain
    differentiable; this does not detach the main recurrent feature stream.
    """

    def __init__(self, embed_dims=256, rank=32, strength=0.1):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.rank = int(rank)
        self.strength = float(strength)
        if self.embed_dims <= 0 or self.rank <= 0:
            raise ValueError("Adapter dimensions must be positive")
        if not 0 < self.strength < 1:
            raise ValueError("Adapter strength must be strictly between zero and one")
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(2 * self.embed_dims + 1, self.rank)
            self.up = nn.Linear(self.rank, self.embed_dims)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        self.last_diagnostics = {}

    def forward(self, state, increment, normalized_time):
        time = state.new_full((*state.shape[:-1], 1), float(normalized_time))
        inputs = torch.cat((F.layer_norm(state, (self.embed_dims,)),
                            F.layer_norm(increment, (self.embed_dims,)), time), -1)
        raw = self.up(F.gelu(self.down(inputs)))
        # At an exactly zero native increment the residual is exactly zero.
        scale = increment.detach().square().mean(-1, keepdim=True).sqrt()
        return self.strength * scale * raw.tanh()

    def update(self, state, fused, position, normalized_time, force_zero=False):
        # Keep the native floating-point addition order, including zero init.
        native_next = state + fused + position
        increment = fused + position
        delta = (torch.zeros_like(state) if force_zero else
                 self(state, increment, normalized_time))
        with torch.no_grad():
            norm_delta = delta.norm(dim=-1, keepdim=True)
            norm_increment = increment.norm(dim=-1, keepdim=True)
            ratio = norm_delta / norm_increment.clamp_min(1e-12)
            valid_cosine = (norm_delta > 0) & (norm_increment > 0)
            cosine = F.cosine_similarity(delta, increment, dim=-1).unsqueeze(-1)
            self.last_diagnostics = dict(
                delta_norm=norm_delta, increment_norm=norm_increment,
                cosine=cosine, cosine_valid=valid_cosine)
        return native_next + delta, ratio


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGNativeStateAdapterWorld(SparseOpusFullHorizonBaseline):
    """Retain native branches, births and detaches with one new state residual.

    The copied forecast loop differs only at the shared feature update and
    its diagnostics. No GT region, object speed or future image enters the
    adapter. A bound on one residual does not bound accumulated output drift.
    """

    def __init__(self, state_adapter_rank=32, state_adapter_strength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.native_state_adapter = NativeStateAdapter(
            self.out_dim, state_adapter_rank, state_adapter_strength)
        self.force_zero_adapter = False
        self.last_adapter_statistics = []

    @torch.no_grad()
    def _record_adapter(self, step, ratio):
        values = self.native_state_adapter.last_diagnostics
        valid = values["cosine_valid"]
        count = valid.sum()
        # One small transfer per step keeps the public diagnostics JSON-ready.
        summaries = torch.stack((
            ratio.mean(), ratio.max(), values["delta_norm"].mean(),
            values["increment_norm"].mean(), count.to(ratio),
            (values["cosine"] * valid).sum() / count.clamp_min(1))).cpu().tolist()
        self.last_adapter_statistics.append(dict(
            seconds=step * .5, query_evaluations=ratio.numel(),
            ratio_mean=summaries[0], ratio_max=summaries[1],
            delta_norm_mean=summaries[2], increment_norm_mean=summaries[3],
            cosine_valid_count=int(summaries[4]), cosine_mean=summaries[5]))

    def _forecast(self, outputs, ego_feature, metas):
        batch = ego_feature.shape[0]
        stamps = self.pts_bbox_head.ind_stamps_all
        all_features = outputs['query_feat']
        all_points = outputs['all_refine_pts'][-1]
        features = all_features[:, stamps == 0]
        points = all_points[:, stamps == 0].detach()
        timestamp = points.new_zeros(batch, self.num_query, self.num_refines, 1)
        # The parent full-horizon baseline also unfolds all six steps.
        horizon = self.num_fu_frames
        forecast_points, forecast_classes = [], []
        forecast_features, trajectories, range_masks = [], [], []
        self.last_adapter_statistics = []
        for interval in range(horizon):
            fused, _ = self.ego_cross_attn(
                ego_feature.new_ones(batch, 1, 3) * .5, ego_feature,
                points.detach(), features.detach())
            trajectories.append(self.traj_head(fused))
            future = stamps == interval + 1
            features = torch.cat((features, all_features[:, future]), dim=1)
            points = torch.cat((points, all_points[:, future]), dim=1).detach()
            timestamp = torch.cat((timestamp, points.new_full(
                (batch, int(future.sum()), self.num_refines, 1), .5)), dim=1)
            position = self.position_encoder(
                torch.cat((points, timestamp), dim=-1).flatten(2, 3))
            features, ratio = self.native_state_adapter.update(
                features, fused, position, (interval + 1) / horizon,
                force_zero=self.force_zero_adapter)
            self._record_adapter(interval + 1, ratio)
            offset = self.reg_branch(features).unflatten(-1, (-1, 3)) * .5
            classes = self.cls_branch(features).unflatten(-1, (-1, 17))
            velocity = self.vel_branch(features).unflatten(-1, (-1, 2))
            labels = classes.argmax(-1)
            moving = ((labels >= 2) & (labels <= 10)).unsqueeze(-1)
            offset = torch.cat((offset[..., :2] + velocity * moving,
                                offset[..., 2:]), dim=-1)
            points = self.refine_points(points, offset.flatten(2, 3))
            forecast_classes.append(classes)
            forecast_points.append(points)
            forecast_features.append(features)
            if self.training:
                ego2lidar = metas['ego2lidar'].to(points)
                world = _opus_decode_points(points, self.pc_range)
                lidar = world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
                lidar = lidar + ego2lidar[:, None, None, :3, 3]
                trajectory = metas['temporal_trajs'][:, interval].to(points)
                lidar = lidar + torch.cat((-trajectory, trajectory.new_zeros(
                    trajectory.shape[0], 1)), dim=-1)[:, None, None]
                lidar2ego = torch.linalg.inv(ego2lidar)
                shifted = lidar @ lidar2ego[:, None, :3, :3].transpose(-1, -2)
                shifted = shifted + lidar2ego[:, None, None, :3, 3]
                shifted = _opus_encode_points(shifted, self.pc_range)
                range_masks.append(shifted[..., 0] >= 0)
        return (forecast_points, forecast_classes, forecast_features,
                trajectories, range_masks)
