"""One query-wise gate inside the native SparseWorld feature recurrence."""

import torch
import torch.nn.functional as F
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline


class QueryUpdateGate(nn.Module):
    """Scale an existing joint state increment; start exactly at unit scale."""

    def __init__(self, embed_dims=256, amplitude=0.5):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.amplitude = float(amplitude)
        if not 0 < self.amplitude < 1:
            raise ValueError("Gate amplitude must be strictly between zero and one")
        # Module construction must not advance the baseline's RNG sequence.
        with torch.random.fork_rng(devices=[]):
            self.linear = nn.Linear(2 * self.embed_dims + 1, 1)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, state, increment, normalized_time):
        time = state.new_full((*state.shape[:-1], 1), float(normalized_time))
        inputs = torch.cat((F.layer_norm(state, (self.embed_dims,)),
                            F.layer_norm(increment, (self.embed_dims,)), time), -1)
        return 1.0 + self.amplitude * self.linear(inputs).tanh()

    def update(self, state, fused, position, normalized_time, force_unit=False):
        # Preserve native FP32 addition order, even at zero initialization.
        native_next = state + fused + position
        increment = fused + position
        if force_unit:
            gate = state.new_ones((*state.shape[:-1], 1))
        else:
            gate = self(state, increment, normalized_time)
        return native_next + (gate - 1.0) * increment, gate


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGQueryUpdateGateWorld(SparseOpusFullHorizonBaseline):
    """Retain native branches, births and detaches; gate their shared input.

    This isolated full-horizon implementation copies the native rollout with
    one modified feature-update expression. No GT region enters the gate.
    """

    def __init__(self, query_gate_amplitude=0.5, **kwargs):
        super().__init__(**kwargs)
        self.query_update_gate = QueryUpdateGate(self.out_dim, query_gate_amplitude)
        self.force_unit_gate = False
        self.last_gate_values = []
        self.collect_gate_statistics = False
        self._gate_statistics = {}

    def reset_gate_statistics(self, enabled=True):
        self._gate_statistics = {}
        self.collect_gate_statistics = bool(enabled)

    @torch.no_grad()
    def _record_gate(self, step, gate):
        if not self.collect_gate_statistics:
            return
        values = gate.detach().double().flatten()
        current = torch.stack((values.new_tensor(values.numel()), values.sum(),
                               values.square().sum(), values.min(), values.max(),
                               (values < .95).sum(), (values > 1.05).sum()))
        if step not in self._gate_statistics:
            self._gate_statistics[step] = current
        else:
            record = self._gate_statistics[step]
            record[:3] += current[:3]
            record[3] = torch.minimum(record[3], current[3])
            record[4] = torch.maximum(record[4], current[4])
            record[5:] += current[5:]

    @torch.no_grad()
    def gate_statistics(self):
        rows = []
        for step, value in sorted(self._gate_statistics.items()):
            count, total, squared, lower, upper, smaller, larger = value.cpu().tolist()
            mean = total / count
            rows.append(dict(seconds=step * .5, query_evaluations=int(count),
                mean=mean, std=max(0., squared / count - mean * mean) ** .5,
                minimum=lower, maximum=upper, fraction_below_095=smaller / count,
                fraction_above_105=larger / count))
        return dict(scope="All query slots, not GT instances or motion labels", rows=rows)

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
        self.last_gate_values = []
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
            features, gate = self.query_update_gate.update(
                features, fused, position, (interval + 1) / horizon, self.force_unit_gate)
            self.last_gate_values.append(gate.detach())
            self._record_gate(interval + 1, gate)
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
