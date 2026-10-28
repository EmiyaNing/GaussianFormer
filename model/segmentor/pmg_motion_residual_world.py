"""Output-side motion corrections; the original autoregressive rollout is intact."""

import torch
from torch import nn
from mmengine.registry import MODELS
from mmseg.models import SEGMENTORS

from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from .pmg_sparse_opus_world import PMGSparseOpusWorld


@MODELS.register_module()
@SEGMENTORS.register_module()
class PMGMotionResidualWorld(PMGSparseOpusWorld):
    """Add bounded, per-child metric corrections without writing them into rollout.

    New module construction preserves the caller's RNG. A zero output layer
    exactly recovers baseline coordinates at initialization. The sigmoid gate
    starts at 0.5 so the correction output layer can receive a gradient.
    """

    def __init__(self, correction_radius_m=1.0, **kwargs):
        super().__init__(**kwargs)
        if self.geometry_mode != "baseline" or self.persistent_content:
            raise ValueError(
                "the first motion correction must use native propagation and decoding"
            )
        self.correction_radius_m = float(correction_radius_m)
        if self.correction_radius_m <= 0:
            raise ValueError("positive correction radius required")
        with torch.random.fork_rng(devices=[]):
            self.motion_correction = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, self.num_refines * 4)
            )
            nn.init.zeros_(self.motion_correction[-1].weight)
            nn.init.zeros_(self.motion_correction[-1].bias)

    def correction(self, features):
        raw = self.motion_correction(features).unflatten(-1, (self.num_refines, 4))
        radius = self.correction_radius_m
        return raw[..., 3:].sigmoid() * radius * torch.tanh(raw[..., :3] / radius)

    def _forecast(self, outputs, ego_feature, metas):
        points, classes, features, trajectories, masks = super()._forecast(
            outputs, ego_feature, metas
        )
        count = self.num_query
        corrected = []
        for original, feature in zip(points, features):
            delta = self.correction(feature[:, :count])
            # Adding in normalized space avoids a decode/encode roundtrip
            # perturbing the native output when the residual is exactly zero.
            extent = self.pc_range[3:] - self.pc_range[:3]
            current = original[:, :count] + delta / extent
            new_points = torch.cat((current, original[:, count:]), 1)
            corrected.append(new_points)
        if self.training:
            # Match native future range-mask semantics using corrected centers.
            masks = []
            for interval, state in enumerate(corrected):
                world = _opus_decode_points(state, self.pc_range)
                ego2lidar = metas["ego2lidar"].to(state)
                lidar = world @ ego2lidar[:, None, :3, :3].transpose(-1, -2)
                lidar = lidar + ego2lidar[:, None, None, :3, 3]
                trajectory = metas["temporal_trajs"][:, interval].to(state)
                lidar = (
                    lidar
                    + torch.cat(
                        (-trajectory, trajectory.new_zeros(state.shape[0], 1)), -1
                    )[:, None, None]
                )
                inverse = torch.linalg.inv(ego2lidar)
                shifted = (
                    lidar @ inverse[:, None, :3, :3].transpose(-1, -2)
                    + inverse[:, None, None, :3, 3]
                )
                masks.append(_opus_encode_points(shifted, self.pc_range)[..., 0] >= 0)
        return corrected, classes, features, trajectories, masks
