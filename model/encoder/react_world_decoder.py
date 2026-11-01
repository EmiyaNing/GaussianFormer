"""Autoregressive Gaussian forecasting and planning from one perceived scene.

All positions and plans use the current ego frame. ``vel_branch`` predicts
metres per step, matching SparseWorld's displacement convention, not m/s.
The Gaussian count, scales, rotations and opacities stay fixed during rollout.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.registry import MODELS


def prediction_branch(width, output_dim):
    """Use a small, readable MLP for each independently interpreted output."""
    return nn.Sequential(
        nn.Linear(width, width), nn.ReLU(),
        nn.Linear(width, width), nn.ReLU(), nn.Linear(width, output_dim))


def inverse_softplus(value):
    """Convert React's positive semantic contributions into stable logits."""
    value = value.clamp_min(1e-8)
    return value + torch.log(-torch.expm1(-value))


@MODELS.register_module()
class ReactWorldDecoder(nn.Module):
    def __init__(self, embed_dims=128, ego_state_dim=21, num_heads=4,
                 future_steps=6, step_seconds=.5, reg_limit=.2,
                 displacement_limit=10., dynamic_classes=tuple(range(2, 11)),
                 pc_range=(-40., -40., -1., 40., 40., 5.4)):
        super().__init__()
        if future_steps < 1 or step_seconds <= 0:
            raise ValueError('future_steps and step_seconds must be positive')
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads')
        self.embed_dims = embed_dims
        self.ego_state_dim = ego_state_dim
        self.future_steps = future_steps
        self.step_seconds = step_seconds
        self.reg_limit = reg_limit
        self.displacement_limit = displacement_limit
        self.register_buffer('pc_min', torch.tensor(pc_range[:3]), persistent=False)
        extent = [upper - lower for lower, upper in zip(pc_range[:3], pc_range[3:])]
        self.register_buffer('pc_extent', torch.tensor(extent), persistent=False)
        self.register_buffer('dynamic_classes', torch.tensor(dynamic_classes), persistent=False)

        self.ego_encoder = prediction_branch(embed_dims, embed_dims)
        self.ego_input = nn.Linear(ego_state_dim, embed_dims)
        # xyz, elapsed seconds and the current planned xy waypoint.
        self.position_encoder = nn.Sequential(
            nn.Linear(6, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(),
            nn.Linear(embed_dims, embed_dims))
        self.scene_norm = nn.LayerNorm(embed_dims)
        self.ego_cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=0., batch_first=True)
        self.ego_update = nn.GRUCell(embed_dims, embed_dims)
        self.scene_update = nn.Sequential(
            nn.LayerNorm(embed_dims), nn.Linear(embed_dims, embed_dims * 2),
            nn.GELU(), nn.Linear(embed_dims * 2, embed_dims))
        self.reg_branch = prediction_branch(embed_dims, 3)
        self.vel_branch = prediction_branch(embed_dims, 2)
        self.cls_branch = prediction_branch(embed_dims, 17)
        self.traj_head = prediction_branch(embed_dims, 2)
        # A zero residual preserves the reliable perception at initialization.
        for branch in (self.reg_branch, self.vel_branch, self.cls_branch, self.traj_head):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def _position_features(self, means, waypoint, elapsed):
        normalized = (means - self.pc_min) / self.pc_extent
        time = means.new_full((*means.shape[:2], 1), elapsed)
        plan = (waypoint / self.pc_extent[:2])[:, None].expand(-1, means.shape[1], -1)
        return self.position_encoder(torch.cat((normalized, time, plan), dim=-1))

    def forward(self, gaussians, features, ego_states):
        if features.ndim != 3 or features.shape[:2] != gaussians.means.shape[:2]:
            raise ValueError('features must have one vector per Gaussian')
        if features.shape[-1] != self.embed_dims or features.shape[1] == 0:
            raise ValueError('A nonempty scene with matching feature width is required')
        if ego_states.shape != (features.shape[0], self.ego_state_dim):
            raise ValueError(f'ego_states must have shape [B, {self.ego_state_dim}]')
        if gaussians.semantics.shape[-1] != 17:
            raise ValueError('ReactWorldDecoder requires 17 Gaussian semantic channels')

        ego = self.ego_encoder(self.ego_input(ego_states.to(features)))
        waypoint = features.new_zeros(features.shape[0], 2)
        semantic_logits = inverse_softplus(gaussians.semantics)
        states, plans, regression, displacements = [], [], [], []
        state = gaussians

        for step in range(self.future_steps):
            elapsed = (step + 1) * self.step_seconds
            position = self._position_features(state.means, waypoint, elapsed)
            scene = self.scene_norm(features + position)
            context, _ = self.ego_cross_attn(
                ego[:, None], scene, scene, need_weights=False)
            ego = self.ego_update(context[:, 0], ego)
            waypoint = waypoint + self.traj_head(ego)

            position = self._position_features(state.means, waypoint, elapsed)
            features = features + self.scene_update(features + position + ego[:, None])
            offset = self.reg_limit * torch.tanh(self.reg_branch(features))
            displacement = self.displacement_limit * torch.tanh(self.vel_branch(features))
            semantic_logits = semantic_logits + self.cls_branch(features)
            semantics = F.softplus(semantic_logits)
            labels = semantics.argmax(dim=-1)
            moving = (labels[..., None] == self.dynamic_classes).any(dim=-1)
            planar_motion = F.pad(displacement * moving[..., None], (0, 1))
            state = state._replace(
                means=state.means + offset + planar_motion, semantics=semantics)
            states.append(state)
            plans.append(waypoint)
            regression.append(offset)
            displacements.append(displacement)

        return dict(future_gaussians=states, pred_traj=torch.stack(plans, dim=1),
                    forecast_reg_offsets=regression, forecast_displacements=displacements,
                    horizons_seconds=[self.step_seconds * (i + 1)
                                     for i in range(self.future_steps)])
