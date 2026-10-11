"""Residual future-query alignment used by SparseWorld-FFA."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .sparse_latent_dynamics import transform_points


class _FFALayer(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=512,
                 dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims), nn.Dropout(dropout))

    def forward(self, value):
        attended = self.attention(
            value, value, value, need_weights=False)[0]
        value = self.norm1(value + attended)
        return self.norm2(value + self.ffn(value))


class FutureFeatureAlignment(nn.Module):
    """Predict metric geometry and latent-feature residuals for one horizon.

    The current points are first rigidly expressed in the target future ego
    frame.  All learned geometry is therefore a physically meaningful metric
    residual on top of the ego-motion warp.
    """

    def __init__(self, embed_dims=256, num_points=48, num_layers=3,
                 num_heads=8, ffn_dims=512, dropout=0.1,
                 target_horizon_index=6, alignment_dims=128,
                 max_translation_m=12.0, max_deformation_m=2.0,
                 checkpoint_layers=True, shape_start_epoch=8,
                 shape_ramp_epochs=4):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_points = int(num_points)
        self.target_horizon_index = int(target_horizon_index)
        self.max_translation_m = float(max_translation_m)
        self.max_deformation_m = float(max_deformation_m)
        self.checkpoint_layers = bool(checkpoint_layers)
        self.shape_start_epoch = int(shape_start_epoch)
        self.shape_ramp_epochs = max(1, int(shape_ramp_epochs))
        self.current_epoch = 0

        self.position_encoder = nn.Sequential(
            nn.Linear(6, embed_dims), nn.LayerNorm(embed_dims), nn.GELU(),
            nn.Linear(embed_dims, embed_dims))
        self.motion_encoder = nn.Sequential(
            nn.Linear(12, embed_dims), nn.LayerNorm(embed_dims), nn.GELU(),
            nn.Linear(embed_dims, embed_dims))
        self.horizon_embedding = nn.Parameter(torch.empty(1, embed_dims))
        self.layers = nn.ModuleList([
            _FFALayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(int(num_layers))])
        self.feature_residual = nn.Linear(embed_dims, embed_dims)
        self.center_residual = nn.Linear(embed_dims, 3)
        self.shape_residual = nn.Linear(embed_dims, num_points * 3)

        # The student learns a map into a fixed teacher space.  The frozen
        # orthogonal teacher projection cannot collapse during distillation.
        self.student_projection = nn.Linear(embed_dims, alignment_dims,
                                            bias=False)
        teacher_projection = torch.empty(alignment_dims, embed_dims)
        nn.init.orthogonal_(teacher_projection)
        self.register_buffer('teacher_projection', teacher_projection)

        nn.init.normal_(self.horizon_embedding, std=0.02)
        for layer in (self.feature_residual, self.center_residual,
                      self.shape_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    @property
    def shape_scale(self):
        progress = ((self.current_epoch - self.shape_start_epoch + 1) /
                    self.shape_ramp_epochs)
        return min(1.0, max(0.0, progress))

    def _run_layer(self, layer, tokens):
        if self.checkpoint_layers and self.training and tokens.requires_grad:
            return checkpoint(layer, tokens, use_reentrant=False)
        return layer(tokens)

    def forward(self, query_features, query_points_metric,
                current_to_future):
        if query_points_metric.shape[2] != self.num_points:
            raise ValueError(
                'FFA num_points does not match the SparseWorld state')
        if current_to_future.shape[-2:] != (4, 4):
            raise ValueError('current_to_future must end in [4,4]')

        # Distance-sensitive operations stay in FP32 under global AMP.
        warped = transform_points(
            query_points_metric.float(), current_to_future.float())
        center = warped.mean(2)
        spread = warped.std(2, unbiased=False)
        motion = current_to_future[:, :3].reshape(
            current_to_future.shape[0], 12).float()
        tokens = (query_features + self.position_encoder(
            torch.cat([center, spread], -1).to(query_features.dtype)) +
            self.motion_encoder(motion.to(query_features.dtype))[:, None] +
            self.horizon_embedding.to(query_features.dtype))
        for layer in self.layers:
            tokens = self._run_layer(layer, tokens)

        center_delta = torch.tanh(
            self.center_residual(tokens).float()) * self.max_translation_m
        shape_delta = torch.tanh(
            self.shape_residual(tokens).float().view(
                *tokens.shape[:2], self.num_points, 3))
        shape_delta = shape_delta * self.max_deformation_m * self.shape_scale
        feature_delta = self.feature_residual(tokens)
        predicted_features = query_features + feature_delta
        predicted_points = (warped + center_delta[:, :, None] + shape_delta)

        return dict(
            points_metric=predicted_points,
            query_features=predicted_features,
            feature_delta=feature_delta,
            center_delta=center_delta,
            shape_delta=shape_delta,
            warped_points_metric=warped,
            projected_features=self.student_projection(predicted_features),
            shape_scale=predicted_points.new_tensor(self.shape_scale))

    def project_teacher(self, teacher_features):
        return torch.nn.functional.linear(
            teacher_features.float(), self.teacher_projection.float()).detach()
