"""Stage-one sparse latent matching and shared residual dynamics."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def transform_points(points, transform):
    """Apply a batched column-vector SE(3) transform to metric points."""
    if transform.shape[-2:] != (4, 4):
        raise ValueError('transform must end in [4,4]')
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    flat = points.reshape(points.shape[0], -1, 3)
    warped = torch.bmm(flat, rotation.transpose(1, 2)) + translation[:, None]
    return warped.reshape_as(points)


@dataclass
class SparseMatches:
    prediction: list
    target: list
    costs: list


class HungarianSparseMatcher:
    """Non-differentiable audit-friendly matching for unordered queries."""
    def __init__(self, position_weight=2., semantic_weight=1.,
                 feature_weight=.25, max_cost=20.):
        self.position_weight = float(position_weight)
        self.semantic_weight = float(semantic_weight)
        self.feature_weight = float(feature_weight)
        self.max_cost = float(max_cost)

    @torch.no_grad()
    def __call__(self, predicted, target):
        pred_center = predicted['points_metric'].mean(2).float()
        target_center = target['points_metric'].mean(2).float()
        pred_sem = predicted['semantic_logits'].float().log_softmax(-1)
        target_sem = target['semantic_logits'].float().softmax(-1)
        pred_feat = F.normalize(predicted['query_feats'].float(), dim=-1)
        target_feat = F.normalize(target['query_feats'].float(), dim=-1)
        pred_indices, target_indices, matched_costs = [], [], []
        for batch_index in range(pred_center.shape[0]):
            position = torch.cdist(
                pred_center[batch_index], target_center[batch_index], p=1)
            semantic = -(target_sem[batch_index][None] *
                         pred_sem[batch_index][:, None]).sum(-1)
            feature = 1. - pred_feat[batch_index] @ target_feat[batch_index].T
            cost = (self.position_weight * position +
                    self.semantic_weight * semantic +
                    self.feature_weight * feature)
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
            rows = torch.as_tensor(rows, device=cost.device, dtype=torch.long)
            cols = torch.as_tensor(cols, device=cost.device, dtype=torch.long)
            values = cost[rows, cols]
            keep = values <= self.max_cost
            pred_indices.append(rows[keep])
            target_indices.append(cols[keep])
            matched_costs.append(values[keep])
        return SparseMatches(pred_indices, target_indices, matched_costs)


class _SparseDynamicsLayer(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, ffn_dims=512,
                 dropout=.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dims, embed_dims))

    def forward(self, query, context):
        query = self.norm1(query + self.cross_attn(
            query, context, context, need_weights=False)[0])
        query = self.norm2(query + self.self_attn(
            query, query, query, need_weights=False)[0])
        return self.norm3(query + self.ffn(query))


class SharedSparseLatentPredictor(nn.Module):
    """Predict future OPUS final states from unordered observed query sets.

    Observed states are expressed in current-ego coordinates.  The current
    points are rigidly mapped into the requested future ego frame first; the
    network predicts a shared residual without static/dynamic routing.
    """
    def __init__(self, embed_dims=256, point_dims=32, num_points=128,
                 num_classes=17, num_horizons=6, num_layers=3,
                 num_heads=8, max_position_residual=12.):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.point_dims = int(point_dims)
        self.num_points = int(num_points)
        self.max_position_residual = float(max_position_residual)
        self.position_encoder = nn.Sequential(
            nn.Linear(6, embed_dims), nn.LayerNorm(embed_dims), nn.GELU(),
            nn.Linear(embed_dims, embed_dims))
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dims, embed_dims), nn.LayerNorm(embed_dims),
            nn.GELU(), nn.Linear(embed_dims, embed_dims))
        self.semantic_encoder = nn.Linear(num_classes, embed_dims)
        self.time_embedding = nn.Parameter(torch.empty(32, embed_dims))
        self.horizon_embedding = nn.Parameter(
            torch.empty(num_horizons, embed_dims))
        self.layers = nn.ModuleList([
            _SparseDynamicsLayer(embed_dims, num_heads)
            for _ in range(num_layers)])
        self.query_residual = nn.Linear(embed_dims, embed_dims)
        self.center_residual = nn.Linear(embed_dims, 3)
        self.shape_residual = nn.Linear(embed_dims, num_points * 3)
        self.point_residual = nn.Linear(embed_dims, num_points * point_dims)
        self.semantic_head = nn.Linear(embed_dims, num_classes)
        self.existence_head = nn.Linear(embed_dims, 1)
        nn.init.normal_(self.time_embedding, std=.02)
        nn.init.normal_(self.horizon_embedding, std=.02)
        for layer in (self.query_residual, self.center_residual,
                      self.shape_residual, self.point_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _tokens(self, state, time_index):
        points = state['points_metric']
        center = points.mean(2)
        spread = points.std(2, unbiased=False)
        return (state['query_feats'] + self.position_encoder(
                    torch.cat([center, spread], -1)) +
                self.point_encoder(state['point_feats'].mean(2)) +
                self.semantic_encoder(state['semantic_logits'].softmax(-1)) +
                self.time_embedding[time_index])

    def forward(self, observed_states, current_to_future, horizon):
        if not observed_states:
            raise ValueError('at least one observed sparse state is required')
        current = observed_states[-1]
        if current['point_feats'].shape[2] != self.num_points:
            raise ValueError('predictor num_points does not match OPUS state')
        horizon = int(horizon)
        aligned_states = []
        for state in observed_states:
            aligned = dict(state)
            aligned['points_metric'] = transform_points(
                state['points_metric'], current_to_future)
            aligned_states.append(aligned)
        query = self._tokens(aligned_states[-1], len(observed_states) - 1)
        query = query + self.horizon_embedding[horizon]
        context = torch.cat([
            self._tokens(state, time_index)
            for time_index, state in enumerate(aligned_states)], dim=1)
        for layer in self.layers:
            query = layer(query, context)

        warped = transform_points(current['points_metric'], current_to_future)
        center = warped.mean(2, keepdim=True)
        local = warped - center
        center_delta = torch.tanh(self.center_residual(query))[:, :, None]
        center_delta = center_delta * self.max_position_residual
        shape_delta = self.shape_residual(query).view(
            *query.shape[:2], self.num_points, 3)
        shape_delta = torch.tanh(shape_delta) * self.max_position_residual
        points = center + center_delta + local + shape_delta
        point_delta = self.point_residual(query).view(
            *query.shape[:2], self.num_points, self.point_dims)
        return dict(
            points_metric=points,
            query_feats=current['query_feats'] + self.query_residual(query),
            point_feats=current['point_feats'] + point_delta,
            semantic_logits=self.semantic_head(query),
            existence_logits=self.existence_head(query).squeeze(-1))
