"""SparseWorld's trajectory/forecast branch on the local strict OPUS stack."""
import torch
import torch.nn.functional as F
from torch import nn
from mmseg.models import SEGMENTORS

from .opus_segmentor import OPUSSegmentor


class _TrajectoryCrossAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, dropout, pc_range):
        super().__init__()
        self.pc_range = tuple(pc_range)
        self.attention = nn.MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.tau = nn.Linear(embed_dims, num_heads)
        nn.init.zeros_(self.tau.weight)
        nn.init.uniform_(self.tau.bias, 0., 2.)

    def forward(self, query_feature, key_feature, key_points):
        # Source OPUSCrossAttention uses a distance/tau mask.  The query is a
        # single ego token at the normalized origin, so construct the same
        # geometric preference without importing upstream OPUS code.
        lower = key_points.new_tensor(self.pc_range[:3]); upper = key_points.new_tensor(self.pc_range[3:])
        world = key_points.mean(2) * (upper - lower) + lower
        distance = world.norm(dim=-1)
        penalty = distance[:, None, :].expand(-1, self.attention.num_heads, -1)
        penalty = penalty * F.softplus(self.tau(key_feature)).transpose(1, 2) / 40.
        # ``MultiheadAttention`` accepts either [L, S] or [B * H, L, S].
        # There is one ego query (L=1), so keep that dimension when expanding
        # the per-head geometric penalty.  Flattening [B, H, S] directly made
        # PyTorch interpret [B * H, S] as a 2-D mask with L=B*H.
        mask = penalty.unsqueeze(2).flatten(0, 1)
        return self.attention(query_feature, key_feature, key_feature, attn_mask=mask,
                              need_weights=False)[0]


@SEGMENTORS.register_module()
class SparseWorldTrajSegmentor(OPUSSegmentor):
    """Current strict OPUS plus SparseWorld's future occupancy recurrence."""
    def __init__(self, future_queries=(60, 60, 60, 60, 40, 40), future_steps=6,
                 embed_dims=256, num_refines=48, pc_range=(-40., -40., -1., 40., 40., 5.4),
                 ego_state_dim=21, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        if len(future_queries) != future_steps:
            raise ValueError('future_queries must contain one entry per future step')
        self.future_queries, self.future_steps = tuple(future_queries), future_steps
        self.num_refines, self.pc_range = num_refines, tuple(pc_range)
        self.future_reference = nn.Parameter(torch.empty(sum(future_queries), 3))
        nn.init.uniform_(self.future_reference, 0., 1.)
        self.plan_head = nn.Sequential(nn.Linear(ego_state_dim, embed_dims), nn.ReLU(True),
                                       nn.Linear(embed_dims, embed_dims), nn.ReLU(True),
                                       nn.Linear(embed_dims, embed_dims))
        self.cross_attention = _TrajectoryCrossAttention(embed_dims, 8, dropout, pc_range)
        self.position_encoder = nn.Sequential(nn.Linear(4 * num_refines, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True),
                                              nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(True))
        def branch(out):
            return nn.Sequential(nn.Linear(embed_dims, embed_dims), nn.ReLU(True), nn.Linear(embed_dims, embed_dims), nn.ReLU(True), nn.Linear(embed_dims, out))
        self.reg_branch, self.vel_branch, self.cls_branch = branch(num_refines * 3), branch(num_refines * 2), branch(num_refines * 17)
        self.traj_head = nn.Sequential(nn.Linear(embed_dims, embed_dims * 2), nn.Softplus(), nn.Linear(embed_dims * 2, 2))
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _world(self, points):
        lower = points.new_tensor(self.pc_range[:3]); upper = points.new_tensor(self.pc_range[3:])
        return points * (upper - lower) + lower

    def _normalized(self, points):
        lower = points.new_tensor(self.pc_range[:3]); upper = points.new_tensor(self.pc_range[3:])
        return (points - lower) / (upper - lower)

    def forward(self, imgs=None, metas=None, points=None, **kwargs):
        results = {'imgs': imgs, 'metas': metas, 'points': points}; results.update(kwargs)
        results.update(self.extract_img_feat(**results))
        results.update(self.lifter(**results))
        encoded = self.encoder(**results); results.update(encoded)
        results.update(self.head(**results))
        last = encoded['representation'][-1]
        features, positions = last['query_features'], last['query_points'].detach()
        state = metas['temporal_ego_states'].to(features).reshape(features.shape[0], -1)
        ego = self.plan_head(state).unsqueeze(1)
        predictions, logits, trajectories = [], [], []
        future_offset = 0
        horizon = self.future_steps if not self.training else max(1, min(self.current_epoch - 5 + 1, self.future_steps))
        for step in range(horizon):
            fused = self.cross_attention(ego, features.detach(), positions.detach())
            trajectories.append(self.traj_head(fused))
            count = self.future_queries[step]
            reference = self.future_reference[future_offset:future_offset + count].view(1, count, 1, 3).expand(features.shape[0], -1, self.num_refines, -1)
            future_offset += count
            positions = torch.cat([positions, reference], 1)
            features = torch.cat([features, torch.zeros(features.shape[0], count, features.shape[-1], device=features.device, dtype=features.dtype)], 1)
            timestamp = positions.new_zeros(*positions.shape[:-1], 1); timestamp[:, -count:] = .5
            features = features + fused.expand(-1, features.shape[1], -1) + self.position_encoder(torch.cat([positions, timestamp], -1).flatten(2))
            offsets = self.reg_branch(features).view(features.shape[0], features.shape[1], self.num_refines, 3) * .5
            semantic = self.cls_branch(features).view(features.shape[0], features.shape[1], self.num_refines, 17)
            velocity = self.vel_branch(features).view(features.shape[0], features.shape[1], self.num_refines, 2)
            moving = ((semantic.argmax(-1) >= 2) & (semantic.argmax(-1) <= 10)).unsqueeze(-1)
            offsets[..., :2] = offsets[..., :2] + velocity * moving
            positions = self._normalized(self._world(positions).mean(2, keepdim=True) + offsets).clamp(0., 1.).detach()
            predictions.append(self._world(positions).flatten(1, 2)); logits.append(semantic.flatten(1, 2))
        # SparseWorld pretraining supervises ego displacement but defers future
        # occupancy loss until epoch five.  Preserve the predictions for local
        # debugging while expose empty loss lists during that warm-up phase.
        loss_predictions, loss_logits = predictions, logits
        if self.training and self.current_epoch < 5:
            loss_predictions, loss_logits = [], []
        results.update(future_pred_points=loss_predictions, future_pred_logits=loss_logits,
                       future_predictions=predictions, future_logits=logits,
                       pred_traj=torch.cat(trajectories, 1), future_horizon=horizon)
        return results
