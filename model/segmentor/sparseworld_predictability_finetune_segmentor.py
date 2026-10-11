"""Experiment-local full fine tuning with native SparseWorld task heads.

No production segmentor registration or behavior is changed.  The zero-output
residual adapter starts as identity. Each adapted feature is decoded at its
own horizon; the unadapted rollout still supplies the subsequent state.
"""
import torch
from torch import nn

from model.segmentor.sparseworld_strict_segmentor import SparseWorldStrictSegmentor


class ResidualFutureAdapter(nn.Module):
    def __init__(self, dims=256):
        super().__init__()
        self.delta = nn.Sequential(nn.LayerNorm(dims), nn.Linear(dims, dims),
                                   nn.GELU(), nn.Linear(dims, dims))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, features):
        return features + self.delta(features)


class SparseWorldPredictabilityFinetuneSegmentor(SparseWorldStrictSegmentor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.future_adapter = ResidualFutureAdapter()

    def decode(self, features, positions):
        b, n = features.shape[:2]
        logits = self.semantic(features).view(b, n, self.num_refines, 17)
        offset = self.reg(features).view(b, n, self.num_refines, 3) * .5
        velocity = self.velocity(features).view(b, n, self.num_refines, 2)
        moving = ((logits.argmax(-1) >= 2) & (logits.argmax(-1) <= 10)).unsqueeze(-1)
        offset = torch.cat([offset[..., :2] + velocity * moving, offset[..., 2:]], -1)
        return self._refine(positions, offset), logits

    def forward(self, imgs, metas, **kwargs):
        result = dict(imgs=imgs, metas=metas, points=None)
        result.update(self.extract_img_feat(**result))
        result.update(self.lifter(**result))
        ego = self.plan_head(metas['temporal_ego_states'].float().reshape(imgs.shape[0], -1)).unsqueeze(1)
        scale = (torch.tanh(self.points_scale(ego)) + 1.) * .35 + .8
        result['query_points'] = result['query_points'] * scale
        result.update(self.encoder(**result))
        stages, stamps = result['representation'], result['query_stamps']
        current = stamps == 0
        current_points = [self._world(s['query_points'][:, current].flatten(1, 2)).float()
                          for s in stages]
        current_logits = [s['opus_logits'][:, current].flatten(1, 2).float() for s in stages]
        all_features, all_points = stages[-1]['query_features'], stages[-1]['query_points']
        features, positions = all_features[:, current], all_points[:, current].detach()
        timestamp = positions.new_zeros(*positions.shape[:-1], 1)
        points, logits, states, trajectories = [], [], [], []
        for step in range(self.future_steps):
            fused = self.ego_attention(ego, features.detach(), positions.detach())
            trajectories.append(self.traj(fused))
            born = stamps == step + 1
            features = torch.cat([features, all_features[:, born]], 1)
            positions = torch.cat([positions, all_points[:, born]], 1).detach()
            timestamp = torch.cat([timestamp, timestamp.new_full(
                (imgs.shape[0], int(born.sum()), self.num_refines, 1), .5)], 1)
            features = features + fused.expand(-1, features.shape[1], -1) + self.position(
                torch.cat([positions, timestamp], -1).flatten(2))
            adapted = self.future_adapter(features)
            predicted_positions, semantic = self.decode(adapted, positions)
            points.append(self._world(predicted_positions).flatten(1, 2).float())
            logits.append(semantic.flatten(1, 2).float())
            states.append(dict(features=adapted, positions=predicted_positions.detach()))
            # Preserve the original, unadapted recurrent geometry. The native
            # output above remains differentiable for this step's geometry loss.
            with torch.no_grad():
                positions, _ = self.decode(features, positions)
        return dict(opus_pred_points=current_points, opus_pred_logits=current_logits,
                    temporal_pred_points=[self._world(s['query_points'].flatten(1, 2)).float() for s in stages],
                    temporal_pred_logits=[s['opus_logits'].flatten(1, 2).float() for s in stages],
                    future_pred_points=points, future_pred_logits=logits,
                    pred_traj=torch.cat(trajectories, 1).float(),
                    future_query_states=states, sparseworld_pretrain=False, metas=metas)
