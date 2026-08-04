"""Isolated query initialization used by the faithful SparseWorld port."""
import torch
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class SparseWorldStrictQueryLifter(BaseModule):
    """Create the original 720 current and 320 future OPUS queries together.

    The future queries are deliberately *not* created by the trajectory head.
    They enter the image transformer with the current queries, exactly as in
    SparseWorld. ``stamp_statistics`` is updated by the temporal loss and is
    used after warm-up to reassign query timestamps with the source greedy
    assignment rule.
    """
    def __init__(self, num_queries=720, future_queries=(60, 60, 60, 60, 40, 40),
                 embed_dims=256, query_grad=True, learnable_features=None,
                 reference_mode=None, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        # ``learnable_features`` and ``reference_mode`` are OPUSQueryLifter
        # options inherited from old configs.  Strict SparseWorld always uses
        # zero query features and direct source-style reference anchors.
        # Accepting them here makes the module safe under MMEngine deep merge
        # while preserving those fixed semantics.
        if kwargs:
            unknown = ', '.join(sorted(kwargs))
            raise TypeError(f'Unsupported SparseWorldStrictQueryLifter options: {unknown}')
        self.num_queries = int(num_queries)
        self.future_queries = tuple(int(v) for v in future_queries)
        self.total_queries = self.num_queries + sum(self.future_queries)
        self.future_steps = len(self.future_queries)
        self.reference_points = nn.Parameter(torch.empty(self.total_queries, 3), requires_grad=query_grad)
        self.register_buffer('query_features', torch.zeros(self.total_queries, embed_dims))
        self.register_buffer('stamp_statistics', torch.ones(self.total_queries, self.future_steps + 1))
        self.register_buffer('query_stamps', self._initial_stamps(), persistent=True)
        self.reset_parameters()

    def _initial_stamps(self):
        return torch.cat([torch.full((count,), stamp, dtype=torch.long)
                          for stamp, count in enumerate((self.num_queries,) + self.future_queries)])

    def reset_parameters(self):
        nn.init.uniform_(self.reference_points[:, 0], 0., 1.1)
        nn.init.uniform_(self.reference_points[:, 1:], 0., 1.)

    @staticmethod
    def _matched(stamps, capacities):
        """Source ``get_matched_inds``: greedily respect exact stamp capacity."""
        ranking = stamps.clone()
        assigned = torch.full((ranking.shape[0],), -1, dtype=torch.long, device=ranking.device)
        for _ in range(ranking.shape[0]):
            index = ranking.amax(-1).argmax()
            stamp = ranking[index].argmax()
            assigned[index] = stamp
            ranking[index].fill_(-float('inf'))
            if (assigned == stamp).sum() == capacities[stamp]:
                ranking[:, stamp] = -float('inf')
        return assigned

    @torch.no_grad()
    def set_pretrain(self, enabled):
        if enabled:
            self.stamp_statistics.fill_(1.)
            self.query_stamps.copy_(self._initial_stamps().to(self.query_stamps))
        else:
            probability = self.stamp_statistics / self.stamp_statistics.sum(-1, keepdim=True).clamp_min(1.)
            self.query_stamps.copy_(self._matched(probability, (self.num_queries,) + self.future_queries))

    @torch.no_grad()
    def update_stamp_statistics(self, assigned_stamps):
        """Accumulate [B,Q,R] source target stamps for post-warm-up routing."""
        for stamp in range(self.future_steps + 1):
            self.stamp_statistics[:, stamp].add_((assigned_stamps == stamp).sum((0, 2)).to(self.stamp_statistics))

    def forward(self, ms_img_feats, **kwargs):
        batch = ms_img_feats[0].shape[0]
        return dict(query_features=self.query_features.unsqueeze(0).expand(batch, -1, -1),
                    query_points=self.reference_points.unsqueeze(0).expand(batch, -1, -1),
                    query_stamps=self.query_stamps)
