import torch
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class OPUSQueryLifter(BaseModule):
    """Initializes the fixed-size sparse query set used by OPUS."""

    def __init__(self, num_queries=600, embed_dims=128, query_grad=True, init_cfg=None):
        super().__init__(init_cfg)
        self.num_queries = num_queries
        self.query_features = nn.Parameter(torch.empty(num_queries, embed_dims),
                                           requires_grad=query_grad)
        # Stored as logits so updates can remain in the normalized unit cube.
        self.reference_points = nn.Parameter(torch.empty(num_queries, 3),
                                             requires_grad=query_grad)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.query_features)
        # Initialize uniformly in model space. Sampling logits directly in
        # [-1, 1] only covers [0.27, 0.73] after sigmoid and misses boundaries.
        points = torch.empty_like(self.reference_points).uniform_(0.01, 0.99)
        self.reference_points.data.copy_(torch.logit(points))

    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        return {
            'query_features': self.query_features.unsqueeze(0).expand(batch_size, -1, -1),
            'query_points': self.reference_points.sigmoid().unsqueeze(0).expand(batch_size, -1, -1),
        }
