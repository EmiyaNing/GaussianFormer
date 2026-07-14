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
        nn.init.uniform_(self.reference_points, -1.0, 1.0)

    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        return {
            'query_features': self.query_features.unsqueeze(0).expand(batch_size, -1, -1),
            'query_points': self.reference_points.sigmoid().unsqueeze(0).expand(batch_size, -1, -1),
        }
