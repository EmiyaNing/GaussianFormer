import torch
from torch import nn
from mmengine.model import BaseModule
from mmengine.registry import MODELS


@MODELS.register_module()
class OPUSQueryLifter(BaseModule):
    """Initializes the fixed-size sparse query set used by OPUS."""

    def __init__(self, num_queries=600, embed_dims=128, query_grad=True,
                 learnable_features=True, reference_mode='sigmoid', init_cfg=None):
        super().__init__(init_cfg)
        self.num_queries = num_queries
        self.learnable_features = learnable_features
        self.reference_mode = reference_mode
        if reference_mode not in ('sigmoid', 'direct'):
            raise ValueError("reference_mode must be 'sigmoid' or 'direct'")
        features = torch.empty(num_queries, embed_dims)
        if learnable_features:
            self.query_features = nn.Parameter(features, requires_grad=query_grad)
        else:
            self.register_buffer('query_features', torch.zeros_like(features))
        # Legacy mode stores logits; strict OPUS-V1 uses direct unit-cube points.
        self.reference_points = nn.Parameter(torch.empty(num_queries, 3),
                                             requires_grad=query_grad)
        self.init_weights()

    def init_weights(self):
        if self.learnable_features:
            nn.init.xavier_uniform_(self.query_features)
        else:
            nn.init.zeros_(self.query_features)
        if self.reference_mode == 'direct':
            nn.init.uniform_(self.reference_points, 0., 1.)
        else:
            points = torch.empty_like(self.reference_points).uniform_(0.01, 0.99)
            self.reference_points.data.copy_(torch.logit(points))

    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        return {
            'query_features': self.query_features.unsqueeze(0).expand(batch_size, -1, -1),
            'query_points': (self.reference_points if self.reference_mode == 'direct'
                             else self.reference_points.sigmoid()).unsqueeze(0).expand(batch_size, -1, -1),
        }
