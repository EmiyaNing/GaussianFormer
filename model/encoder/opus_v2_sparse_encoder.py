"""Single-frame OPUS-V2 point-feature decoder for the local Occ3D pipeline.

The official V2 decoder predicts 32-channel features for child points, then
discretizes them with a sparse voxel head.  This adapter uses the repository's
strict OPUS sampling implementation and V2's five-stage point budget.
"""
import torch
from mmengine.registry import MODELS

from .opus_encoder import StrictOPUSV1Encoder


@MODELS.register_module()
class OPUSV2SparseEncoder(StrictOPUSV1Encoder):
    def __init__(self, point_feature_dims=32, num_decoder=5,
                 num_refines=(8, 16, 32, 64, 128), **kwargs):
        super().__init__(num_classes=point_feature_dims, num_decoder=num_decoder,
                         num_refines=num_refines, **kwargs)
        self.point_feature_dims = point_feature_dims
        # V1's negative class-prior bias is inappropriate for V2 point features.
        for layer in self.layers:
            torch.nn.init.zeros_(layer.cls_branch[-1].bias)

    def forward(self, query_points, **kwargs):
        result = super().forward(query_points=query_points, **kwargs)
        for state in result['representation']:
            state['point_features'] = state.pop('opus_logits')
        lower = query_points.new_tensor(self.layers[0].pc_range[:3])
        upper = query_points.new_tensor(self.layers[0].pc_range[3:])
        result['v2_init_points'] = query_points * (upper - lower) + lower
        return result
