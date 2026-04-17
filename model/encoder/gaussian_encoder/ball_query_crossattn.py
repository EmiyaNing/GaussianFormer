import torch.nn as nn, torch
import torch.nn.functional as F
import frnn

from functools import partial

from mmengine.registry import MODELS
from mmengine.model import BaseModule

from .utils import spherical2cartesian, cartesian

@MODELS.register_module()
class BallQueryCrossAttn(BaseModule):
    '''
        A simple version ball-query based cross attention.
        This module use ball-query mechanism to aggregate 27 neirghboor for each point.
        Each point will use the 27 neighboor point to perform windows cross-attention.
    '''
    def __init__(self, 
                 pc_range,
                 embed_dim,
                 query_k,
                 radius):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_k   = query_k
        self.radius    = radius

        self.value_mapping = nn.Linear(embed_dim, embed_dim)
        self.get_xyz = partial(cartesian, pc_range=pc_range, use_sigmoid=True)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))


    def forward(self,
                anchor,
                instance_features):
        '''
            This module use a similiar pathway with spconv3d_module.
            anchor is the attribute of gaussian ball.
            instance_features is a feature tensor for each gaussian ball.
        '''
        bs, g, _ = instance_features.shape


        anchor_xyz = anchor[..., :3]
        anchor_xyz = self.get_xyz(anchor_xyz)

        _, idx, _, _ = frnn.frnn_grid_points(
            anchor_xyz, anchor_xyz, K=self.query_k, r=self.radius, return_nn=True
        )

        zero_mask  = (idx == -1).unsqueeze(-1)

        query_feat = frnn.frnn_gather(instance_features, idx)
        query_feat = query_feat * zero_mask

        value_feat = self.value_mapping(query_feat) * zero_mask

        # construct the multi-head now fix the head num as 8
        # so the multiple idx is  sqrt(128 / 8) = 4
        key_feat   = instance_features.reshape(bs, g, 8, -1).unsqueeze(2).permute(0, 1, 3, 2, 4)
        query_feat = query_feat.reshape(bs, g, 27, 8, -1).permute(0, 1, 3, 4, 2)
        value_feat = value_feat.reshape(bs, g, 27, 8, -1).permute(0, 1, 3, 2, 4)

        # matrix dot of K and Q, then softmax
        mask_feat  = torch.einsum('abcde, abcef -> abcdf', key_feat, query_feat)
        mask_feat  = F.softmax(mask_feat) / 4

        # matrix dot of softmax results with value to get final result.
        result_feat= torch.einsum('abcdf, abcfe -> abcde', mask_feat, value_feat)
        result_feat= result_feat.permute(0, 1, 3, 2, 4).reshape(bs, g, -1)

        return result_feat