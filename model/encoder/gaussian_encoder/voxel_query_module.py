import torch, torch.nn as nn
import frnn

import torch.nn.functional as F


from mmseg.registry import MODELS

@MODELS.register_module()
class VoxelQueryModule(nn.Module):
    '''
        This module should be used to integrate gaussian ball's feature tensor with multi-stride 
        voxel points.
    '''
    def __init__(self,
                 pc_range,
                 backbone_channels,
                 mlp_channels):
        super(VoxelQueryModule, self).__init__()
        self.pc_range = pc_range
        self.backbone_channels = backbone_channels
        #self.mlp_channels = mlp_channels

        self.mlp_in = nn.ModuleList()
        self.mlp_pos= nn.ModuleList()
        self.mlp_out= nn.ModuleList()

        for idx in range(len(mlp_channels)):
            input_ch = backbone_channels[idx]
            mlp_ch   = mlp_channels[idx]

            cur_in = nn.Sequential(
                nn.Conv1d(input_ch, mlp_ch[0], kernel_size=1, stride=1),
                nn.BatchNorm1d(mlp_ch[0]),
                nn.GELU(),
                nn.Conv1d(mlp_ch[0], mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm1d(mlp_ch[1])
            )

            cur_pos= nn.Sequential(
                nn.Conv2d(3, mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm2d(mlp_ch[2])
            )

            cur_out= nn.Sequential(
                nn.Conv1d(mlp_ch[1], mlp_ch[1], kernel_size=1, stride=1),
                nn.BatchNorm1d(mlp_ch[1]),
                nn.GELU()
            )
            self.mlp_in.append(cur_in)
            self.mlp_pos.append(cur_pos)
            self.mlp_out.append(cur_out)

    def forward(self,
                gaussian,
                instance_features,
                multi_stride_features):
        pass


