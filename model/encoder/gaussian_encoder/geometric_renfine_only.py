from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class GaussianGeometricRefine(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        restrict_xyz=False,
        unit_xyz=None,
        semantic_dim=None,
        include_opa=True,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super(GaussianGeometricRefine, self).__init__()
        self.embed_dims = embed_dims              
        self.semantic_start = 10 + int(include_opa)

        self.include_opa = include_opa
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        self.unit_xyz = unit_xyz
        self.semantic_dim = semantic_dim
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            if xyz_activation == "sigmoid":
                unit_prob = [4 * unit_prob[i] for i in range(3)]
            self.unit_sigmoid = unit_prob
        

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.semantic_start),
            Scale([1.0] * self.semantic_start))

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        ori_gaussians: GaussianPrediction,
    ):
        geo_refine = self.layers(instance_feature)
        geo_attribute = anchor[..., :self.semantic_start]
        if self.restrict_xyz:
            delta_xyz_sigmoid = geo_attribute[..., :3]
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
            output = torch.cat([delta_xyz, geo_attribute[..., 3:]], dim=-1)
        sem_attribute = anchor[..., self.semantic_start:(self.semantic_start + self.semantic_dim)]
        geo_output    = (geo_attribute + geo_refine) / 2
        output        = torch.cat([geo_output, sem_attribute], dim=-1)


        if self.xyz_act == "sigmoid":
            xyz = output[..., :3]
        else:
            xyz = output[..., :3].clamp(min=1e-6, max=1-1e-6)
        
        if self.scale_act == "sigmoid":
            scale = output[..., 3:6]
        else:
            scale = output[..., 3:6].clamp(min=1e-6, max=1-1e-6)

        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1)
        output = torch.cat([xyz, scale, rot, output[..., 10:]], dim=-1)
        
        if self.xyz_act == 'sigmoid':
            xyz = safe_sigmoid(xyz)
        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        if self.scale_act == 'sigmoid':
            gs_scales = safe_sigmoid(scale)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        

        
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]),
            semantics=ori_gaussians.semantics
        )
        return output, gaussian 

