from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction



@MODELS.register_module()
class GaussianSemanticRefine(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        semantics_activation='softmax',
        **kwargs,
    ):
        super(GaussianSemanticRefine, self).__init__()
        self.embed_dims = embed_dims

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
                
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
  

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.semantic_dim),
            Scale([1.0] * self.semantic_dim))

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        ori_gaussian: GaussianPrediction,
    ):
        sem_refine = self.layers(instance_feature)
        geo_attribute = anchor[..., :self.semantic_start]
        sem_attribute = anchor[..., self.semantic_start:(self.semantic_start + self.semantic_dim)]
        sem_outputs   = (sem_attribute + sem_refine) / 2
        
        if self.semantics_activation == 'softmax':
            sem_outputs = sem_outputs.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            sem_outputs = F.softplus(sem_outputs)
                    
        output = torch.cat([geo_attribute, sem_outputs], dim=-1)
        gaussian = GaussianPrediction(
            means=ori_gaussian.means,
            scales=ori_gaussian.scales,
            rotations=ori_gaussian.rotations,
            opacities=ori_gaussian.opacities,
            semantics=sem_outputs
        )
        return output, gaussian 

