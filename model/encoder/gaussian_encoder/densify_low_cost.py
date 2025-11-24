from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch as torch
from .utils import GaussianPrediction

@MODELS.register_module()
class DensifyModuleV2(BaseModule):
    '''
    This version densify module should consider two things:
        1) only densify the most important gaussian ball.
        2) and we should discard these low confidence score ball.
    '''
    def __init__(
        self,
        scale_ratio = 1.6,
        semantic_dim = 17,
        anchor_num = 25600,
        **kwargs,
    ):
        self.scale_ratio = scale_ratio
        self.semantic_dim = semantic_dim
        self.anchor_num   = anchor_num
        super(DensifyModuleV2, self).__init__()

    def forward(self,
                instance_feature: torch.Tensor,
                anchor: torch.Tensor,
                gaussian: GaussianPrediction):
        '''
            Firstly, the gaussian ball total count should be kept the same.
            Secondly, we should define a strategy to update the feature vector.
            Thirdly, the gaussian ball should kept the same with anchors.
        '''
        opacities = gaussian.opacities
        semantics = gaussian.semantics
        scales    = gaussian.scales
        rotations = gaussian.rotations


        




        pass

        