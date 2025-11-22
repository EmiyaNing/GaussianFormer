import frnn
import torch, torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

from .base_loss import BaseLoss
from .occupancy_loss import sigmoid_focal_loss

from . import OPENOCC_LOSS

@OPENOCC_LOSS.register_module()
class GaussianSemanticLoss(BaseLoss):
    def __init__(self, 
                 weight=1.0,
                 num_classes=18,
                 manual_class_weight=None,
                 input_dict=None):
        super().__init__()
        self.weight = weight 
        self.num_classses  = num_classes
        self.class_weights = torch.tensor(manual_class_weight)
        self.input_dict = input_dict
        self.loss_func  = self.gaussian_semantic_loss
        self.semantic_func = nn.CrossEntropyLoss(
            reduction="mean"
        )
        self.opacities_func = sigmoid_focal_loss


    def gaussian_semantic_loss(self, gaussian, sampled_label, sampled_xyz, empty_label=17):
        """
        gaussians should be a GaussianPrediction class
        sampled_label should be semantic label of each voxel
        sampled_xyz should be the coordinates of world
        """

        B, G, _ = gaussian.means.shape
        N = sampled_xyz.shape[1]
        device = gaussian.means.device
        

        
        # 前景掩码
        foreground_mask = (sampled_label != empty_label)
        

        means     = gaussian.means
        scales    = gaussian.scales
        opacities = gaussian.opacities
        semantics = gaussian.semantics

        dist, idxs, point_nn, grid = frnn.frnn_grid_points(
            means, sampled_xyz, K=27, r=2.0, return_nn=True
        )

        foreground_mask = frnn.frnn_gather(foreground_mask.unsqueeze(-1), idxs).squeeze(-1)

        filter_mask  = foreground_mask.sum(-1) > 0
        filter_semantics = semantics[filter_mask]

        gaussian_label  = frnn.frnn_gather(sampled_label.unsqueeze(-1), idxs).squeeze(-1)
        filter_gs_label = gaussian_label[filter_mask]


        semantic_label  = torch.zeros(*filter_semantics.shape[:-1], self.num_classses - 1, dtype=semantics.dtype, device=semantics.device)
        
        semantic_label  = semantic_label.permute(1, 0)
        for i in range(self.num_classses - 1):
            cur_sem_counts = (filter_gs_label == i).sum(-1)
            semantic_label[i] += cur_sem_counts
        
        semantic_label = semantic_label.permute(1, 0)
        semantic_label = semantic_label / 27

        filter_semantics     = filter_semantics.softmax(dim=-1)

        cls_weights   = self.class_weights.to(filter_semantics.device)
        semantic_loss = self.semantic_func(filter_semantics * cls_weights[:-1], semantic_label* cls_weights[:-1])
        #import pdb
        #pdb.set_trace()
        opacities_loss= self.opacities_func(opacities.squeeze(0), filter_mask.squeeze(0).long())

        total_loss = (semantic_loss + opacities_loss) * self.weight


        return total_loss
    
