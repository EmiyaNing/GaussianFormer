import frnn
import torch, torch.nn as nn
import torch.nn.functional as F

from .base_loss import BaseLoss

from . import OPENOCC_LOSS

@OPENOCC_LOSS.register_module()
class GaussianCoverageLoss(BaseLoss):
    def __init__(self, 
                 weight=1.0,
                 coverage_threshold=0.5, 
                 empty_weight=0.1, 
                 size_penalty=0.01,
                 input_dict=None):
        super().__init__()
        self.weight = weight 
        self.coverage_threshold = coverage_threshold
        self.empty_weight = empty_weight
        self.size_penalty = size_penalty
        self.loss_func    = self.loss_converage
        self.input_dict   = input_dict
        
    def loss_converage(self, gaussians, sampled_label, sampled_xyz, empty_label=17):
        """极致的显存优化版本"""
        B, G, _ = gaussians.means.shape
        N = sampled_xyz.shape[1]
        device = gaussians.means.device
        

        
        # 前景掩码
        foreground_mask = (sampled_label != empty_label)
        

        means     = gaussians.means
        scales    = gaussians.scales
        opacities = gaussians.opacities
        
        _, idxs, nn, grid = frnn.frnn_grid_points(
            means, sampled_xyz, K=27, r=2.0, return_nn=True
        )

        foreground_mask = frnn.frnn_gather(foreground_mask.unsqueeze(-1), idxs).squeeze(-1)

        # caculate the mahalanobis distance 
        dists     = means.unsqueeze(2) - nn
        cov_invs  = 1.0 / (scales.unsqueeze(2)**2 + 1e-6)
        mahalanobis = (dists**2 * cov_invs).sum(-1)

        # caculate the weights for each gaussian ball
        weights = opacities * torch.exp(-0.5 * mahalanobis)

        # caculate the foreground_weight 
        coverage_numerator   = (weights * foreground_mask).sum(-1)
        # caculate the total weights
        coverage_denominator = weights.sum(-1)
                    
        # 计算损失
        coverage_ratio = coverage_numerator / (coverage_denominator + 1e-6)
        coverage_loss  = coverage_ratio.mean()
        
        #coverage_loss = F.mse_loss(
        #    coverage_ratio, 
        #    torch.ones_like(coverage_ratio) * self.coverage_threshold
        #)
        
        empty_penalty = F.relu(self.coverage_threshold - coverage_ratio).mean()
        scale_penalty = self.size_penalty * gaussians.scales.norm(dim=-1).mean()
        
        total_loss = coverage_loss + self.empty_weight * empty_penalty + scale_penalty

        disp_dict  = {
            'coverage_ratio_mean': coverage_ratio.mean().item(),
            'coverage_loss': coverage_loss.item(),
            'empty_penalty': empty_penalty.item(),
            'scale_penalty': scale_penalty.item()
        }
        
        return total_loss
