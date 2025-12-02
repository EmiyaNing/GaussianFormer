from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class DensifyModule(BaseModule):
    def __init__(
        self,
        scale_ratio = 1.6,
        **kwargs,
    ):
        self.scale_ratio = scale_ratio
        super(DensifyModule, self).__init__()
        

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        gaussian: GaussianPrediction,
    ):
        """高斯至密化处理
        
        对语义类别在前10个的高斯球进行至密化，每个高斯球替换为两个较小的高斯球
        新高斯球的位置相较于原高斯球偏移其尺寸半径
        同时复制对应的instance_feature和output
        """
        # 获取语义预测结果
        sem_pred = torch.argmax(gaussian.semantics, dim=-1)  # [B, N]
        
        # 创建掩码：语义类别在前10个（0-9）
        densify_mask = sem_pred < 10  # [B, N]
        
        # 如果没有需要至密化的高斯球，直接返回原始数据
        if not torch.any(densify_mask):
            return gaussian, instance_feature, anchor
        
        # 获取批次大小和原始高斯数量
        batch_size, num_gaussians = gaussian.means.shape[:2]
        embed_dims = instance_feature.shape[-1]
        output_dims = anchor.shape[-1]
        
        # 收集需要至密化的高斯球和相关数据
        densify_means = []
        densify_scales = []
        densify_rotations = []
        densify_opacities = []
        densify_semantics = []
        densify_instance_features = []
        densify_outputs = []
        
        # 收集不需要至密化的高斯球和相关数据
        keep_means = []
        keep_scales = []
        keep_rotations = []
        keep_opacities = []
        keep_semantics = []
        keep_instance_features = []
        keep_outputs = []
        
        for b in range(batch_size):
            batch_densify_mask = densify_mask[b]
            
            # 分离需要至密化和不需要至密化的高斯球
            if torch.any(batch_densify_mask):
                # 需要至密化的高斯球
                densify_means_b = gaussian.means[b][batch_densify_mask]  # [M, 3]
                densify_scales_b = gaussian.scales[b][batch_densify_mask]  # [M, 3]
                densify_rotations_b = gaussian.rotations[b][batch_densify_mask]  # [M, 4]
                densify_opacities_b = gaussian.opacities[b][batch_densify_mask]  # [M, 1]
                densify_semantics_b = gaussian.semantics[b][batch_densify_mask]  # [M, semantic_dim]
                densify_instance_features_b = instance_feature[b][batch_densify_mask]  # [M, embed_dims]
                densify_outputs_b = anchor[b][batch_densify_mask]  # [M, output_dims]
                
                # 将每个高斯球替换为两个较小的高斯球
                # 尺寸缩小为原来的1/1.6
                new_scales = densify_scales_b / self.scale_ratio
                
                # 为每个高斯球生成两个新位置，偏移其尺寸半径
                # 使用随机方向进行偏移，避免所有偏移方向相同
                M = densify_means_b.shape[0]
                random_directions = torch.randn(M, 3, device=densify_means_b.device)
                random_directions = torch.nn.functional.normalize(random_directions, dim=-1)
                
                # 计算偏移量：尺寸半径（使用平均尺度作为半径）
                radius = densify_scales_b / 2.0
                offset = random_directions * radius  # [M, 3]
                
                # 生成两个新位置：一个正向偏移，一个反向偏移
                new_means1 = densify_means_b + offset
                new_means2 = densify_means_b - offset
                
                # 合并新位置
                new_means = torch.cat([new_means1, new_means2], dim=0)  # [2*M, 3]
                
                # 使用unsqueeze + expand组合来正确扩展张量
                # 先添加一个维度，然后使用expand，最后reshape
                new_scales = new_scales.unsqueeze(1).expand(M, 2, 3).reshape(2*M, 3)  # [2*M, 3]
                new_rotations = densify_rotations_b.unsqueeze(1).expand(M, 2, 4).reshape(2*M, 4)  # [2*M, 4]
                new_opacities = densify_opacities_b.unsqueeze(1).expand(M, 2, 1).reshape(2*M, 1)  # [2*M, 1]
                new_semantics = densify_semantics_b.unsqueeze(1).expand(M, 2, densify_semantics_b.shape[1]).reshape(2*M, densify_semantics_b.shape[1])  # [2*M, semantic_dim]
                new_instance_features = densify_instance_features_b.unsqueeze(1).expand(M, 2, densify_instance_features_b.shape[1]).reshape(2*M, densify_instance_features_b.shape[1])  # [2*M, embed_dims]
                new_outputs = densify_outputs_b.unsqueeze(1).expand(M, 2, densify_outputs_b.shape[1]).reshape(2*M, densify_outputs_b.shape[1])  # [2*M, output_dims]
                
                densify_means.append(new_means)
                densify_scales.append(new_scales)
                densify_rotations.append(new_rotations)
                densify_opacities.append(new_opacities)
                densify_semantics.append(new_semantics)
                densify_instance_features.append(new_instance_features)
                densify_outputs.append(new_outputs)
            
            # 不需要至密化的高斯球
            if torch.any(~batch_densify_mask):
                keep_means_b = gaussian.means[b][~batch_densify_mask]
                keep_scales_b = gaussian.scales[b][~batch_densify_mask]
                keep_rotations_b = gaussian.rotations[b][~batch_densify_mask]
                keep_opacities_b = gaussian.opacities[b][~batch_densify_mask]
                keep_semantics_b = gaussian.semantics[b][~batch_densify_mask]
                keep_instance_features_b = instance_feature[b][~batch_densify_mask]
                keep_outputs_b = anchor[b][~batch_densify_mask]
                
                keep_means.append(keep_means_b)
                keep_scales.append(keep_scales_b)
                keep_rotations.append(keep_rotations_b)
                keep_opacities.append(keep_opacities_b)
                keep_semantics.append(keep_semantics_b)
                keep_instance_features.append(keep_instance_features_b)
                keep_outputs.append(keep_outputs_b)
        

        # 找到最大的高斯数量以进行填充
        max_gaussians = 0
        for b in range(batch_size):
            batch_keep_means = keep_means[b] if b < len(keep_means) else torch.empty(0, 3, device=gaussian.means.device)
            batch_densify_means = densify_means[b] if b < len(densify_means) else torch.empty(0, 3, device=gaussian.means.device)
            total_batch_gaussians = batch_keep_means.shape[0] + batch_densify_means.shape[0]
            max_gaussians = max(max_gaussians, total_batch_gaussians)
        
        semantic_dim = gaussian.semantics.shape[-1]
        # 初始化填充后的张量
        padded_means = torch.zeros(batch_size, max_gaussians, 3, device=gaussian.means.device)
        padded_scales = torch.zeros(batch_size, max_gaussians, 3, device=gaussian.scales.device)
        padded_rotations = torch.zeros(batch_size, max_gaussians, 4, device=gaussian.rotations.device)
        padded_opacities = torch.zeros(batch_size, max_gaussians, 1, device=gaussian.opacities.device)
        padded_semantics = torch.zeros(batch_size, max_gaussians, semantic_dim, device=gaussian.semantics.device)
        padded_instance_features = torch.zeros(batch_size, max_gaussians, embed_dims, device=instance_feature.device)
        padded_outputs = torch.zeros(batch_size, max_gaussians, output_dims, device=anchor.device)
        
        # 填充数据
        for b in range(batch_size):
            batch_keep_means = keep_means[b] if b < len(keep_means) else torch.empty(0, 3, device=gaussian.means.device)
            batch_densify_means = densify_means[b] if b < len(densify_means) else torch.empty(0, 3, device=gaussian.means.device)
            batch_keep_scales = keep_scales[b] if b < len(keep_scales) else torch.empty(0, 3, device=gaussian.scales.device)
            batch_densify_scales = densify_scales[b] if b < len(densify_scales) else torch.empty(0, 3, device=gaussian.scales.device)
            batch_keep_rotations = keep_rotations[b] if b < len(keep_rotations) else torch.empty(0, 4, device=gaussian.rotations.device)
            batch_densify_rotations = densify_rotations[b] if b < len(densify_rotations) else torch.empty(0, 4, device=gaussian.rotations.device)
            batch_keep_opacities = keep_opacities[b] if b < len(keep_opacities) else torch.empty(0, 1, device=gaussian.opacities.device)
            batch_densify_opacities = densify_opacities[b] if b < len(densify_opacities) else torch.empty(0, 1, device=gaussian.opacities.device)
            batch_keep_semantics = keep_semantics[b] if b < len(keep_semantics) else torch.empty(0, semantic_dim, device=gaussian.semantics.device)
            batch_densify_semantics = densify_semantics[b] if b < len(densify_semantics) else torch.empty(0, semantic_dim, device=gaussian.semantics.device)
            batch_keep_instance_features = keep_instance_features[b] if b < len(keep_instance_features) else torch.empty(0, embed_dims, device=instance_feature.device)
            batch_densify_instance_features = densify_instance_features[b] if b < len(densify_instance_features) else torch.empty(0, embed_dims, device=instance_feature.device)
            batch_keep_outputs = keep_outputs[b] if b < len(keep_outputs) else torch.empty(0, output_dims, device=anchor.device)
            batch_densify_outputs = densify_outputs[b] if b < len(densify_outputs) else torch.empty(0, output_dims, device=anchor.device)
            
            # 合并当前批次的数据
            batch_means = torch.cat([batch_keep_means, batch_densify_means], dim=0)
            batch_scales = torch.cat([batch_keep_scales, batch_densify_scales], dim=0)
            batch_rotations = torch.cat([batch_keep_rotations, batch_densify_rotations], dim=0)
            batch_opacities = torch.cat([batch_keep_opacities, batch_densify_opacities], dim=0)
            batch_semantics = torch.cat([batch_keep_semantics, batch_densify_semantics], dim=0)
            batch_instance_features = torch.cat([batch_keep_instance_features, batch_densify_instance_features], dim=0)
            batch_outputs = torch.cat([batch_keep_outputs, batch_densify_outputs], dim=0)
            
            # 获取当前批次的实际高斯数量
            current_gaussians = batch_means.shape[0]
            
            # 填充数据
            if current_gaussians > 0:
                padded_means[b, :current_gaussians] = batch_means
                padded_scales[b, :current_gaussians] = batch_scales
                padded_rotations[b, :current_gaussians] = batch_rotations
                padded_opacities[b, :current_gaussians] = batch_opacities
                padded_semantics[b, :current_gaussians] = batch_semantics
                padded_instance_features[b, :current_gaussians] = batch_instance_features
                padded_outputs[b, :current_gaussians] = batch_outputs
        
        # 创建新的高斯预测对象
        densified_gaussian = GaussianPrediction(
            means=padded_means,
            scales=padded_scales,
            rotations=padded_rotations,
            opacities=padded_opacities,
            semantics=padded_semantics
        )
        
        return padded_outputs, densified_gaussian, padded_instance_features