from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch.nn as nn, torch
import torch.nn.functional as F
from ..utils import GaussianPrediction
from ....utils.safe_ops import safe_sigmoid

# 裁剪到点云范围内
def clamp_means(means, pc_range):
    means_x = torch.clamp(means[:, 0], pc_range[0]+1e-6, pc_range[3]-1e-6)
    means_y = torch.clamp(means[:, 1], pc_range[1]+1e-6, pc_range[4]-1e-6)
    means_z = torch.clamp(means[:, 2], pc_range[2]+1e-6, pc_range[5]-1e-6)
    return torch.stack([means_x, means_y, means_z], dim=-1)

# 填充函数
def pad_tensor_list(tensor_list, shape, device):
    padded = []
    for t in tensor_list:
        pad_size = [0] * (len(shape) * 2)
        pad_size[-1] = shape[-1] - t.shape[-1]
        # 简化：仅填充第二维
        if t.shape[0] < shape[0]:
            pad = torch.zeros(shape, device=device, dtype=t.dtype)
            pad[:t.shape[0]] = t
            padded.append(pad)
        else:
            padded.append(t)
    return torch.stack(padded, dim=0)

@MODELS.register_module()
class DensifyAdaptive(BaseModule):
    '''
    This version densify module perform adaptive densify on the current gaussian primites.
    The densifyadaptive module use a threshold to decide which gaussian primite will perform the densify process.
    The densify gaussian will perform the clone or split decide by its scale.
    '''
    def __init__(
        self,
        embed_dim = 128,
        semantic_dim = 17,
        threshold = 0.6,
        scale_threshold = 0.5,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        **kwargs,
    ):
        super().__init__()
        self.threshold = threshold
        self.scale_threshold = scale_threshold
        self.pc_range   = pc_range
        self.scale_range= scale_range
        unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
        unit_prob = [4 * unit_prob[i] for i in range(3)]
        self.unit_sigmoid = unit_prob

        self.score_map = nn.Linear(embed_dim, 1)

        # this module perform feature shift for clone and split operation.
        self.feature_shift = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU())
        

        self.location_shift = nn.Linear(embed_dim, 3)
        self.rotation_shift = nn.Linear(embed_dim, 4)
        self.scale_shift    = nn.Linear(embed_dim, 3)
        self.semantic_shift = nn.Linear(embed_dim, semantic_dim)
        self.opacities_shift= nn.Linear(embed_dim, 1)


    def forward(
            self,
            instance_feature: torch.Tensor,
            anchor: torch.Tensor,
            gaussian: GaussianPrediction
        ):
        '''
        This module perform adaptive densify on current gaussian primitives.
            instance_feature: A tensor with shape B, N, C, represent the feature tensor for each gaussian.
            gaussian: A GaussianPrediction object, containing the gaussian parameters for each primitive.
            anchor:   A tensor encode the gaussian parameter into a tensor with shape B, N, 3+3+4+1+semantic_dim.
                      the order of gaussian parameter in anchor is [mean, scale, rotation, opacity, semantic].
                      the mean, scale will be normalized into 0-1 range, the rotation is represented in quaternion, the opacity is also normalized into 0-1 range.                
        '''
        opacities = gaussian.opacities # B, N, 1
        means     = gaussian.means
        scales    = gaussian.scales
        rotations = gaussian.rotations
        semantics = gaussian.semantics

        batch_size= opacities.shape[0]
        # this used to store the new generated gaussian parameters.
        new_features = []
        new_outputs  = []
        new_opacitices = []
        new_semantics  = []
        new_rotations  = []
        new_means      = []
        new_scales     = []

        for b in range(batch_size):
            cur_opacities = opacities[b] # N, 1
            cur_means     = means[b]     # N, 3
            cur_scales    = scales[b]    # N, 3
            cur_rots      = rotations[b] # N, 4 
            cur_sems      = semantics[b] # N, 17
            cur_anchor    = anchor[b]
            cur_features  = instance_feature[b]

            densify_features_set = []
            densify_output_set   = []
            densify_opacities_set = []
            densify_semantics_set = []
            densify_rotations_set = []
            densify_means_set = []
            densify_scales_set = []

            scores_f = self.score_map(instance_feature[b]) # N, 1
            scores_f = safe_sigmoid(scores_f)
            scores_s = cur_sems.softmax(dim=-1).max(dim=-1, keepdim=True)[0] # N, 1


            scores = (scores_f + scores_s + cur_opacities) / 3 # N, 1

            densify_mask = (scores > self.threshold).squeeze(-1) # N,

            # perform densify operation
            pre_densify_opa = cur_opacities[densify_mask] # M, 1
            pre_densify_means = cur_means[densify_mask]   # M, 3
            pre_densify_scales= cur_scales[densify_mask]  # M, 3
            pre_densify_rots  = cur_rots[densify_mask]    # M, 4
            pre_densify_sems  = cur_sems[densify_mask]    # M, 17
            pre_densify_anchor= cur_anchor[densify_mask]  # M, 3+3+4+1+semantic_dim
            pre_densify_feats = cur_features[densify_mask] # M, C
            pre_densify_feats = self.feature_shift(pre_densify_feats) # M, C

            cur_scale_range   = (pre_densify_scales ** 2).sum(dim=-1).sqrt()
            clone_mask = cur_scale_range < self.scale_threshold
            split_mask = ~clone_mask

            clone_densify_opa = pre_densify_opa[clone_mask] # M1, 1
            clone_densify_means = pre_densify_means[clone_mask] #
            clone_densify_scales= pre_densify_scales[clone_mask]
            clone_densify_rots  = pre_densify_rots[clone_mask]
            clone_densify_sems  = pre_densify_sems[clone_mask]
            clone_densify_anchor= pre_densify_anchor[clone_mask]
            clone_features = pre_densify_feats[clone_mask]

            # 在接下来的代码中我们对clone 部分的gaussian进行操作。
            clone_features = pre_densify_feats[clone_mask] # M1, C
            # 针对clone的基元，我们通过location_shift对其进行偏移一定的位置。
            # 注意，优化偏移的过程中，我们需要参考refine 模块的做法，对每个gaussian基元进行残差优化，确保优化的稳定性。
            # 针对clone出来的gaussian，我们认为其scale，rotation，semantic以及opacity保持不变，我们只对其location进行优化。
            # 计算位置偏移
            delta_xyz_sigmoid = self.location_shift(clone_features)
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            mean_shift_out = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
            clone_new_means = clone_densify_means + mean_shift_out
            # 裁剪到点云范围内
            clone_new_means_x = torch.clamp(clone_new_means[:, 0], self.pc_range[0]+1e-6, self.pc_range[3]-1e-6)
            clone_new_means_y = torch.clamp(clone_new_means[:, 1], self.pc_range[1]+1e-6, self.pc_range[4]-1e-6)
            clone_new_means_z = torch.clamp(clone_new_means[:, 2], self.pc_range[2]+1e-6, self.pc_range[5]-1e-6)
            clone_new_means = torch.stack([clone_new_means_x, clone_new_means_y, clone_new_means_z], dim=-1)
            clone_new_xyz_output = safe_sigmoid(clone_new_means)
            # 尺度、旋转、语义、不透明度保持不变
            clone_new_scales = clone_densify_scales
            clone_new_rots = clone_densify_rots
            clone_new_sems = clone_densify_sems
            clone_new_opa = clone_densify_opa
            # 更新 anchor
            clone_new_output = torch.cat([clone_new_xyz_output, clone_densify_anchor[:, 3:]], dim=-1)
            # 将新的参数添加到列表中
            densify_features_set.append(clone_features)
            densify_output_set.append(clone_new_output)
            densify_opacities_set.append(clone_new_opa)
            densify_semantics_set.append(clone_new_sems)
            densify_rotations_set.append(clone_new_rots)
            densify_means_set.append(clone_new_means)
            densify_scales_set.append(clone_new_scales)

            # 在下面的代码中，我们对split部分的gaussian进行操作。
            split_features = pre_densify_feats[split_mask] # M2, C
            # 针对split的基元，其location优化，我们让其偏移的距离限制在gaussian基元内部。
            # 同时，split出来的两个基元在scale上是原基元1/1.6。
            # 而，rotations，semantic以及opacity则通过对应的shift模块进行预测偏移。
            # 注意，rotations，semantic，opacity的优化过程同样需要使用残差优化的方式进行，以确保优化的稳定性。
            # 获取分割的高斯参数
            split_densify_opa = pre_densify_opa[split_mask]  # M2, 1
            split_densify_means = pre_densify_means[split_mask]  # M2, 3
            split_densify_scales = pre_densify_scales[split_mask]  # M2, 3
            split_densify_rots = pre_densify_rots[split_mask]  # M2, 4
            split_densify_sems = pre_densify_sems[split_mask]  # M2, 17
            split_densify_anchor = pre_densify_anchor[split_mask]  # M2, 3+3+4+1+semantic_dim
            
            M2 = split_densify_means.shape[0]
            device = split_densify_means.device
            
            # 尺度缩小为原尺度的 1/1.6
            scale_ratio = 1.6
            split_new_scales = split_densify_scales / scale_ratio  # M2, 3
            
            # 位置偏移：在随机方向上偏移半径距离（限制在高斯内部）
            random_directions = torch.randn(M2, 3, device=device)
            random_directions = F.normalize(random_directions, dim=-1)
            radius = split_densify_scales / 2.0  # 使用原尺度的一半作为半径
            offset = random_directions * radius  # M2, 3
            split_new_means1 = split_densify_means + offset
            split_new_means2 = split_densify_means - offset
            
            
            split_new_means1 = clamp_means(split_new_means1, self.pc_range)
            split_new_means2 = clamp_means(split_new_means2, self.pc_range)
            
            # 计算位置归一化输出
            split_new_xyz_output1 = safe_sigmoid(split_new_means1)
            split_new_xyz_output2 = safe_sigmoid(split_new_means2)
            
            # 使用 shift 模块预测旋转、语义、不透明度的偏移量
            delta_rots = self.rotation_shift(split_features)  # M2, 4
            delta_sems = self.semantic_shift(split_features)  # M2, semantic_dim
            delta_opa = self.opacities_shift(split_features)  # M2, 1
            
            # 残差优化：新值 = 原始值 + 偏移量（经过激活函数）
            rots_shift = F.normalize(delta_rots)  # 归一化四元数
            sems_shift = safe_sigmoid(delta_sems)
            opa_shift = safe_sigmoid(delta_opa)
            
            split_new_rots = split_densify_rots / 2 + rots_shift / 2  # 残差融合
            split_new_sems = split_densify_sems / 2 + sems_shift / 2
            split_new_opa = split_densify_opa / 2 + opa_shift / 2
            
            # 为两个新高斯复制参数
            # 第一个高斯
            split_new_means = split_new_means1
            split_new_scales = split_new_scales
            split_new_rots = split_new_rots
            split_new_sems = split_new_sems
            split_new_opa = split_new_opa
            # 第二个高斯（尺度相同，旋转、语义、不透明度相同）
            split_new_means2 = split_new_means2
            split_new_scales2 = split_new_scales  # 相同尺度
            split_new_rots2 = split_new_rots  # 相同旋转
            split_new_sems2 = split_new_sems  # 相同语义
            split_new_opa2 = split_new_opa  # 相同不透明度
            
            # 更新 anchor
            split_scale_out = split_densify_anchor[:, 3:6] / scale_ratio  # 尺度缩小
            split_rots_out = split_densify_anchor[:, 6:10] / 2 + rots_shift / 2
            split_opas_out = split_densify_anchor[:, 10:11] / 2 + opa_shift / 2
            split_sems_out = split_densify_anchor[:, 11:] / 2 + sems_shift / 2
            
            # 为两个高斯构建 anchor 输出
            split_new_output1 = torch.cat([split_new_xyz_output1, split_scale_out, split_rots_out, split_opas_out, split_sems_out], dim=-1)
            split_new_output2 = torch.cat([split_new_xyz_output2, split_scale_out, split_rots_out, split_opas_out, split_sems_out], dim=-1)
            
            # 将两个新高斯的参数添加到列表中
            # 第一个高斯
            densify_features_set.append(split_features)
            densify_output_set.append(split_new_output1)
            densify_opacities_set.append(split_new_opa)
            densify_semantics_set.append(split_new_sems)
            densify_rotations_set.append(split_new_rots)
            densify_means_set.append(split_new_means)
            densify_scales_set.append(split_new_scales)
            # 第二个高斯（特征相同）
            densify_features_set.append(split_features)
            densify_output_set.append(split_new_output2)
            densify_opacities_set.append(split_new_opa2)
            densify_semantics_set.append(split_new_sems2)
            densify_rotations_set.append(split_new_rots2)
            densify_means_set.append(split_new_means2)
            densify_scales_set.append(split_new_scales2)

            # cat all new_gaussians.
            new_features.append(torch.cat(densify_features_set, dim=0))
            new_outputs.append(torch.cat(densify_output_set, dim=0))
            new_opacitices.append(torch.cat(densify_opacities_set, dim=0))
            new_semantics.append(torch.cat(densify_semantics_set, dim=0))
            new_rotations.append(torch.cat(densify_rotations_set, dim=0))
            new_means.append(torch.cat(densify_means_set, dim=0))
            new_scales.append(torch.cat(densify_scales_set, dim=0))

            # 最后，需要注意的是，clone和split出来的基元对应的anchor属性也需要进行相应的更新，确保anchor属性能够正确地反映出新的gaussian基元的参数。
        # return的结果，新的anchor, 新的gaussian prediction， 新的instance feature。
        # 如果没有任何 densify 的高斯，直接返回原始输入
        if len(new_features) == 0:
            return anchor, gaussian, instance_feature
        
        # 将列表中的张量按批次堆叠
        # 注意：new_features 等列表包含了每个批次的新高斯，但可能每个批次的数量不同
        # 这里简化处理，假设每个批次的新高斯数量相同，直接堆叠
        try:
            new_features = torch.stack(new_features, dim=0)  # B, M, C
            new_outputs = torch.stack(new_outputs, dim=0)    # B, M, D
            new_opacitices = torch.stack(new_opacitices, dim=0)  # B, M, 1
            new_semantics = torch.stack(new_semantics, dim=0)    # B, M, semantic_dim
            new_rotations = torch.stack(new_rotations, dim=0)    # B, M, 4
            new_means = torch.stack(new_means, dim=0)            # B, M, 3
            new_scales = torch.stack(new_scales, dim=0)          # B, M, 3
        except RuntimeError:
            # 如果数量不同，无法堆叠，则进行填充
            max_M = max([f.shape[0] for f in new_features])
            B = len(new_features)
            C = new_features[0].shape[1]
            D = new_outputs[0].shape[1]
            semantic_dim = new_semantics[0].shape[1]
            device = new_features[0].device
            
            new_features = pad_tensor_list(new_features, (max_M, C), device)
            new_outputs = pad_tensor_list(new_outputs, (max_M, D), device)
            new_opacitices = pad_tensor_list(new_opacitices, (max_M, 1), device)
            new_semantics = pad_tensor_list(new_semantics, (max_M, semantic_dim), device)
            new_rotations = pad_tensor_list(new_rotations, (max_M, 4), device)
            new_means = pad_tensor_list(new_means, (max_M, 3), device)
            new_scales = pad_tensor_list(new_scales, (max_M, 3), device)
        
        # 与原始输入拼接
        result_features = torch.cat([instance_feature, new_features], dim=1)
        result_outputs = torch.cat([anchor, new_outputs], dim=1)
        result_opacitices = torch.cat([opacities, new_opacitices], dim=1)
        result_semantics = torch.cat([semantics, new_semantics], dim=1)
        result_means = torch.cat([means, new_means], dim=1)
        result_scales = torch.cat([scales, new_scales], dim=1)
        result_rotations = torch.cat([rotations, new_rotations], dim=1)
        
        # 构建新的高斯预测对象
        result_gaussian = GaussianPrediction(
            means=result_means,
            scales=result_scales,
            rotations=result_rotations,
            opacities=result_opacitices,
            semantics=result_semantics
        )
        return result_outputs, result_gaussian, result_features

            
 


