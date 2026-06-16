"""GaussianDensifyLoss —— 辅助损失模块，为 AdaptiveAllocationV4 densify 提供直接梯度监督。

包含 4 个子损失:
  - CoverageLoss:         高斯覆盖体素的覆盖率损失（可微马氏距离匹配）
  - CloneDiversityLoss:   克隆子高斯的特征/空间多样性损失
  - RoutingEntropyLoss:   路由概率熵损失（防止操作退化）
  - RatioBalanceLoss:     逐场景操作比率平衡损失（★核心）

所有辅助损失均直接作用于 gaussian 属性，梯度路径绕开 head，为 densify 参数
提供"梯度高速公路"。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmengine.model import BaseModule
from typing import Dict, Optional, Tuple

from loss import OPENOCC_LOSS

# ---------------------------------------------------------------------------
#  工具函数
# ---------------------------------------------------------------------------

def _get_rotation_matrix(rotations: torch.Tensor) -> torch.Tensor:
    """四元数 [w,x,y,z] → 旋转矩阵 [3,3]（批量版本）。"""
    tensor = F.normalize(rotations, dim=-1)
    w, x, y, z = tensor.unbind(dim=-1)
    col_x = torch.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ], dim=-1)
    col_y = torch.stack([
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ], dim=-1)
    col_z = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ], dim=-1)
    return torch.stack([col_x, col_y, col_z], dim=-1)


# ---------------------------------------------------------------------------
#  子损失 1: CoverageLossComputer
# ---------------------------------------------------------------------------

class CoverageLossComputer:
    """可微覆盖率损失：计算高斯对可见体素的覆盖率。

    使用软化的马氏距离匹配 + 分块计算控制显存。
    """

    def __init__(self, cov_threshold: float = 3.0, temperature: float = 0.1,
                 chunk_size: int = 10000, voxel_sample_ratio: float = 0.05):
        self.cov_threshold = cov_threshold
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.voxel_sample_ratio = voxel_sample_ratio
        self.tau_sq = cov_threshold ** 2

    def __call__(self, means: torch.Tensor, scales: torch.Tensor,
                 rotations: torch.Tensor, metas: dict) -> torch.Tensor:
        """计算覆盖率损失。

        Args:
            means:     [B, N, 3]
            scales:    [B, N, 3]
            rotations:[B, N, 4]
            metas:     dict with 'occ_xyz' [B,X,Y,Z,3], 'occ_cam_mask' [B,X,Y,Z]
        Returns:
            loss_coverage: scalar tensor
        """
        B, N, _ = means.shape
        device = means.device
        tau_sq = self.tau_sq

        occ_xyz = metas.get('occ_xyz')
        occ_cam_mask = metas.get('occ_cam_mask')
        if occ_xyz is None or occ_cam_mask is None:
            return torch.tensor(0.0, device=device)

        # 展平并提取可见体素
        if occ_xyz.dim() == 5:
            occ_xyz = occ_xyz[0]
        if occ_cam_mask.dim() == 4:
            occ_cam_mask = occ_cam_mask[0]

        occ_flat = occ_xyz.reshape(-1, 3)              # [V_total, 3]
        mask_flat = occ_cam_mask.reshape(-1).bool()     # [V_total]
        if not mask_flat.any():
            return torch.tensor(0.0, device=device)

        valid_xyz = occ_flat[mask_flat]                 # [V, 3]
        V = valid_xyz.shape[0]

        # 随机降采样体素
        V_sample = max(100, int(V * self.voxel_sample_ratio))
        if V_sample < V:
            indices = torch.randperm(V, device=device)[:V_sample]
            valid_xyz = valid_xyz[indices]
            V = V_sample

        # 预计算协方差相关量
        R = _get_rotation_matrix(rotations)             # [B, N, 3, 3]
        search_radius = self.cov_threshold * scales.max(dim=-1).values  # [B, N]

        total_covered = torch.tensor(0.0, device=device)
        total_voxels = torch.tensor(float(V), device=device)

        for b in range(B):
            means_b = means[b]                          # [N, 3]
            scales_b = scales[b]                        # [N, 3]
            R_b = R[b]                                  # [N, 3, 3]
            sr_b = search_radius[b]                     # [N]

            covered_sum = torch.tensor(0.0, device=device)

            for start in range(0, V, self.chunk_size):
                end = min(start + self.chunk_size, V)
                chunk_v = valid_xyz[start:end]          # [C, 3]

                # 欧氏距离预筛
                dists = torch.cdist(chunk_v, means_b)   # [C, N]
                cand_mask = dists < sr_b.unsqueeze(0)    # [C, N]
                if not cand_mask.any():
                    continue

                # 马氏距离精筛（可微）
                vi, gj = cand_mask.nonzero(as_tuple=True)
                diff = chunk_v[vi] - means_b[gj]         # [P, 3]
                d_rot = torch.bmm(
                    R_b[gj], diff.unsqueeze(-1)).squeeze(-1)  # [P, 3]
                mahal = (d_rot / scales_b[gj].clamp(min=1e-8)).pow(2).sum(dim=-1)  # [P]

                # 软命中：sigmoid((τ² - mahal) / T)
                hit_soft = torch.sigmoid((tau_sq - mahal) / self.temperature)  # [P]

                # 每个体素取最大命中值（任一高斯覆盖即算覆盖）
                max_hit = torch.zeros(end - start, device=device)
                max_hit = max_hit.scatter_reduce(
                    0, vi, hit_soft, reduce='amax', include_self=False)
                covered_sum += max_hit.sum()

            total_covered += covered_sum

        coverage = total_covered / total_voxels.clamp(min=1)
        return 1.0 - coverage  # 损失 = 1 - 覆盖率


# ---------------------------------------------------------------------------
#  子损失 2: CloneDiversityLossComputer
# ---------------------------------------------------------------------------

class CloneDiversityLossComputer:
    """克隆多样性损失：鼓励克隆子高斯在特征和空间上与父高斯不同。"""

    def __init__(self, spatial_weight: float = 0.5, min_dist_ratio: float = 0.5):
        self.spatial_weight = spatial_weight
        self.min_dist_ratio = min_dist_ratio

    def __call__(self, child_gaussian, parent_gaussian,
                 child_feat: torch.Tensor, parent_feat: torch.Tensor) -> torch.Tensor:
        """计算克隆多样性损失。

        Args:
            child_gaussian:  GaussianPrediction (N_c, ...) 子高斯
            parent_gaussian: GaussianPrediction (N_c, ...) 父高斯
            child_feat:      [N_c, E] 子特征
            parent_feat:     [N_c, E] 父特征
        Returns:
            loss_clone_div: scalar
        """
        N_c = child_feat.shape[0]
        if N_c == 0:
            return torch.tensor(0.0, device=child_feat.device)

        # 特征多样性：鼓励低余弦相似度
        feat_sim = F.cosine_similarity(child_feat, parent_feat, dim=-1)  # [N_c]
        loss_feat_div = -feat_sim.mean()  # 最大化不相似度

        # 空间多样性：鼓励最小间距 > ratio × mean(parent_scale)
        child_means = child_gaussian.means
        parent_means = parent_gaussian.means
        parent_scales = parent_gaussian.scales

        spatial_dist = (child_means - parent_means).norm(dim=-1)        # [N_c]
        min_dist = parent_scales.mean(dim=-1) * self.min_dist_ratio     # [N_c]
        loss_spatial = F.relu(min_dist - spatial_dist).mean()

        return loss_feat_div + self.spatial_weight * loss_spatial


# ---------------------------------------------------------------------------
#  子损失 3: RoutingEntropyLossComputer
# ---------------------------------------------------------------------------

class RoutingEntropyLossComputer:
    """路由熵损失：最大化逐场景的操作概率熵，防止过早退化到单一操作。"""

    def __call__(self, op_prob: torch.Tensor) -> torch.Tensor:
        """计算路由熵损失。

        Args:
            op_prob: [B, N, 4] 软操作概率（来自 softmax）
        Returns:
            loss_route_ent: scalar (负熵均值，越小越好)
        """
        B = op_prob.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=op_prob.device)

        eps = 1e-8
        # 逐场景平均概率
        p_bar = op_prob.mean(dim=1)                      # [B, 4]
        # 熵 H = -Σ p*log(p)
        entropy = -(p_bar * (p_bar + eps).log()).sum(dim=-1)  # [B]
        # 最大化熵 = 最小化负熵
        return -entropy.mean()


# ---------------------------------------------------------------------------
#  子损失 4: RatioBalanceLossComputer（★ 核心）
# ---------------------------------------------------------------------------

class RatioBalanceLossComputer:
    """操作比率平衡损失：确保 clone/split/atten 在每个场景中保有合理比率。

    使用双侧 relu 惩罚：只在实际比率超出 [target_min, target_max] 时施加惩罚。
    基于 op_prob 软统计，梯度可穿过 operation_head → h → state_encoder。
    """

    def __init__(self, ratio_targets: Optional[Dict[int, Tuple[float, float]]] = None,
                 cross_scene_var_weight: float = 0.1):
        # 默认目标范围
        if ratio_targets is None:
            ratio_targets = {
                0: (0.40, 0.85),  # KEEP: 允许较大范围
                1: (0.05, 0.30),  # CLONE
                2: (0.05, 0.25),  # SPLIT
                3: (0.02, 0.15),  # ATTEN
            }
        self.ratio_targets = ratio_targets
        self.cross_scene_var_weight = cross_scene_var_weight

    def __call__(self, op_prob: torch.Tensor) -> torch.Tensor:
        """计算比率平衡损失。

        Args:
            op_prob: [B, N, 4] 软操作概率
        Returns:
            loss_ratio_balance: scalar
        """
        B = op_prob.shape[0]
        if B == 0:
            return torch.tensor(0.0, device=op_prob.device)

        ratio_soft = op_prob.mean(dim=1)  # [B, 4] 逐场景软比率

        loss = torch.tensor(0.0, device=op_prob.device)
        for op_id, (t_min, t_max) in self.ratio_targets.items():
            r = ratio_soft[:, op_id]                     # [B]
            # 双侧 relu: 只在超出范围时惩罚
            loss_low = F.relu(t_min - r).pow(2).mean()
            loss_high = F.relu(r - t_max).pow(2).mean()
            loss += loss_low + loss_high

        # 跨场景方差惩罚：防止某些场景极端偏离
        if B > 1:
            for op_id in self.ratio_targets:
                loss += self.cross_scene_var_weight * ratio_soft[:, op_id].std()

        return loss

    def get_current_ratios(self, op_prob: torch.Tensor) -> Dict[int, float]:
        """返回当前逐场景平均操作比率（用于日志/监控）。"""
        if op_prob.shape[0] == 0:
            return {}
        ratio_soft = op_prob.mean(dim=1).mean(dim=0)  # [4]
        return {i: ratio_soft[i].item() for i in range(4)}


# ---------------------------------------------------------------------------
#  主损失类: GaussianDensifyLoss
# ---------------------------------------------------------------------------

@OPENOCC_LOSS.register_module()
class GaussianDensifyLoss(BaseModule):
    """Gaussian Densify 辅助损失。

    为 AdaptiveAllocationV4 模块提供直接的梯度监督，绕过 head 的间接路径。

    Args:
        coverage_weight:       覆盖率损失权重 (默认 0.1)
        clone_div_weight:      克隆多样性损失权重 (默认 0.05)
        route_ent_weight:      路由熵损失权重 (默认 0.01)
        ratio_balance_weight:  比率平衡损失权重 (默认 0.05)
        cov_threshold:         马氏距离覆盖阈值 (默认 3.0)
        cov_temperature:       软命中温度 (默认 0.1)
        voxel_sample_ratio:    体素降采样比例 (默认 0.05)
        coverage_compute_every: 覆盖率计算频率 (默认 5, 每5步算一次)
        ratio_targets:         操作比率目标范围 dict
        init_cfg:              mmengine 初始化配置
    """

    def __init__(self,
                 coverage_weight: float = 0.1,
                 clone_div_weight: float = 0.05,
                 route_ent_weight: float = 0.01,
                 ratio_balance_weight: float = 0.05,
                 cov_threshold: float = 3.0,
                 cov_temperature: float = 0.1,
                 voxel_sample_ratio: float = 0.05,
                 coverage_compute_every: int = 5,
                 ratio_targets: Optional[dict] = None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.coverage_weight = coverage_weight
        self.clone_div_weight = clone_div_weight
        self.route_ent_weight = route_ent_weight
        self.ratio_balance_weight = ratio_balance_weight
        self.coverage_compute_every = coverage_compute_every

        # 子损失计算机（无参数，不需要 optimizer）
        self.coverage_loss = CoverageLossComputer(
            cov_threshold=cov_threshold,
            temperature=cov_temperature,
            voxel_sample_ratio=voxel_sample_ratio,
        )
        self.clone_div_loss = CloneDiversityLossComputer()
        self.route_ent_loss = RoutingEntropyLossComputer()

        # 解析 ratio_targets (支持整数键 0-3 或名称 'clone'/'split'/'atten'/'keep')
        parsed_targets = None
        if ratio_targets is not None:
            _name_to_id = {'keep': 0, 'clone': 1, 'split': 2, 'atten': 3}
            parsed_targets = {}
            for k, v in ratio_targets.items():
                if isinstance(k, str) and k in _name_to_id:
                    op_id = _name_to_id[k]
                else:
                    op_id = int(k)
                parsed_targets[op_id] = (float(v[0]), float(v[1]))
        self.ratio_balance_loss = RatioBalanceLossComputer(
            ratio_targets=parsed_targets)

        self._step_counter = 0

    def forward(self, inputs):
        """计算所有辅助损失的总和（符合 MultiLoss 接口）。

        Args:
            inputs: dict with keys:
                - gaussian:         GaussianPrediction [B, N, ...]
                - metas:            dict with occ_xyz, occ_cam_mask
                - global_iter:      int
                - densify_op_prob:  [B, N, 4] (optional)
                - densify_op_id:    [B, N]    (optional)
        Returns:
            tot_loss: scalar tensor
        """
        gaussian = inputs.get('gaussian')
        metas = inputs.get('metas', {})
        densify_op_prob = inputs.get('densify_op_prob')
        densify_op_id = inputs.get('densify_op_id')

        if gaussian is None:
            return torch.tensor(0.0)

        device = gaussian.means.device
        total = torch.tensor(0.0, device=device)

        # ---- 路由熵损失 ----
        if self.route_ent_weight > 0 and densify_op_prob is not None:
            total = total + self.route_ent_weight * self.route_ent_loss(densify_op_prob)

        # ---- 比率平衡损失 ----
        if self.ratio_balance_weight > 0 and densify_op_prob is not None:
            total = total + self.ratio_balance_weight * self.ratio_balance_loss(densify_op_prob)

        # ---- 覆盖率损失（每 N 步计算一次以节省开销） ----
        if self.coverage_weight > 0:
            self._step_counter += 1
            if self._step_counter % self.coverage_compute_every == 0:
                total = total + self.coverage_weight * self.coverage_loss(
                    gaussian.means, gaussian.scales,
                    gaussian.rotations, metas)

        # ---- 克隆多样性损失（当前为占位，需 parent_indices） ----
        # if self.clone_div_weight > 0 and densify_op_prob is not None:
        #     pass

        return total
