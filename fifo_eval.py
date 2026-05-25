"""
FIFO 时序融合评估模块

按 FIFO_eval_breakdown.md 实现，包含:
  - FIFOQueue:         FIFO 队列数据结构 (子任务 1)
  - build_chain_transform: 坐标变换链构建 (子任务 2)
  - warp_occupancy:    占据体素 Warping / 3D Grid Sampling (子任务 3)
  - fuse_predictions:  时序融合逻辑 (子任务 4)

形状约定:
  - 软预测扁平格式 (flat):  (C, N)  其中 N = W * H * D, 元素顺序与 occ_xyz flatten 一致
  - 软预测网格格式 (grid):  (C, D, H, W)  用于 grid_sample 的输入格式
  - grid_sample 输出:       (C, H, W, D)
  - 硬标签:                 (N,)  或 (H, W, D)
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from model.encoder.gaussian_encoder.utils import GaussianPrediction


# ─── 子任务 1: FIFO 队列数据结构 ─────────────────────────────────


@dataclass
class FIFOQueueItem:
    """FIFO 队列元素

    Attributes:
        soft_pred_grid:  (C, D, H, W)  软预测 (已 reshape 为 grid_sample 输入格式)
        lidar2prev:      (4, 4)        该帧 → 前一帧的变换矩阵
        frame_idx:       int           场景内帧索引
        scene_token:     str           所属场景 token
    """
    soft_pred_grid: torch.Tensor  # (C, D, H, W)
    lidar2prev: torch.Tensor      # (4, 4)
    frame_idx: int
    scene_token: str


class FIFOQueue:
    """FIFO 环形缓冲队列 (子任务 1)"""

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer: List[FIFOQueueItem] = []

    def push(self, soft_pred_grid, lidar2prev, frame_idx, scene_token):
        """队尾追加; 超出 maxlen 则弹出最旧"""
        item = FIFOQueueItem(
            soft_pred_grid=soft_pred_grid,
            lidar2prev=lidar2prev,
            frame_idx=frame_idx,
            scene_token=scene_token,
        )
        if len(self.buffer) >= self.maxlen:
            self.buffer.pop(0)
        self.buffer.append(item)

    def clear(self):
        """清空队列 (场景切换时调用)"""
        self.buffer = []

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def size(self) -> int:
        return len(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) == self.maxlen

    def get_all(self) -> List[FIFOQueueItem]:
        """返回所有元素 (从旧到新)"""
        return self.buffer


# ─── 子任务 2: 坐标变换链构建 ─────────────────────────────────


def build_chain_transform(fifo_queue, curr_lidar2prev):
    """构建从当前帧 T 到每个历史帧 T-k 的变换链.

    变换含义: pos_{T-k} = T_{T→T-k} @ pos_T

    Args:
        fifo_queue:      FIFOQueue, 每个 item 包含 lidar2prev
        curr_lidar2prev: (4, 4)      当前帧 data['lidar2prev']

    Returns:
        chains: list[(4,4)]  每个历史帧对应的 T_{T→T-k} 变换矩阵
                chains[0] = T_{T→T-1}, chains[1] = T_{T→T-2}, ...
                与 fifo_queue.get_all() 一对一对应 (从最近到最远)
    """
    chains = []
    # 统一为 float32, 兼容 dataset 输出的 float64 与模型输出的 float32
    cumulative = curr_lidar2prev.clone().float()

    # FIFO 队列内容: [旧 ... T-k, ..., T-2, T-1]
    # reversed:     [T-1, T-2, ..., T-k]  — 从最近到最远
    for item in reversed(fifo_queue.get_all()):
        chains.append(cumulative.clone())
        # 向更远的过去累积: cumulative = lidar2prev_{T-k} @ cumulative
        cumulative = item.lidar2prev.float().to(cumulative.device) @ cumulative

    # chains: [T_{T→T-1}, T_{T→T-2}, ..., T_{T→T-k}]
    return chains


# ─── 子任务 3: 占据体素 Warping ────────────────────────────────


def _flat_to_grid(soft_pred_flat, W, H, D):
    """将扁平软预测转换为 grid_sample 输入格式.

    soft_pred_flat shape: (C, N)  where N = W*H*D
    扁平顺序: index = w * H * D + h * D + d  (与 occ_xyz flatten 一致)

    Returns:
        soft_pred_grid: (C, D, H, W)  适用于 grid_sample(input, ...)
    """
    C = soft_pred_flat.shape[0]
    # (C, N) → (C, W, H, D) → (C, D, H, W)
    return soft_pred_flat.reshape(C, W, H, D).permute(0, 3, 2, 1).contiguous()


def _grid_to_flat(soft_pred_grid, W, H, D):
    """将 grid_sample 输出格式转换回扁平格式.

    soft_pred_grid shape: (C, H, W, D)  (grid_sample 输出)
    或 (C, D, H, W) (grid_sample 输入)

    Returns:
        soft_pred_flat: (C, N)  扁平格式, 顺序与 occ_xyz flatten 一致
    """
    C = soft_pred_grid.shape[0]

    if soft_pred_grid.shape[1:] == (D, H, W):
        # 输入格式 (C, D, H, W) → (C, H, W, D) → (C, W, H, D) → (C, N)
        x = soft_pred_grid.permute(0, 2, 3, 1)  # (C, H, W, D)
    elif soft_pred_grid.shape[1:] == (H, W, D):
        # 输出格式 (C, H, W, D)
        x = soft_pred_grid
    else:
        raise ValueError(f"Unexpected shape: {soft_pred_grid.shape}")

    # (C, H, W, D) → (C, W, H, D) → (C, N)
    x = x.permute(0, 2, 1, 3).contiguous()  # (C, W, H, D)
    return x.reshape(C, -1)  # (C, N)


def warp_occupancy(hist_soft_pred_grid, T_matrix, sampled_xyz_curr, grid_params):
    """将历史帧的占据预测 warp 到当前帧坐标系.

    Args:
        hist_soft_pred_grid: (C, D, H, W)  历史帧软预测 (grid_sample 输入格式)
        T_matrix:            (4, 4)         T_{T→T-k} 变换矩阵
        sampled_xyz_curr:    (N, 3)         当前帧体素中心坐标
        grid_params:         dict           包含 H, W, D, pc_min, pc_max

    Returns:
        warped_pred: (C, H, W, D)  warp 后的预测 (grid_sample 输出格式)
    """
    H = grid_params['H']
    W = grid_params['W']
    D = grid_params['D']
    pc_min = grid_params['pc_min']
    pc_max = grid_params['pc_max']

    device = hist_soft_pred_grid.device

    # ===== Step 1: Reshape 体素坐标为网格格式 (H, W, D, 3) =====
    # sampled_xyz_curr 扁平顺序: index = w*H*D + h*D + d
    # reshape(H, W, D, 3) 按 C-order: d 最快, 然后 h, 然后 w
    voxel_coords = sampled_xyz_curr.reshape(H, W, D, 3).to(device)
    # voxel_coords[h, w, d] = (x, y, z)  ← 注意: h, w, d 顺序
    # 但由于 H=W=200, 顺序不影响数值正确性, 关键是坐标值本身正确

    # ===== Step 2: 堆叠为齐次坐标 =====
    ones = torch.ones(H, W, D, 1, device=device)
    coords = torch.cat([voxel_coords, ones], dim=-1)  # (H, W, D, 4)

    # ===== Step 3: 应用变换矩阵 =====
    # coords @ T.T: 每个位置 (h,w,d) 的齐次坐标左乘 T 矩阵
    # 统一为 float32: sampled_xyz 来自模型为 float32, 而 T_matrix 可能为 float64
    T_device = T_matrix.to(device).float()
    coords_hist = coords @ T_device.T  # (H, W, D, 4)
    # 去齐次化 (防止除零)
    coords_hist = coords_hist[..., :3] / (coords_hist[..., 3:4] + 1e-8)

    # ===== Step 4: 归一化到 [-1, 1] =====
    norm_coords = torch.zeros_like(coords_hist)
    norm_coords[..., 0] = 2.0 * (coords_hist[..., 0] - pc_min[0]) / (pc_max[0] - pc_min[0]) - 1.0
    norm_coords[..., 1] = 2.0 * (coords_hist[..., 1] - pc_min[1]) / (pc_max[1] - pc_min[1]) - 1.0
    norm_coords[..., 2] = 2.0 * (coords_hist[..., 2] - pc_min[2]) / (pc_max[2] - pc_min[2]) - 1.0

    # 添加 batch 维度
    norm_coords = norm_coords.unsqueeze(0)  # (1, H, W, D, 3)

    # ===== Step 5: Grid Sampling =====
    # hist_soft_pred_grid: (C, D, H, W) → (1, C, D, H, W)
    hist_pred_batch = hist_soft_pred_grid.unsqueeze(0)

    warped = F.grid_sample(
        hist_pred_batch,    # (1, C, D, H, W)
        norm_coords,        # (1, H, W, D, 3)
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    # warped: (1, C, H, W, D) → (C, H, W, D)
    return warped.squeeze(0)


# ─── 子任务 4: 时序融合逻辑 ─────────────────────────────────


def compute_nonempty_mask(soft_pred):
    """计算非空体素掩码: argmax != 0 的位置为"非空".

    第 0 类为背景/空类, 第 1~16 类为语义类别.

    Args:
        soft_pred: (C, *)  软预测 logits, 第 0 维为类别维度

    Returns:
        nonempty_mask: (*)  bool 张量, True 表示该体素预测为非空
    """
    pred_class = soft_pred.argmax(dim=0)   # (*)
    return pred_class != 0                 # (*) bool


def fuse_predictions(curr_soft_flat, fifo_queue, alpha,
                     curr_lidar2prev, sampled_xyz_curr, grid_params,
                     mode='conditional'):
    """时序融合: 当前帧与历史帧 warped 预测的融合.

    支持两种模式:
      - 'simple':      所有体素无差别加权平均 (旧逻辑)
                        fused = curr * α + hist_mean * (1-α)
      - 'conditional': 根据空/非空状态分三种情况融合 (新逻辑)
                        情况 A: 当前空 + 历史非空 → 使用历史
                        情况 B: 当前非空 + 历史空 → 使用当前
                        情况 C: 当前非空 + 历史非空 → 加权融合
                        情况 D: 当前空 + 历史空 → 使用当前 (均为背景)

    Args:
        curr_soft_flat:   (C, N)         当前帧软预测 (扁平格式)
        fifo_queue:       FIFOQueue      历史帧队列
        alpha:            float          融合权重 (0.0 ~ 1.0)
        curr_lidar2prev:  (4, 4)         当前帧 lidar2prev
        sampled_xyz_curr: (N, 3)         当前帧体素中心坐标
        grid_params:      dict           网格参数 {H, W, D, pc_min, pc_max}
        mode:             str            融合模式: 'simple' | 'conditional'

    Returns:
        fused_soft: (C, N)  融合后的软预测 (扁平格式)
    """
    if fifo_queue.is_empty():
        return curr_soft_flat

    H = grid_params['H']
    W = grid_params['W']
    D = grid_params['D']

    # 构建变换链
    chains = build_chain_transform(fifo_queue, curr_lidar2prev)

    # 对每个历史帧做 warping
    warped_flat_list = []
    items = fifo_queue.get_all()
    for item, T in zip(items, chains):
        # item.soft_pred_grid: (C, D, H, W)
        warped_grid = warp_occupancy(
            item.soft_pred_grid, T, sampled_xyz_curr, grid_params
        )
        # warped_grid: (C, H, W, D) → (C, N)
        warped_flat = _grid_to_flat(warped_grid, W, H, D)
        warped_flat_list.append(warped_flat)

    # 历史帧等权平均
    warped_stack = torch.stack(warped_flat_list, dim=0)  # (K, C, N)
    warped_mean = warped_stack.mean(dim=0)                # (C, N)

    if mode == 'simple':
        # ── 旧逻辑: 所有体素无差别加权平均 ──
        fused = curr_soft_flat * alpha + warped_mean * (1.0 - alpha)
    elif mode == 'conditional':
        # ── 新逻辑: 根据空/非空状态分情况融合 ──

        # 计算空/非空掩码
        curr_mask = compute_nonempty_mask(curr_soft_flat)  # (N,) bool
        hist_mask = compute_nonempty_mask(warped_mean)     # (N,) bool

        # 初始化 fused = curr (覆盖情况 B 和情况 D)
        fused = curr_soft_flat.clone()

        # 情况 A: 当前空, 历史非空 → 完全使用历史
        mask_A = (~curr_mask) & hist_mask
        if mask_A.any():
            fused[:, mask_A] = warped_mean[:, mask_A]

        # 情况 C: 当前非空, 历史非空 → 加权融合
        mask_C = curr_mask & hist_mask
        if mask_C.any():
            fused[:, mask_C] = (
                curr_soft_flat[:, mask_C] * alpha +
                warped_mean[:, mask_C] * (1.0 - alpha)
            )

        # 情况 B: 当前非空 + 历史空 → fused 已 = curr, 无需操作
        # 情况 D: 当前空 + 历史空 → fused 已 = curr (都是背景, 无差异)
    else:
        raise ValueError(f"Unknown fusion mode: '{mode}'. Expected 'simple' or 'conditional'.")

    return fused


# ─── 辅助函数: 软预测转换与硬标签获取 ─────────────────────────


def soft_pred_to_grid(soft_pred_flat, W, H, D):
    """将模型输出的软预测转换为 grid_sample 输入格式.

    Args:
        soft_pred_flat: (C, N)  来自 result_dict['pred_occ'][-1][idx]
        W, H, D:        网格维度

    Returns:
        soft_pred_grid: (C, D, H, W)
    """
    return _flat_to_grid(soft_pred_flat, W, H, D)


def fused_soft_to_hard(fused_soft_flat):
    """将融合后的软预测转换为硬标签 (参考 gaussian_head.py:186).

    Args:
        fused_soft_flat: (C, N)  融合后的软预测

    Returns:
        hard_label: (N,)   argmax 后的类别索引
    """
    return fused_soft_flat.argmax(dim=0)


# ═══════════════════════════════════════════════════════════════
# Gaussian FIFO 流式融合模块 (gaussian_fifo_breakdown.md)
# ═══════════════════════════════════════════════════════════════

# ─── 子任务 5: GaussianFIFOQueue 数据结构 ──────────────────────


@dataclass
class GaussianFIFOQueueItem:
    """Gaussian FIFO 队列元素

    Attributes:
        gaussian:    GaussianPrediction  历史帧语义高斯 (CPU 存储)
        lidar2prev:  (4, 4)             该帧 → 前一帧的变换矩阵 (CPU)
        frame_idx:   int                场景内帧索引
        scene_token: str                所属场景 token
    """
    gaussian: GaussianPrediction
    lidar2prev: torch.Tensor  # (4, 4)
    frame_idx: int
    scene_token: str


class GaussianFIFOQueue:
    """Gaussian FIFO 环形缓冲队列 (子任务 5)

    存储历史帧的语义高斯 (GaussianPrediction)，
    所有 Tensor 均 detach().cpu() 以节省 GPU 显存。
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer: List[GaussianFIFOQueueItem] = []

    def push(self, gaussian, lidar2prev, frame_idx, scene_token):
        """队尾追加; 超出 maxlen 则弹出最旧。

        所有 Tensor 自动 detach 并移至 CPU。
        """
        item = GaussianFIFOQueueItem(
            gaussian=GaussianPrediction(
                means=gaussian.means.detach().cpu(),
                scales=gaussian.scales.detach().cpu(),
                rotations=gaussian.rotations.detach().cpu(),
                opacities=gaussian.opacities.detach().cpu(),
                semantics=gaussian.semantics.detach().cpu(),
            ),
            lidar2prev=lidar2prev.detach().cpu(),
            frame_idx=frame_idx,
            scene_token=scene_token,
        )
        if len(self.buffer) >= self.maxlen:
            self.buffer.pop(0)
        self.buffer.append(item)

    def clear(self):
        """清空队列 (场景切换时调用)"""
        self.buffer = []

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def size(self) -> int:
        return len(self.buffer)

    def get_all(self) -> List[GaussianFIFOQueueItem]:
        """返回所有元素 (从旧到新)"""
        return self.buffer


# ─── 子任务 1: 高斯坐标变换与越界过滤 ─────────────────────────


def transform_and_filter_gaussians(hist_gaussian, T_curr2hist, pc_min, pc_max):
    """将历史帧高斯变换到当前帧坐标系，并过滤越界高斯。

    变换方向: pos_curr = inv(T_curr2hist) @ pos_hist
    即对 build_chain_transform 输出的 T_{T→T-k} 求逆。

    Args:
        hist_gaussian:  GaussianPrediction  (1, G, *)  历史帧高斯
        T_curr2hist:    Tensor (4, 4)       T_{T→T-k} 变换矩阵
        pc_min:         Tensor (3,)         有效空间下界
        pc_max:         Tensor (3,)         有效空间上界

    Returns:
        filtered: GaussianPrediction  (1, G', *)  过滤后的高斯, G' ≤ G
    """
    device = T_curr2hist.device

    # Step 1: 求逆得到正向变换 T_{T-k→T}
    T_hist2curr = torch.inverse(T_curr2hist.float())  # (4, 4)

    # Step 2: 变换高斯 means
    means_hist = hist_gaussian.means.to(device)
    if means_hist.dim() == 3:
        means_hist = means_hist.squeeze(0)  # (G, 3)

    G = means_hist.shape[0]
    if G == 0:
        return _make_empty_gaussian(device, hist_gaussian)

    ones = torch.ones(G, 1, device=device, dtype=means_hist.dtype)
    means_homo = torch.cat([means_hist, ones], dim=-1)          # (G, 4)
    means_transformed = (T_hist2curr @ means_homo.T).T          # (G, 4)
    means_transformed = means_transformed[..., :3]               # (G, 3)

    # Step 3: 边界检查
    pc_min_dev = pc_min.to(device)
    pc_max_dev = pc_max.to(device)
    valid_mask = (means_transformed >= pc_min_dev) & (means_transformed <= pc_max_dev)
    valid_mask = valid_mask.all(dim=-1)  # (G,) bool

    if not valid_mask.any():
        return _make_empty_gaussian(device, hist_gaussian)

    # Step 4: 过滤所有属性（means 使用变换后的坐标）
    scales = hist_gaussian.scales.to(device)
    rotations = hist_gaussian.rotations.to(device)
    opacities = hist_gaussian.opacities.to(device)
    semantics = hist_gaussian.semantics.to(device)

    return GaussianPrediction(
        means=means_transformed[valid_mask].unsqueeze(0),              # (1, G', 3)
        scales=_index_dim1(scales, valid_mask),                        # (1, G', 3)
        rotations=_index_dim1(rotations, valid_mask),                  # (1, G', 4)
        opacities=_index_dim1(opacities, valid_mask),                  # (1, G', 1)
        semantics=_index_dim1(semantics, valid_mask),                  # (1, G', C)
    )


def _index_dim1(tensor, mask):
    """沿 dim=1 按 bool mask 索引，兼容 3D 和 2D tensor。"""
    if tensor.dim() == 3:
        return tensor[:, mask, :]
    else:
        return tensor[mask].unsqueeze(0)


def _make_empty_gaussian(device, ref_gaussian):
    """创建一个 G=0 的空 GaussianPrediction。"""
    C = ref_gaussian.semantics.shape[-1]
    return GaussianPrediction(
        means=torch.zeros(1, 0, 3, device=device),
        scales=torch.zeros(1, 0, 3, device=device),
        rotations=torch.zeros(1, 0, 4, device=device),
        opacities=torch.zeros(1, 0, 1, device=device),
        semantics=torch.zeros(1, 0, C, device=device),
    )


# ─── 子任务 2: 多帧高斯叠加与合并 ─────────────────────────────


def merge_gaussians(curr_gaussian, hist_gaussians_list):
    """将当前帧高斯与多个历史帧高斯沿 G 维度拼接。

    Args:
        curr_gaussian:       GaussianPrediction  (1, G_curr, *)
        hist_gaussians_list: list[GaussianPrediction]  每个 (1, G_i', *)

    Returns:
        merged: GaussianPrediction  (1, G_total, *)  G_total = G_curr + Σ G_i'
    """
    # 收集所有要合并的高斯（跳过 G=0 的空高斯）
    all_gaussians = [curr_gaussian]
    for hg in hist_gaussians_list:
        if hg.means.shape[1] > 0:  # G_i' > 0
            all_gaussians.append(hg)

    if len(all_gaussians) == 1:
        return curr_gaussian

    # 沿 G 维度 (dim=1) 拼接所有属性
    merged = GaussianPrediction(
        means=torch.cat([g.means for g in all_gaussians], dim=1),
        scales=torch.cat([g.scales for g in all_gaussians], dim=1),
        rotations=torch.cat([g.rotations for g in all_gaussians], dim=1),
        opacities=torch.cat([g.opacities for g in all_gaussians], dim=1),
        semantics=torch.cat([g.semantics for g in all_gaussians], dim=1),
    )
    return merged


# ─── 子任务 3: 合并高斯渲染为 Occupancy ────────────────────────


def render_gaussian_to_occupancy(merged_gaussian, sampled_xyz, head):
    """将合并后的高斯渲染为 occupancy 硬标签。

    调用 head.prepare_gaussian_args() 准备参数，
    再调用 head.aggregator() 进行 CUDA 渲染，
    最后 argmax 得到硬标签。

    Args:
        merged_gaussian: GaussianPrediction  (1, G_total, *)
        sampled_xyz:     Tensor              (1, N, 3)  查询点坐标
        head:            GaussianHead        模型 head (已包含 aggregator)

    Returns:
        hard_pred: Tensor (N,)  argmax 后的类别索引
    """
    # Step 1: 准备高斯渲染参数
    # prepare_gaussian_args 处理 with_empty / use_localaggprob 等逻辑
    means, origi_opa, opacities, scales, CovInv = \
        head.prepare_gaussian_args(merged_gaussian)

    # Step 2: CUDA 渲染
    bs, g = means.shape[:2]
    logits = head.aggregator(
        sampled_xyz.clone().float(),   # (1, N, 3)
        means,                          # (1, G_total+empty, 3)
        origi_opa.reshape(bs, g),       # (1, G_total+empty)
        opacities,                      # (1, G_total+empty, C)
        scales,                         # (1, G_total+empty, 3)
        CovInv,                         # (1, G_total+empty, 3, 3)
    )
    # logits: (N, C)  — local_aggregate 模式返回单个 Tensor

    # Step 3: argmax 得到硬标签 (参考 gaussian_head.py:186)
    hard_pred = logits.argmax(dim=-1)   # (N,)

    return hard_pred


# ─── 子任务 4: Gaussian FIFO 融合主入口 ───────────────────────


def gaussian_fifo_fuse_and_render(curr_gaussian, fifo_queue, curr_lidar2prev,
                                   sampled_xyz, head, grid_params):
    """Gaussian FIFO 融合主函数: 变换 + 合并 + 渲染。

    流程:
      1. 队列空 → 直接渲染当前帧高斯
      2. build_chain_transform() 构建 T_{T→T-k} 链
      3. 对每个历史帧: transform_and_filter_gaussians()
      4. merge_gaussians() 沿 G 维拼接
      5. render_gaussian_to_occupancy() 渲染为硬标签

    Args:
        curr_gaussian:    GaussianPrediction  (1, G_curr, *)
        fifo_queue:       GaussianFIFOQueue
        curr_lidar2prev:  Tensor (4, 4)       当前帧 lidar2prev
        sampled_xyz:      Tensor (1, N, 3)    查询点坐标
        head:             GaussianHead
        grid_params:      dict  {pc_min, pc_max, H, W, D}

    Returns:
        hard_pred:       Tensor (N,)          融合后的硬标签
        merged_gaussian: GaussianPrediction   融合后的语义高斯 (用于可视化)
    """
    # 队列为空 → 直接渲染当前帧
    if fifo_queue.is_empty():
        return render_gaussian_to_occupancy(curr_gaussian, sampled_xyz, head), curr_gaussian

    # 构建变换链 (复用现有函数，通过 duck-typing 兼容 GaussianFIFOQueue)
    chains = build_chain_transform(fifo_queue, curr_lidar2prev)
    # chains[i] = T_{T→T-i-1}

    # 对每个历史帧: 移至 GPU → 变换 → 过滤
    hist_filtered = []
    items = fifo_queue.get_all()
    pc_min = grid_params['pc_min']
    pc_max = grid_params['pc_max']

    for item, T_curr2hist in zip(items, chains):
        # 将 CPU 上的历史高斯移至 GPU
        hist_g = GaussianPrediction(
            means=item.gaussian.means.cuda(),
            scales=item.gaussian.scales.cuda(),
            rotations=item.gaussian.rotations.cuda(),
            opacities=item.gaussian.opacities.cuda(),
            semantics=item.gaussian.semantics.cuda(),
        )
        filtered = transform_and_filter_gaussians(
            hist_g, T_curr2hist, pc_min, pc_max
        )
        hist_filtered.append(filtered)

    # 合并当前帧与所有历史帧高斯
    merged = merge_gaussians(curr_gaussian, hist_filtered)

    # 渲染为 occupancy + 返回 merged gaussian 供可视化
    return render_gaussian_to_occupancy(merged, sampled_xyz, head), merged
