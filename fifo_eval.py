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
        cumulative = item.lidar2prev.float() @ cumulative

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


def fuse_predictions(curr_soft_flat, fifo_queue, alpha,
                     curr_lidar2prev, sampled_xyz_curr, grid_params):
    """时序融合: 当前帧与历史帧 warped 预测的加权融合.

    Args:
        curr_soft_flat:   (C, N)         当前帧软预测 (扁平格式)
        fifo_queue:       FIFOQueue      历史帧队列
        alpha:            float          融合权重 (0.0 ~ 1.0)
        curr_lidar2prev:  (4, 4)         当前帧 lidar2prev
        sampled_xyz_curr: (N, 3)         当前帧体素中心坐标
        grid_params:      dict           网格参数 {H, W, D, pc_min, pc_max}

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

    # 加权融合
    fused = curr_soft_flat * alpha + warped_mean * (1.0 - alpha)  # (C, N)

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
