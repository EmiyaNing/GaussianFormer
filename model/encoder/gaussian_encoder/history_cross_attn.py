from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmengine.model import xavier_init, constant_init
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from .utils import linear_relu_ln


@MODELS.register_module()
class HistoryCrossAttention(BaseModule):
    """跨帧图像特征交叉注意力模块。

    对当前帧和历史帧的图像特征进行 deformable cross attention，
    使用 SparseGaussian3DKeyPointsGenerator 生成 3D 采样点，
    将采样点投影到各帧图像（仅使用最大特征图），
    再通过跨帧注意力权重进行加权融合。

    输出与 DeformableFeatureAggregation 完全一致：
        (B, num_anchor, embed_dims)

    输入:
        instance_feature:   (B, num_anchor, embed_dims)  实例特征
        anchor:             (B, num_anchor, anchor_dims)  3DGS 锚点参数
        anchor_embed:       (B, num_anchor, embed_dims)  锚点位置编码
        multi_stride_features: List[(B, total_cams, C, H, W)]  多尺度 FPN 特征
        metas:              dict  包含 projection_mat, image_wh 等
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_camera_embed: bool = False,
    ):
        super(HistoryCrossAttention, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.attn_drop = attn_drop
        self.use_camera_embed = use_camera_embed
        self.proj_drop = nn.Dropout(proj_drop)

        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.num_pts = self.kps_generator.num_pts

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            # 有 camera_embed 时 FC 输出不包含 num_cams 维度
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        multi_stride_features: List[torch.Tensor],
        metas: dict,
        **kwargs,
    ):
        bs, num_anchor = instance_feature.shape[:2]

        # ---- Step 1: 生成 3D 关键点 ----
        key_points = self.kps_generator(anchor, instance_feature)
        # (B, num_anchor, num_pts, 3)

        # ---- Step 2: 确定帧数，拆分特征图与投影矩阵 ----
        total_cams = multi_stride_features[0].shape[1]
        num_frames = total_cams // self.num_cams

        # 仅使用最大特征图 (第一个 FPN level)
        largest_fm = multi_stride_features[0]  # (B, total_cams, C, H, W)
        fm_per_frame = self._split_by_frame(largest_fm, self.num_cams, num_frames)
        # fm_per_frame[t]: (B, num_cams, C, H, W)

        proj_per_frame = self._split_by_frame(
            metas["projection_mat"], self.num_cams, num_frames, dim=1
        )
        # proj_per_frame[t]: (B, num_cams, 4, 4)

        wh_per_frame = self._split_by_frame(
            metas.get("image_wh"), self.num_cams, num_frames, dim=1
        )
        # wh_per_frame[t]: (B, num_cams, 2)

        # ---- Step 3: 计算注意力权重 ----
        weights, weight_mask = self._get_weights(
            instance_feature, anchor_embed, num_frames, metas
        )
        # weights:     (B, num_anchor, num_cams, num_frames, num_pts, num_groups)
        # weight_mask: (B, num_anchor, num_cams, num_frames, num_pts, num_groups)

        # ---- Step 4: 逐帧投影 + 采样 ----
        per_frame_feats = []
        per_frame_masks = []
        for t in range(num_frames):
            points_2d, mask = self.project_points(
                key_points,
                proj_per_frame[t],
                wh_per_frame[t],
            )
            # points_2d: (B, num_cams, num_anchor, num_pts, 2) 归一化到 [0, 1]
            # mask:      (B, num_cams, num_anchor, num_pts)    可见性

            # grid_sample: 坐标映射到 [-1, 1]
            points_2d_norm = points_2d * 2 - 1
            points_2d_norm = points_2d_norm.flatten(end_dim=1)
            # (B * num_cams, num_anchor, num_pts, 2)

            sampled = torch.nn.functional.grid_sample(
                fm_per_frame[t].flatten(end_dim=1),
                points_2d_norm,
                align_corners=False,
            )
            # (B * num_cams, C, num_anchor, num_pts)
            sampled = sampled.reshape(bs, self.num_cams, -1, num_anchor, self.num_pts)
            sampled = sampled.permute(0, 3, 4, 1, 2)
            # (B, num_anchor, num_pts, num_cams, C)

            per_frame_feats.append(sampled)
            per_frame_masks.append(mask)

        # ---- Step 5: 跨帧融合 ----
        # 堆叠所有帧: (B, num_anchor, num_pts, num_cams, num_frames, C)
        stacked_feats = torch.stack(per_frame_feats, dim=4)
        # 堆叠 mask:   (B, num_cams, num_anchor, num_pts, num_frames)
        stacked_masks = torch.stack(per_frame_masks, dim=4)

        # 转换维度以匹配权重:
        # feats:  (B, num_anchor, num_cams, num_frames, num_pts, C)
        # masks:  (B, num_anchor, num_pts, num_cams, num_frames) → (B, NA, NC, NF, NP)
        # weights: (B, num_anchor, num_cams, num_frames, num_pts, num_groups)
        stacked_feats = stacked_feats.permute(0, 1, 3, 5, 2, 4).contiguous()
        # (B, num_anchor, num_cams, num_frames, num_pts, C)

        visibility_mask = stacked_masks.permute(0, 2, 1, 4, 3)
        # (B, num_anchor, num_cams, num_frames, num_pts)

        # 组合可见性 mask + dropout mask
        combined_mask = (
            visibility_mask[..., None] & weight_mask
        )
        # (B, num_anchor, num_cams, num_frames, num_pts, num_groups)

        # 掩码处理
        weights[~combined_mask] = -torch.inf

        # Flatten camera+frame dimensions for softmax
        # (B, NA, NC, NF, NP, NG) → (B, NA, NC*NF, NP, NG)
        weights_flat = weights.flatten(2, 3)

        # 检测每个 (anchor, point, group) 是否在所有 camera+frame 上都不可见
        # 若是则 softmax 会产生 NaN，需要特殊处理
        all_inf_mask = torch.all(
            torch.isinf(weights_flat) & (weights_flat < 0), dim=2, keepdim=True
        )  # (B, NA, 1, NP, NG)
        # 将全 inf 的 slice 置零（softmax 前，确保 softmax 产生均匀分布而非 NaN）
        weights_flat.masked_fill_(
            all_inf_mask.expand(-1, -1, weights_flat.shape[2], -1, -1), 0.0
        )

        # Softmax over (num_cams × num_frames) 维度
        weights_flat = weights_flat.softmax(dim=2)

        # 全不可见的 slice 权重置零（softmax 对全零输入输出均匀分布 1/(NC*NF)）
        weights_flat = weights_flat * (~all_inf_mask).float()

        weights = weights_flat.reshape(
            bs, num_anchor, self.num_cams, num_frames, self.num_pts, self.num_groups
        )

        # 分组特征
        grouped_feats = stacked_feats.reshape(
            bs, num_anchor, self.num_cams, num_frames, self.num_pts, self.num_groups, self.group_dims
        )

        # 加权融合: sum over cameras + frames
        fused = (weights.unsqueeze(-1) * grouped_feats).sum(dim=2).sum(dim=2)
        # (B, num_anchor, num_pts, num_groups, group_dims)

        fused = fused.reshape(bs, num_anchor, self.num_pts, self.embed_dims)


        # 融合多采样点
        features = fused.sum(dim=2)
        # (B, num_anchor, embed_dims)

        # ---- Step 6: 输出投影 + residual ----
        output = self.proj_drop(self.output_proj(features))
        output = output + instance_feature
        if output.isnan().any():
            import pdb
            pdb.set_trace()

        return output

    @staticmethod
    def _split_by_frame(tensor, num_cams, num_frames, dim=1):
        """将合并的帧数据按帧拆分。

        Args:
            tensor: (B, total_cams, ...) 合并后的张量
            num_cams: 每帧相机数
            num_frames: 帧数
            dim: camera 所在的维度

        Returns:
            list of tensors, each (B, num_cams, ...)
        """
        if tensor is None:
            return [None] * num_frames
        chunks = tensor.chunk(num_frames, dim=dim)
        return list(chunks)

    def _get_weights(self, instance_feature, anchor_embed, num_frames, metas=None):
        """生成单帧注意力权重并广播到所有帧。

        权重生成遵循 DeformableFeatureAggregation 的模式：
        - 基于 instance_feature + anchor_embed（可选 + camera_embed）
        - 输出单帧权重后广播到 num_frames 帧
        - 最终由 softmax 在 (camera, frame) 上分配注意力

        Returns:
            weights: (B, num_anchor, num_cams, num_frames, num_pts, num_groups)
            mask:    (B, num_anchor, num_cams, num_frames, num_pts, num_groups)
        """
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed

        if self.camera_encoder is not None and metas is not None:
            # 取当前帧（第一个 num_cams 个）的投影矩阵编码相机位置
            cur_proj = metas["projection_mat"][:, :self.num_cams]
            camera_embed = self.camera_encoder(
                cur_proj[:, :, :3].reshape(bs, self.num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]
            # 仅用 num_cams 维度广播 → 生成 (B, NA, NC, NP, NG)
            # 插入 num_frames(=1) 维度以对齐 6D 结构
            weights = (
                self.weights_fc(feature)
                .reshape(bs, num_anchor, self.num_cams, self.num_pts, self.num_groups)
                .unsqueeze(3)
            )
        else:
            weights = (
                self.weights_fc(feature)
                .reshape(bs, num_anchor, -1, self.num_groups)
                .reshape(
                    bs, num_anchor, self.num_cams, 1, self.num_pts, self.num_groups
                )
            )

        # 广播到所有帧
        weights = weights.expand(
            bs, num_anchor, self.num_cams, num_frames, self.num_pts, self.num_groups
        ).clone()

        if self.training and self.attn_drop > 0:
            mask = torch.rand_like(weights) > self.attn_drop
        else:
            mask = torch.ones_like(weights) > 0
        return weights, mask

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """将 3D 关键点投影到 2D 图像平面。

        Args:
            key_points: (B, num_anchor, num_pts, 3)  世界坐标
            projection_mat: (B, num_cams, 4, 4)      lidar2img 矩阵
            image_wh: (B, num_cams, 2)               图像宽高

        Returns:
            points_2d: (B, num_cams, num_anchor, num_pts, 2)  归一化坐标 [0, 1]
            mask:      (B, num_cams, num_anchor, num_pts)     可见性
        """
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        mask = (
            (depth > 1e-5)
            & (points_2d[..., 0] > 0)
            & (points_2d[..., 0] < 1)
            & (points_2d[..., 1] > 0)
            & (points_2d[..., 1] < 1)
        )
        return points_2d, mask
