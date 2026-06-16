import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from torch_scatter import scatter_mean

from LitePT.litept import LitePT_Ours

from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class LitePTGaussianLearner(BaseLifter):
    def __init__(
        self,
        embed_dims,
        num_anchor=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        pc_range=(-50, -50, -5, 50, 50, 3),
        voxel_size=0.5,
        litept_grid_size=0.0625,
        output_grid_size=0.5,
        align_mode="crop_to_min",
        min_output_points=1,
        litept=None,
        **kwargs,
    ):
        super().__init__()
        if abs(litept_grid_size * 8 - output_grid_size) >= 1e-6:
            raise ValueError(
                "LitePTGaussianLearner expects output_grid_size == litept_grid_size * 8"
            )
        if align_mode not in ("crop_to_min", "pad_to_max", "fixed_topk"):
            raise ValueError(f"Unsupported align_mode: {align_mode}")

        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.input_grid_size = litept_grid_size
        self.output_grid_size = output_grid_size
        self.align_mode = align_mode
        self.min_output_points = min_output_points
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.semantics = semantics
        self.semantic_dim = semantic_dim if semantics else 0
        if semantics:
            assert semantic_dim is not None

        litept_cfg = dict(litept or {})
        self.litept = LitePT_Ours(
            in_channels=litept_cfg.pop("in_channels", 4),
            preset=litept_cfg.pop("preset", "litept_base"),
            input_grid_size=litept_grid_size,
            target_grid_size=output_grid_size,
            num_decoders=litept_cfg.pop("num_decoders", 1),
            decoder_block_depths=litept_cfg.pop("decoder_block_depths", (0,)),
            **litept_cfg,
        )

        self.feature_proj = nn.Linear(self.litept.out_channels, embed_dims)
        self.scale_learner = nn.Linear(embed_dims, 3)
        self.rot_learner = nn.Linear(embed_dims, 4)
        if include_opa:
            self.opa_learner = nn.Linear(embed_dims, 1)
        if semantics:
            self.semantic_learner = nn.Linear(embed_dims, semantic_dim)

        gaussian_dim = 10 + int(include_opa) + self.semantic_dim
        self.fallback_feature = nn.Parameter(torch.zeros(min_output_points, embed_dims))
        self.fallback_anchor = nn.Parameter(torch.zeros(min_output_points, gaussian_dim))
        with torch.no_grad():
            self.fallback_anchor[:, 6] = 1.0

        pc_range_tensor = torch.tensor(pc_range, dtype=torch.float32)
        self.register_buffer("pc_range_tensor", pc_range_tensor)

        with torch.no_grad():
            self.rot_learner.bias.zero_()
            self.rot_learner.bias[0] = 1.0

    def init_weights(self):
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)
        nn.init.xavier_uniform_(self.scale_learner.weight)
        nn.init.zeros_(self.scale_learner.bias)
        nn.init.xavier_uniform_(self.rot_learner.weight)
        nn.init.zeros_(self.rot_learner.bias)
        with torch.no_grad():
            self.rot_learner.bias[0] = 1.0
        if self.include_opa:
            nn.init.xavier_uniform_(self.opa_learner.weight)
            nn.init.zeros_(self.opa_learner.bias)
        if self.semantics:
            nn.init.xavier_uniform_(self.semantic_learner.weight)
            nn.init.zeros_(self.semantic_learner.bias)

    def _fallback_lidar(self, device, dtype):
        pc_min = self.pc_range_tensor[:3].to(device=device, dtype=dtype)
        pc_max = self.pc_range_tensor[3:].to(device=device, dtype=dtype)
        center = (pc_min + pc_max) * 0.5
        return torch.cat([center, center.new_zeros(1)], dim=0).reshape(1, 4)

    def build_litept_input(self, metas):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        pc_min = self.pc_range_tensor[:3].to(device=device, dtype=dtype)
        pc_max = self.pc_range_tensor[3:].to(device=device, dtype=dtype)

        all_coord = []
        all_grid_coord = []
        all_feat = []
        all_batch = []

        for b, pts in enumerate(metas["lidar_points"]):
            pts = pts.to(device=device, dtype=dtype)
            if pts.numel() == 0:
                pts = self._fallback_lidar(device, dtype)
            xyz = pts[:, :3]
            intensity = pts[:, 3:4] if pts.shape[-1] > 3 else xyz.new_zeros(xyz.shape[0], 1)

            mask = ((xyz >= pc_min) & (xyz < pc_max)).all(dim=-1)
            xyz = xyz[mask]
            intensity = intensity[mask]
            if xyz.shape[0] == 0:
                pts = self._fallback_lidar(device, dtype)
                xyz = pts[:, :3]
                intensity = pts[:, 3:4]

            grid_coord = torch.floor((xyz - pc_min) / self.input_grid_size).to(torch.int32)
            feat = torch.cat([xyz, intensity], dim=-1)

            unique_grid, inverse = torch.unique(
                grid_coord, sorted=True, return_inverse=True, dim=0
            )
            coord = scatter_mean(xyz, inverse, dim=0)
            voxel_feat = scatter_mean(feat, inverse, dim=0)
            batch = torch.full(
                (unique_grid.shape[0],), b, device=device, dtype=torch.long
            )

            all_coord.append(coord)
            all_grid_coord.append(unique_grid)
            all_feat.append(voxel_feat)
            all_batch.append(batch)

        return dict(
            coord=torch.cat(all_coord, dim=0),
            grid_coord=torch.cat(all_grid_coord, dim=0),
            feat=torch.cat(all_feat, dim=0),
            batch=torch.cat(all_batch, dim=0),
        )

    def decode_anchor(self, coord, feature):
        pc_min = self.pc_range_tensor[:3].to(device=coord.device, dtype=coord.dtype)
        pc_max = self.pc_range_tensor[3:].to(device=coord.device, dtype=coord.dtype)
        xyz = (coord - pc_min) / (pc_max - pc_min)
        xyz = xyz.clamp(1e-6, 1 - 1e-6)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)

        scale = torch.sigmoid(self.scale_learner(feature)).clamp(1e-6, 1 - 1e-6)
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rot = F.normalize(self.rot_learner(feature), dim=-1)
        if self.include_opa:
            opacity = torch.sigmoid(self.opa_learner(feature))
        else:
            opacity = feature.new_zeros(feature.shape[0], 0)

        if self.semantics:
            semantic = self.semantic_learner(feature)
        else:
            semantic = feature.new_zeros(feature.shape[0], 0)

        return torch.cat([xyz, scale, rot, opacity, semantic], dim=-1)

    def _select_indices(self, feature, anchor, target_count):
        count = feature.shape[0]
        if count == 0:
            return None
        if count >= target_count:
            score = feature.norm(dim=-1)
            return torch.topk(score, k=target_count, largest=True, sorted=False).indices
        repeat = torch.arange(target_count, device=feature.device) % count
        return repeat

    def align_as_dense_batch(self, feature, anchor, batch, batch_size):
        counts = torch.bincount(batch, minlength=batch_size)
        if self.align_mode == "pad_to_max":
            target_count = int(counts.max().item())
        elif self.align_mode == "fixed_topk" and self.num_anchor is not None:
            target_count = int(self.num_anchor)
        else:
            target_count = int(counts.min().item())
        target_count = max(target_count, int(self.min_output_points))

        features_list = []
        anchors_list = []
        for b in range(batch_size):
            mask = batch == b
            cur_feat = feature[mask]
            cur_anchor = anchor[mask]
            indices = self._select_indices(cur_feat, cur_anchor, target_count)
            if indices is None:
                fb_idx = torch.arange(target_count, device=feature.device) % self.min_output_points
                features_list.append(self.fallback_feature[fb_idx])
                anchors_list.append(self.fallback_anchor[fb_idx])
            else:
                features_list.append(cur_feat[indices])
                anchors_list.append(cur_anchor[indices])

        return torch.stack(features_list, dim=0), torch.stack(anchors_list, dim=0)

    def forward(self, imgs=None, metas=None, **kwargs):
        batch_size = len(metas["lidar_points"])
        data_dict = self.build_litept_input(metas)
        point = self.litept(data_dict)

        feature = self.feature_proj(point.feat)
        anchor = self.decode_anchor(point.coord, feature)
        feature, anchor = self.align_as_dense_batch(
            feature, anchor, point.batch.long(), batch_size
        )

        return {
            "rep_features": feature,
            "representation": anchor,
            "anchor_init": anchor.clone(),
        }
