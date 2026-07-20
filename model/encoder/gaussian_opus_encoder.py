"""Phase-A Gaussian-OPUS decoder: OPUS sampling with Gaussian outputs."""
import torch
from torch import nn
from mmseg.registry import MODELS
from mmengine.model import BaseModule

from .gaussian_encoder.utils import GaussianPrediction
from .opus_encoder import (_StrictOPUSSampling, _StrictAdaptiveMixing,
                           _StrictOPUSSelfAttention, _opus_encode_points)


def _next_stage_parent(gaussian, cross_stage_geometry_grad=True):
    """Build the parent consumed by the following decoder stage.

    Only centre and scale affect the next stage's position encoding, image
    sampling, and local child parameterisation.  Keeping these two tensors
    attached makes the Gaussian refinement end-to-end: a later-stage loss can
    optimise earlier Gaussian geometry.  Rotation, opacity, and semantics are
    not read by the current Phase-A next-stage computation, so detaching them
    avoids retaining a graph that cannot contribute a cross-stage gradient.
    """
    means = gaussian.means if cross_stage_geometry_grad else gaussian.means.detach()
    scales = gaussian.scales if cross_stage_geometry_grad else gaussian.scales.detach()
    return GaussianPrediction(
        means=means, scales=scales,
        rotations=gaussian.rotations.detach(), opacities=gaussian.opacities.detach(),
        semantics=gaussian.semantics.detach())


class _SemanticGaussianRefinementHead(BaseModule):
    def __init__(self, embed_dims, num_refine, semantic_dim, stage_step,
                 pc_range, scale_min=(.15, .15, .15), scale_max=(.99, .99, .99),
                 position_radius_multiplier=1.1):
        super().__init__()
        self.num_refine = num_refine
        self.stage_step = stage_step
        self.position_radius_multiplier = position_radius_multiplier
        self.register_buffer('scale_min', torch.tensor(scale_min, dtype=torch.float32))
        self.register_buffer('scale_max', torch.tensor(scale_max, dtype=torch.float32))
        self.register_buffer('pc_lower', torch.tensor(pc_range[:3], dtype=torch.float32))
        self.register_buffer('pc_upper', torch.tensor(pc_range[3:], dtype=torch.float32))
        self.geometry = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_refine * 11))
        self.semantic = nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_refine * semantic_dim))
        nn.init.zeros_(self.geometry[-1].weight)
        nn.init.zeros_(self.geometry[-1].bias)
        with torch.no_grad():
            self.geometry[-1].bias.view(num_refine, 11)[:, 6] = 1.
            self.geometry[-1].bias.view(num_refine, 11)[:, 10] = torch.logit(torch.tensor(.1))
            nn.init.constant_(self.semantic[-1].bias, -4.59511985013459)

    def forward(self, features, parent, query_valid_mask):
        batch, queries, _ = features.shape
        geometry = self.geometry(features).view(batch, queries, self.num_refine, 11)
        semantics = self.semantic(features).view(batch, queries, self.num_refine, -1)
        mean = parent.means.mean(dim=2, keepdim=True)
        scale = parent.scales.mean(dim=2, keepdim=True).clamp_min(1e-4)
        # Stage 0 places children inside (or at most 10% outside) the large
        # template ellipsoid.  Later stages use already-small parent scales,
        # so their refinements remain local as well.
        means = mean + (self.stage_step * self.position_radius_multiplier * scale *
                        torch.tanh(geometry[..., :3]))
        # local_aggregate requires every Gaussian mean to map into the voxel grid.
        means = torch.maximum(torch.minimum(means, self.pc_upper.to(means) - 1e-4),
                              self.pc_lower.to(means) + 1e-4)
        scales = torch.exp(scale.log() + torch.log(torch.tensor(1.6, device=scale.device,
                                                                 dtype=scale.dtype)) *
                           torch.tanh(geometry[..., 3:6]))
        scales = scales.clamp(self.scale_min.to(scales), self.scale_max.to(scales))
        rotations = torch.nn.functional.normalize(geometry[..., 6:10], dim=-1, eps=1e-6)
        opacities = geometry[..., 10:11].sigmoid()
        valid = query_valid_mask[:, :, None, None].to(opacities.dtype)
        # Keep invalid-query centers in range: local_aggregate validates every
        # center before opacity is considered, while alpha=0 removes influence.
        return GaussianPrediction(means, scales, rotations, opacities * valid,
                                  semantics * valid)


class _GaussianOPUSDecoderLayer(BaseModule):
    def __init__(self, embed_dims, num_frames, num_views, num_points, num_levels,
                 num_groups, num_heads, feedforward_channels, dropout, last_refine,
                 num_refine, semantic_dim, stage_step, pc_range, child_scale_range,
                 position_radius_multiplier):
        super().__init__()
        self.pc_range = tuple(pc_range)
        self.position_encoder = nn.Sequential(
            nn.Linear(3 * last_refine, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.sampling = _StrictOPUSSampling(embed_dims, num_frames, num_views, num_groups,
                                            num_points, num_levels, pc_range)
        self.mixing = _StrictAdaptiveMixing(embed_dims, num_frames * num_points, num_groups, 32)
        self.self_attn = _StrictOPUSSelfAttention(embed_dims, num_heads, dropout, pc_range)
        self.ffn = nn.Sequential(nn.Linear(embed_dims, feedforward_channels), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(feedforward_channels, embed_dims),
                                 nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.refine = _SemanticGaussianRefinementHead(
            embed_dims, num_refine, semantic_dim, stage_step, pc_range,
            scale_min=child_scale_range[0], scale_max=child_scale_range[1],
            position_radius_multiplier=position_radius_multiplier)

    def forward(self, parent, query_features, mlvl_feats, metas, query_valid_mask):
        parent_points = _opus_encode_points(parent.means, self.pc_range)
        query_features = query_features + self.position_encoder(parent_points.flatten(2))
        sampled, visible = self.sampling(parent_points, query_features, mlvl_feats, metas)
        query_features = self.norm1(self.mixing(sampled, query_features))
        query_features = self.norm2(self.self_attn(parent_points, query_features))
        query_features = self.norm3(query_features + self.ffn(query_features))
        gaussian = self.refine(query_features, parent, query_valid_mask)
        query_features = query_features * query_valid_mask.unsqueeze(-1).to(query_features.dtype)
        return query_features, gaussian, visible


@MODELS.register_module()
class GaussianOPUSEncoder(BaseModule):
    """Phase-A encoder; image sampling sees parent means only."""
    def __init__(self, embed_dims=256, num_decoder=6, num_frames=1, num_views=6,
                 num_points=4, num_levels=4, num_groups=4, num_heads=8,
                 feedforward_channels=512, dropout=.1, semantic_dim=17,
                 num_refines=(1, 4, 16, 32, 64, 128),
                 stage_steps=(1., .75, .5, .35, .25, .2),
                 child_scale_range=((.15, .15, .15), (.99, .99, .99)),
                 position_radius_multiplier=1.1,
                 cross_stage_geometry_grad=True,
                 pc_range=(-40., -40., -1., 40., 40., 5.4), init_cfg=None):
        super().__init__(init_cfg)
        if len(num_refines) != num_decoder or len(stage_steps) != num_decoder:
            raise ValueError('num_refines and stage_steps must have one value per decoder layer')
        if max(child_scale_range[1]) >= 1.0:
            raise ValueError('Phase-A decoded Gaussian scales must be strictly smaller than 1.0m')
        previous = (1,) + tuple(num_refines[:-1])
        self.num_frames, self.num_views, self.num_groups = num_frames, num_views, num_groups
        self.cross_stage_geometry_grad = cross_stage_geometry_grad
        self.layers = nn.ModuleList([
            _GaussianOPUSDecoderLayer(embed_dims, num_frames, num_views, num_points, num_levels,
                                      num_groups, num_heads, feedforward_channels, dropout, last, current,
                                      semantic_dim, step, pc_range, child_scale_range,
                                      position_radius_multiplier)
            for last, current, step in zip(previous, num_refines, stage_steps)
        ])

    @staticmethod
    def _group_features(ms_img_feats, batch, frames, views, groups):
        grouped = []
        for feature in ms_img_feats:
            _, cameras, channels, height, width = feature.shape
            if cameras != frames * views or channels % groups:
                raise ValueError('Gaussian-OPUS feature shape is incompatible with strict OPUS sampling')
            grouped.append(feature.reshape(batch, frames, views, groups, channels // groups, height, width)
                           .permute(0, 1, 3, 2, 5, 6, 4).reshape(
                               batch * frames * groups, views, height, width, channels // groups).contiguous())
        return grouped

    def forward(self, query_features, query_templates, ms_img_feats, metas,
                query_valid_mask=None, **kwargs):
        if ms_img_feats[0].shape[1] != self.num_frames * self.num_views:
            raise ValueError('feature camera dimension must equal num_frames * num_views')
        batch, queries = query_features.shape[:2]
        if query_valid_mask is None:
            query_valid_mask = torch.ones(batch, queries, device=query_features.device, dtype=torch.bool)
        # GaussianPrediction also carries optional debug fields whose values
        # default to None; only the five renderer attributes are grouped.
        parent = GaussianPrediction(
            means=query_templates.means.unsqueeze(2),
            scales=query_templates.scales.unsqueeze(2),
            rotations=query_templates.rotations.unsqueeze(2),
            opacities=query_templates.opacities.unsqueeze(2),
            semantics=query_templates.semantics.unsqueeze(2))
        grouped_features = self._group_features(ms_img_feats, batch, self.num_frames,
                                                self.num_views, self.num_groups)
        representation = []
        for layer in self.layers:
            query_features, gaussian, visible = layer(parent, query_features, grouped_features,
                                                       metas, query_valid_mask)
            representation.append({'query_features': query_features, 'gaussian': gaussian,
                                   'query_valid_mask': query_valid_mask, 'visible': visible})
            parent = _next_stage_parent(
                gaussian, cross_stage_geometry_grad=self.cross_stage_geometry_grad)
        return {'representation': representation}
