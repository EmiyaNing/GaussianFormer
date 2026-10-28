"""Gaussian OPUS with SparseWorld's timestamp attention contract."""
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from model.encoder.opus_encoder import _StrictOPUSSelfAttention, _opus_decode_points
from model.encoder.opus_v1_gaussian_temporal_encoder import OPUSV1GaussianTemporalEncoder
from model.head.sparseworld_strict_head import SparseWorldStrictHead


class TimestampSelfAttention(_StrictOPUSSelfAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('ind_mask', None, persistent=False)

    def forward(self, query_points, query_features):
        world = _opus_decode_points(query_points.detach(), self.pc_range).mean(2)
        distance = torch.cdist(world, world)
        tau = self.gen_tau(query_features).permute(0, 2, 1)
        mask = -distance[:, None] * tau[..., None]
        if self.ind_mask is not None:
            mask = mask + self.ind_mask.to(mask)[None, None]
        qkv = query_features.transpose(0, 1)
        a = self.attention
        output, _ = F.multi_head_attention_forward(
            qkv, qkv, qkv, a.embed_dim, a.num_heads,
            a.in_proj_weight, a.in_proj_bias, a.bias_k, a.bias_v,
            a.add_zero_attn, a.dropout, a.out_proj.weight, a.out_proj.bias,
            training=self.training, key_padding_mask=None, need_weights=False,
            attn_mask=mask.flatten(0, 1), use_separate_proj_weight=False,
            average_attn_weights=True, is_causal=False)
        return query_features + output.transpose(0, 1)


@MODELS.register_module()
class SparseWorldBiasGaussianEncoder(OPUSV1GaussianTemporalEncoder):
    def __init__(self, embed_dims=256, num_heads=8, dropout=.1,
                 num_refines=(1, 4, 16, 32, 64, 128), **kwargs):
        super().__init__(embed_dims=embed_dims, num_heads=num_heads,
                         dropout=dropout, num_refines=num_refines, **kwargs)
        self.embed_dims = embed_dims
        self.num_refines = tuple(num_refines)
        for layer in self.layers:
            layer.self_attn = TimestampSelfAttention(
                embed_dims=embed_dims, num_heads=num_heads,
                dropout=dropout, pc_range=self.pc_range)

    @property
    def decoder_layers(self):
        return self.layers

    def _make_gaussian(self, features, points, logits, valid, stage):
        gaussian = super()._make_gaussian(features, points, logits, valid, stage)
        point_valid = torch.isfinite(points).all(-1)
        point_valid &= ((points >= 0) & (points < 1)).all(-1)
        return gaussian._replace(
            opacities=gaussian.opacities * point_valid[..., None].to(points.dtype))


@MODELS.register_module()
class SparseWorldBiasGaussianTransformer(BaseModule):
    def __init__(self, encoder, init_cfg=None):
        super().__init__(init_cfg)
        self.encoder = MODELS.build(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.num_refines = self.encoder.num_refines

    @property
    def decoder(self):
        return self.encoder

    def forward(self, points, query, features, projection, image_wh):
        states = self.encoder(
            query_features=query, query_points=points.squeeze(2),
            ms_img_feats=features,
            metas=dict(projection_mat=projection, image_wh=image_wh))['representation']
        return (states[-1]['query_features'],
                [s['opus_logits'] for s in states],
                [s['query_points'] for s in states], states)


@MODELS.register_module()
class SparseWorldBiasGaussianPerceptionHead(SparseWorldStrictHead):
    def forward(self, mlvl_feats, metas, points_scale=None):
        batch = mlvl_feats[0].shape[0]
        initial = self.init_points.weight[None, :, None].expand(batch, -1, -1, -1)
        scale = self.points_scale if points_scale is None else points_scale
        if scale is not None:
            initial = initial * scale.unsqueeze(1)
        query = initial.new_zeros(batch, len(self.init_points.weight), self.embed_dims)
        self.update_stamp_assignment(force=self.pretrain)
        query, scores, points, states = self.transformer(
            initial, query, mlvl_feats,
            metas['projection_mat'].to(initial), metas['image_wh'].to(initial))
        return dict(init_points=initial, all_cls_scores=scores,
                    all_refine_pts=points, query_feat=query, representation=states)

