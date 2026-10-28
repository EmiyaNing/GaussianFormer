"""OPUS Gaussian perception with SparseWorld timestamp-aware attention.

The public model ID intentionally uses ``sparse_opus_gausisan_world``. Module
and parameter layouts match the completed D experiment, including the
nonpersistent timestamp mask and the final-stage Gaussian attribute branch.
"""

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from .opus_encoder import _StrictOPUSSelfAttention, _opus_decode_points
from .opus_v1_gaussian_temporal_encoder import OPUSV1GaussianTemporalEncoder


class TimestampSelfAttention(_StrictOPUSSelfAttention):
    """Combine OPUS spatial attention with SparseWorld's causal time mask."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("ind_mask", None, persistent=False)

    def forward(self, query_points, query_features):
        # The attention distance uses detached proposals, as in OPUS-V1.
        centers = _opus_decode_points(query_points.detach(), self.pc_range).mean(2)
        distances = torch.cdist(centers, centers)
        temperatures = self.gen_tau(query_features).permute(0, 2, 1)
        attention_mask = -distances[:, None] * temperatures[..., None]
        if self.ind_mask is not None:
            attention_mask = (
                attention_mask + self.ind_mask.to(attention_mask)[None, None]
            )

        sequence = query_features.transpose(0, 1)
        attention = self.attention
        update, _ = F.multi_head_attention_forward(
            sequence,
            sequence,
            sequence,
            attention.embed_dim,
            attention.num_heads,
            attention.in_proj_weight,
            attention.in_proj_bias,
            attention.bias_k,
            attention.bias_v,
            attention.add_zero_attn,
            attention.dropout,
            attention.out_proj.weight,
            attention.out_proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=attention_mask.flatten(0, 1),
            use_separate_proj_weight=False,
            average_attn_weights=True,
            is_causal=False,
        )
        return query_features + update.transpose(0, 1)


@MODELS.register_module()
class SparseOpusGaussianEncoder(OPUSV1GaussianTemporalEncoder):
    """Keep raw point logits for set loss and softplus logits for rendering.

    The inherited decoder bounds scales to the configured interval, predicts
    Gaussian attributes at the requested stages, and checkpoints decoder
    activations during training. Invalid child points have zero opacity.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        num_refines=(1, 4, 16, 32, 64, 128),
        **kwargs
    ):
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_refines=num_refines,
            **kwargs
        )
        self.embed_dims = embed_dims
        self.num_refines = tuple(num_refines)
        for layer in self.layers:
            layer.self_attn = TimestampSelfAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                pc_range=self.pc_range,
            )

    @property
    def decoder_layers(self):
        """Expose the interface used by SparseWorld timestamp assignment."""
        return self.layers

    def _make_gaussian(self, features, points, logits, valid, stage):
        gaussian = super()._make_gaussian(features, points, logits, valid, stage)
        point_valid = torch.isfinite(points).all(-1)
        point_valid &= ((points >= 0) & (points < 1)).all(-1)
        return gaussian._replace(
            opacities=gaussian.opacities * point_valid[..., None].to(points.dtype)
        )


@MODELS.register_module()
class SparseOpusGaussianTransformer(BaseModule):
    """Adapt grouped Gaussian decoder states to SparseWorld's head interface."""

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
            query_features=query,
            query_points=points.squeeze(2),
            ms_img_feats=features,
            metas=dict(projection_mat=projection, image_wh=image_wh),
        )["representation"]
        return (
            states[-1]["query_features"],
            [state["opus_logits"] for state in states],
            [state["query_points"] for state in states],
            states,
        )
