"""
LitePT variant for GaussianFormer.

This module keeps the original LitePT building blocks, but lets callers stop
after a configurable number of decoder modules instead of always running the
full decoder.
"""

from functools import partial
from math import prod
from pathlib import Path
import sys

import torch
import torch.nn as nn


_LITEPT_ROOT = Path(__file__).resolve().parents[1]
if str(_LITEPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_LITEPT_ROOT))

from .model import (  # noqa: E402
    Block,
    Embedding,
    GridPooling,
    GridUnpooling,
    Point,
    PointModule,
    PointSequential,
)


def _litept_base_preset():
    return dict(
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
    )


def build_litept_preset(preset, decoder_block_depths, num_decoders, **overrides):
    if preset != "litept_base":
        raise ValueError(f"Unsupported LitePT_Ours preset: {preset}")

    cfg = _litept_base_preset()
    cfg.update(overrides)

    num_stages = len(cfg["enc_depths"])
    num_decoder_stages = num_stages - 1
    if not 0 <= num_decoders <= num_decoder_stages:
        raise ValueError(
            f"num_decoders must be in [0, {num_decoder_stages}], got {num_decoders}"
        )

    if isinstance(decoder_block_depths, int):
        decoder_block_depths = (decoder_block_depths,) * num_decoders
    if len(decoder_block_depths) != num_decoders:
        raise ValueError(
            "decoder_block_depths must have one entry per executed decoder module"
        )

    dec_depths = [0] * num_decoder_stages
    for i, depth in enumerate(decoder_block_depths):
        stage_idx = num_decoder_stages - 1 - i
        dec_depths[stage_idx] = depth
    cfg["dec_depths"] = tuple(dec_depths)
    return cfg


def compute_out_stride(stride, num_decoders):
    encoder_stride = prod(stride)
    decoder_factor = 2**num_decoders
    if encoder_stride % decoder_factor != 0:
        raise ValueError(
            f"encoder stride {encoder_stride} is not divisible by decoder factor {decoder_factor}"
        )
    return encoder_stride // decoder_factor


def resolve_out_channels(cfg, num_decoders):
    if num_decoders == 0:
        return cfg["enc_channels"][-1]
    stage_idx = len(cfg["dec_channels"]) - num_decoders
    return cfg["dec_channels"][stage_idx]


class LitePT_Ours(PointModule):
    def __init__(
        self,
        in_channels=4,
        preset="litept_base",
        input_grid_size=0.0625,
        target_grid_size=0.5,
        num_decoders=1,
        decoder_block_depths=(0,),
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        **kwargs,
    ):
        super().__init__()
        cfg = build_litept_preset(
            preset,
            decoder_block_depths=decoder_block_depths,
            num_decoders=num_decoders,
            **kwargs,
        )

        self.num_stages = len(cfg["enc_depths"])
        self.num_decoders = num_decoders
        self.order = [order] if isinstance(order, str) else order
        self.shuffle_orders = shuffle_orders
        self.enc_conv = cfg["enc_conv"]
        self.enc_attn = cfg["enc_attn"]
        self.dec_conv = cfg["dec_conv"]
        self.dec_attn = cfg["dec_attn"]

        self.input_grid_size = input_grid_size
        self.out_stride = compute_out_stride(cfg["stride"], num_decoders)
        self.out_grid_size = input_grid_size * self.out_stride
        if abs(self.out_grid_size - target_grid_size) >= 1e-6:
            raise ValueError(
                f"LitePT_Ours output grid size {self.out_grid_size} does not match "
                f"target_grid_size {target_grid_size}"
            )
        self.out_channels = resolve_out_channels(cfg, num_decoders)

        assert self.num_stages == len(cfg["stride"]) + 1
        assert self.num_stages == len(cfg["enc_channels"])
        assert self.num_stages == len(cfg["enc_num_head"])
        assert self.num_stages == len(cfg["enc_patch_size"])
        assert self.num_stages == len(cfg["dec_depths"]) + 1
        assert self.num_stages == len(cfg["dec_channels"]) + 1
        assert self.num_stages == len(cfg["dec_num_head"]) + 1
        assert self.num_stages == len(cfg["dec_patch_size"]) + 1

        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=cfg["enc_channels"][0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(cfg["enc_depths"]))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(cfg["enc_depths"][:s]) : sum(cfg["enc_depths"][: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=cfg["enc_channels"][s - 1],
                        out_channels=cfg["enc_channels"][s],
                        stride=cfg["stride"][s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=cfg["enc_attn"][s],
                        serialization_order=self.order,
                    ),
                    name="down",
                )
            for i in range(cfg["enc_depths"][s]):
                enc.add(
                    Block(
                        channels=cfg["enc_channels"][s],
                        num_heads=cfg["enc_num_head"][s],
                        patch_size=cfg["enc_patch_size"][s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=cfg["enc_conv"][s],
                        enable_attn=cfg["enc_attn"][s],
                        rope_freq=cfg["enc_rope_freq"][s],
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(cfg["dec_depths"]))
        ]
        self.dec = PointSequential()
        dec_channels = list(cfg["dec_channels"]) + [cfg["enc_channels"][-1]]
        for s in reversed(range(self.num_stages - 1)):
            dec_drop_path_ = dec_drop_path[
                sum(cfg["dec_depths"][:s]) : sum(cfg["dec_depths"][: s + 1])
            ]
            dec_drop_path_.reverse()
            dec = PointSequential()
            dec.add(
                GridUnpooling(
                    in_channels=dec_channels[s + 1],
                    skip_channels=cfg["enc_channels"][s],
                    out_channels=dec_channels[s],
                    norm_layer=bn_layer,
                    act_layer=act_layer,
                ),
                name="up",
            )
            for i in range(cfg["dec_depths"][s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=cfg["dec_num_head"][s],
                        patch_size=cfg["dec_patch_size"][s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=cfg["dec_conv"][s],
                        enable_attn=cfg["dec_attn"][s],
                        rope_freq=cfg["dec_rope_freq"][s],
                    ),
                    name=f"block{i}",
                )
            self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        for i in range(self.num_decoders):
            point = self.dec[i](point)
        return point
