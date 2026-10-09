"""S2GO-style 4x4x4 block dense G2V (forward only).

A separate extension on purpose: it must not touch
``model/ops/semantic_gaussian_localagg``, which other experiments depend on.
"""
from .wrapper import block_g2v_forward, tile_edge, CUDA_BLOCK_G2V_AVAILABLE

__all__ = ['block_g2v_forward', 'tile_edge', 'CUDA_BLOCK_G2V_AVAILABLE']
