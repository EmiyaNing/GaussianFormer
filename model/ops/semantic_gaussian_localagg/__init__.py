"""CUDA local aggregation for point-anchored semantic Gaussians.

The extension is optional at import time so CPU-only environments can still
construct Phase-A models.  Phase-B heads check ``CUDA_LOCALAGG_AVAILABLE``
and fail explicitly if this package has not been compiled.
"""
from .wrapper import CUDA_LOCALAGG_AVAILABLE, semantic_gaussian_localagg

__all__ = ['CUDA_LOCALAGG_AVAILABLE', 'semantic_gaussian_localagg']
