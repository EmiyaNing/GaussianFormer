"""Optional temporal kernels; legacy configurations need no new extension."""

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def extension():
    try:
        return import_module(f'{__name__}.temporal_sampling_ext')
    except ImportError as exc:
        raise RuntimeError(
            'Build temporal sampling: cd model/encoder/gaussian_encoder/temporal_ops '
            '&& MAX_JOBS=2 python3.10 setup.py build_ext --inplace') from exc
