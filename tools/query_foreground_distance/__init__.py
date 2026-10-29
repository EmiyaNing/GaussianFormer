"""Exact Gaussian-center distance to foreground voxel centers and solids."""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    return load(name='query_foreground_distance_v1',
                sources=[str(Path(__file__).with_name('kernels.cu'))],
                extra_cuda_cflags=['-O3'], verbose=False)
