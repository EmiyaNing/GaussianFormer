"""Per-foreground-voxel Gaussian support counts and normalized positions."""
from functools import lru_cache
from pathlib import Path
import os

COUNT_NAMES = ('native', 'within_1sigma', 'within_2sigma', 'within_3sigma',
               'positive_winning_class', 'at_least_1pct', 'at_least_10pct')
MASS_NAMES = ('kernel', 'opacity_kernel', 'winning_class', 'squared_shares',
              'kernel_distance', 'contribution_distance', 'contribution_euclidean',
              'contribution_0to1', 'contribution_1to2', 'contribution_2to3', 'contribution_over3')


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    return load(name='gaussian_voxel_coverage_v1',
                sources=[str(Path(__file__).with_name('kernels.cu'))],
                extra_cuda_cflags=['-O3'], verbose=False)
