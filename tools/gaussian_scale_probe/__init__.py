"""Full-data frozen-model scale and supervision diagnostics."""
from functools import lru_cache
from pathlib import Path
import os

SOURCE_NAMES=('ce_empty','ce_foreground','lovasz')
MOMENT_NAMES=('support_voxels','camera_voxels','camera_foreground_voxels','camera_empty_voxels',
              'camera_fp_voxels','kernel_mass','camera_kernel_mass','camera_fg_kernel_mass',
              'camera_empty_kernel_mass','camera_fp_kernel_mass')
GRAD_NAMES=('opacity','scale_x','scale_y','scale_z','mean_x','mean_y','mean_z','assigned_raw_logit','log_amplitude')


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS','2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    return load(name='gaussian_scale_supervision_v1',sources=[str(Path(__file__).with_name('kernels.cu'))],
                extra_cuda_cflags=['-O3'],verbose=False)
