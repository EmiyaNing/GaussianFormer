"""Read-only decoder diagnostics, with production-compatible GS support."""
from functools import lru_cache
from pathlib import Path
import os


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    return load(name='decoder_layer_support_v1',
                sources=[str(Path(__file__).with_name('kernels.cu'))],
                extra_cuda_cflags=['-O3'], verbose=False)
