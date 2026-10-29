"""Exact local mean point distance with detached ellipsoid membership."""
from functools import lru_cache
from pathlib import Path
import os
import torch

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS','2');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    return load(name='semantic_neighborhood_mean_v1',sources=[str(Path(__file__).with_name('kernels.cu'))],
        extra_cuda_cflags=['-O3'],verbose=False)

class _LocalMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,values,direction):
        ctx.save_for_backward(direction)
        return values.clone()
    @staticmethod
    def backward(ctx,upstream):
        (direction,)=ctx.saved_tensors
        return upstream[:,None]*direction,None,None

def local_mean(means,scales,rotation,classes,xyz,labels,lookup,targets,shape,lower,h):
    with torch.no_grad():
        values,direction,counts,core,same_core,core_per_gs=extension().statistics(
            means.detach().contiguous(),scales.detach().contiguous(),rotation.detach().contiguous(),
            classes.contiguous(),xyz.contiguous(),labels.contiguous(),lookup.contiguous(),
            targets,shape,lower,h)
    return _LocalMean.apply(means,values,direction),counts,core,same_core,core_per_gs
