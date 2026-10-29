"""Boundary Gaussian diagnostics, separate from every model/training operator."""
from functools import lru_cache
from pathlib import Path
import os

SCOPES = ('native_all', 'native_camera', 'core3sigma_all', 'core3sigma_camera')
COHORTS = ('all', 'union_any', 'union_foreground', 'union_fg_empty',
           'dominant_any', 'dominant_foreground', 'dominant_fg_empty',
           'dominant_nonboundary', 'unattributed')
OUTCOMES = ('correct_foreground', 'wrong_class', 'miss', 'false_positive', 'true_empty')


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    os.environ.setdefault('MAX_JOBS', '2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    return load(name='boundary_gaussian_statistics_v1',
                sources=[str(Path(__file__).with_name('kernels.cu'))],
                extra_cuda_cflags=['-O3'], verbose=False)


def rates(counts):
    """All fractions are 0..1; undefined conditional denominators produce None."""
    correct, wrong, miss, fp, tn = map(int, counts)
    total, fg, empty = sum(counts), correct+wrong+miss, fp+tn
    safe = lambda x, n: float(x/n) if n else None
    return dict(voxels=int(total), gt_foreground=fg, gt_empty=empty,
        correct_foreground=correct, wrong_class=wrong, miss=miss,
        false_positive=fp, true_empty=tn,
        wrong_class_fraction=safe(wrong,total), miss_fraction=safe(miss,total),
        false_positive_fraction=safe(fp,total), total_error_fraction=safe(wrong+miss+fp,total),
        wrong_class_rate_gt_foreground=safe(wrong,fg), miss_rate_gt_foreground=safe(miss,fg),
        false_positive_rate_gt_empty=safe(fp,empty),
        wrong_class_rate_detected_foreground=safe(wrong,correct+wrong))


def gaussian_counts(bits):
    import torch
    counts=[]
    for labels in bits:
        # 18-bit class membership, independent of CUDA boundary flag construction.
        distinct=sum((labels>>k)&1 for k in range(18))
        foreground=sum((labels>>k)&1 for k in range(17))
        counts.append(dict(predicted=len(labels), supported=int((distinct>0).sum()),
            boundary_any=int((distinct>=2).sum()),
            boundary_foreground=int((foreground>=2).sum()),
            boundary_fg_empty=int(((foreground>0)&((labels&(1<<17))>0)).sum()),
            class_count_histogram=torch.bincount(distinct.long(),minlength=19).cpu().tolist()))
    return counts
