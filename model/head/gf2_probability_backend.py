"""Load a project-local build of the unchanged official GF2 probability kernel."""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys


@lru_cache(None)
def probability_backend():
    from torch.utils.cpp_extension import load
    source = Path(__file__).resolve().parent / 'localagg_prob'
    if not source.exists():
        raise RuntimeError(f'GF2 probability operator source directory is missing: {source}')
    # Let PyTorch select the visible GPU architectures unless the user sets
    # TORCH_CUDA_ARCH_LIST. Cache compiled binaries outside the source tree.
    os.environ.setdefault('MAX_JOBS', '2')
    extension = load(name='gf2_probability_localagg_C',
        sources=[str(source / p) for p in ['src/aggregator_impl.cu', 'src/forward.cu',
            'src/backward.cu', 'local_aggregate.cu', 'ext.cpp']],
        extra_cuda_cflags=['-Xcompiler', '-fno-gnu-unique'],
        verbose=False)
    name = '_gf2_probability_official_wrapper'
    path = source / 'local_aggregate_prob/__init__.py'
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules[name + '._C'] = extension
    spec.loader.exec_module(module)
    return module
