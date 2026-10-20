"""Load a project-local build of the unchanged official GF2 probability kernel."""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import sys


@lru_cache(None)
def probability_backend():
    import torch
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[2]
    source = root / 'work_dirs/gf2_probability_decomposition/vendor/localagg_prob'
    if not source.exists():
        raise RuntimeError('Run tools/gf2_probability_experiment.py initialize before building the backend')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    os.environ.setdefault('MAX_JOBS', '2')
    build = root / 'work_dirs/gf2_probability_decomposition/backend_build'
    build.mkdir(parents=True, exist_ok=True)
    extension = load(name='gf2_probability_localagg_C',
        sources=[str(source / p) for p in ['src/aggregator_impl.cu', 'src/forward.cu',
            'src/backward.cu', 'local_aggregate.cu', 'ext.cpp']],
        extra_cuda_cflags=['-Xcompiler', '-fno-gnu-unique'],
        build_directory=str(build), verbose=False)
    name = '_gf2_probability_official_wrapper'
    path = source / 'local_aggregate_prob/__init__.py'
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules[name + '._C'] = extension
    spec.loader.exec_module(module)
    return module
