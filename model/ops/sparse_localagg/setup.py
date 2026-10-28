"""Build from the repository root: python3 model/ops/sparse_localagg/setup.py build_ext --inplace."""
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent
VENDORED = HERE.parents[1] / 'head' / 'localagg' / 'src'

setup(
    name='sparse-opus-localagg',
    ext_modules=[CUDAExtension(
        'model.ops.sparse_localagg._C',
        sources=[str(HERE / 'compact_aggregate.cu'),
                 *[str(VENDORED / name) for name in
                   ('aggregator_impl.cu', 'forward.cu', 'backward.cu')]],
        include_dirs=[str(VENDORED)],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-Xcompiler', '-fno-gnu-unique']},
    )],
    cmdclass={'build_ext': BuildExtension},
)
