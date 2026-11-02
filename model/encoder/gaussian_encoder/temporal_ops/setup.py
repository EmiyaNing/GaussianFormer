"""Build separately from the existing DAF extension and running processes."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='temporal_sampling_ext',
    ext_modules=[CUDAExtension(
        'temporal_sampling_ext',
        sources=['src/bindings.cpp', 'src/pyramid_pack.cu',
                 'src/packed_dropout.cu', 'src/temporal_sampling.cu'],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-lineinfo']})],
    cmdclass={'build_ext': BuildExtension})
