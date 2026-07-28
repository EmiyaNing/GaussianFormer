"""Build the standalone Phase-B semantic Gaussian CUDA extension in place."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='semantic_gaussian_localagg',
    ext_modules=[CUDAExtension(
        # This setup.py lives inside the import package.  Building ``_C`` in
        # place therefore creates ``semantic_gaussian_localagg/_C*.so`` next
        # to wrapper.py, matching its relative ``from . import _C`` import.
        name='_C',
        sources=['ext.cpp', 'semantic_gaussian_localagg.cu'],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
    )],
    cmdclass={'build_ext': BuildExtension},
)
