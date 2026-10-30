from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='react_render_ops', ext_modules=[CUDAExtension(
    name='_C', sources=['bindings.cpp', 'render.cu'],
    extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']})],
    cmdclass={'build_ext': BuildExtension})
