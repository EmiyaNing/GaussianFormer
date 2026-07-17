from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='opus_msmv_sampling',
    ext_modules=[CUDAExtension(
        # Build in-place beside ``wrapper.py``; relative import keeps the
        # extension independent of the caller's PYTHONPATH package name.
        name='_msmv_sampling_cuda',
        sources=['csrc/msmv_sampling.cpp', 'csrc/msmv_sampling_forward.cu',
                 'csrc/msmv_sampling_backward.cu'],
        include_dirs=['csrc'],
        extra_compile_args={'nvcc': ['-O3']},
    )],
    cmdclass={'build_ext': BuildExtension},
)
