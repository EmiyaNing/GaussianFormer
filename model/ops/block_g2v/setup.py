"""Build the block G2V extension with a chosen tile edge.

The tile edge is a compile-time constant of the kernel (it fixes both the tile
geometry and the CTA size), so measuring a different tile size means rebuilding.
``BLOCK_G2V_TILE_EDGE`` is read from the environment here and forwarded to nvcc
as ``-DBLOCK_G2V_TILE_EDGE=<n>``; the kernel defaults to 4 when it is unset, so
an ordinary in-place build is unaffected.

    BLOCK_G2V_TILE_EDGE=8 python3.10 model/ops/block_g2v/setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TILE_EDGE = os.environ.get('BLOCK_G2V_TILE_EDGE', '4').strip()

NVCC_ARGS = ['-O3', '--use_fast_math']
if TILE_EDGE and TILE_EDGE != '4':
    NVCC_ARGS.append(f'-DBLOCK_G2V_TILE_EDGE={TILE_EDGE}')

setup(
    name='block_g2v',
    ext_modules=[
        CUDAExtension('block_g2v', ['block_g2v.cu', 'ext.cpp'],
                      extra_compile_args={'nvcc': NVCC_ARGS}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
