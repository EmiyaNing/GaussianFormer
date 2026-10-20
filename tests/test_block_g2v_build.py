"""CPU-only checks on the block G2V build metadata.

The tile edge is a compile-time constant, so a tile-size sweep can only be
trusted if each build can be asked what it actually baked in.  These assertions
run without a GPU (importing the extension needs libcudart, not a device), which
matters because the sweep itself needs the card and cannot be exercised in CI.

``sweep_tile_size.py`` asserts ``tile_edge() == <requested>`` after every rebuild
and aborts otherwise; these tests cover the binding itself and the invariant that
a built extension reports a sane, supported edge.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.ops.block_g2v import CUDA_BLOCK_G2V_AVAILABLE, tile_edge  # noqa: E402

# Edge 16 would need 16**3 = 4096 threads and is rejected by a static_assert in
# the kernel, so anything outside this set means the build is not one a sweep
# could have produced.
SUPPORTED_EDGES = {2, 4, 8}


@pytest.mark.skipif(not CUDA_BLOCK_G2V_AVAILABLE,
                    reason='block G2V CUDA extension is not built')
def test_tile_edge_is_an_int_in_the_supported_set():
    edge = tile_edge()
    assert isinstance(edge, int)
    assert edge in SUPPORTED_EDGES, (
        f'tile_edge()={edge} is outside {sorted(SUPPORTED_EDGES)}; a tile sweep '
        f'row with this value would be meaningless')


@pytest.mark.skipif(not CUDA_BLOCK_G2V_AVAILABLE,
                    reason='block G2V CUDA extension is not built')
def test_default_source_build_is_tile_four():
    """The checked-in default must be 4: every P3/P4 block result used it.

    A failure here means the tree is left on a swept tile size, in which case
    the in-place ``.so`` no longer matches the results it is documented against.
    """
    assert tile_edge() == 4, (
        'the in-place extension was built with a non-default tile edge; the '
        'tile sweep restores 4 in its finally block, so rebuild it before '
        'quoting any P4-3/P4-4/P4-5 number')


def test_macro_default_matches_the_kernel_header():
    """The kernel default is 4 unless -DBLOCK_G2V_TILE_EDGE overrides it."""
    source = (ROOT / 'model/ops/block_g2v/block_g2v.cu').read_text(encoding='utf-8')
    assert '#define BLOCK_G2V_TILE_EDGE 4' in source
    assert 'static_assert(TILE_VOXELS <= 1024' in source
