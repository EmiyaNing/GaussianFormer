import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).parents[1] / 'eval.py'
SPEC = importlib.util.spec_from_file_location('occ_eval', MODULE)
occ_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(occ_eval)


GRID_SHAPE = (2, 2, 2)


def test_occ3d_eval_mask_accepts_dense_batched_mask():
    dense = torch.zeros(4, *GRID_SHAPE, dtype=torch.bool)
    dense[2, 1, 0, 1] = True
    result = occ_eval.occ3d_mask_to_numpy(
        {'occ_cam_mask': dense}, 'occ_cam_mask', 2, GRID_SHAPE)
    assert result.shape == GRID_SHAPE
    assert result[1, 0, 1]


def test_occ3d_eval_mask_reshapes_flattened_opus_mask():
    flat = torch.zeros(4, 8, dtype=torch.bool)
    flat[3, 5] = True
    result = occ_eval.occ3d_mask_to_numpy(
        {'occ_cam_mask': flat}, 'occ_cam_mask', 3, GRID_SHAPE)
    assert result.shape == GRID_SHAPE
    assert result.reshape(-1)[5]
