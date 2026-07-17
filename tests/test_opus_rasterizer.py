import importlib.util
from pathlib import Path

import torch


MODULE = Path(__file__).parents[1] / 'model/head/opus_rasterizer.py'
SPEC = importlib.util.spec_from_file_location('opus_rasterizer', MODULE)
opus_rasterizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(opus_rasterizer)


def test_rasterizer_keeps_highest_confidence_point():
    rasterizer = opus_rasterizer.OPUSRasterizer(
        pc_range=[0, 0, 0, 2, 2, 2], grid_size=1, grid_shape=[2, 2, 2])
    points = torch.tensor([[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]]])
    logits = torch.tensor([[[0.0, 1.0], [0.0, 4.0], [3.0, 0.0]]])
    dense = rasterizer(points, logits)
    assert dense.shape == (1, 8)
    assert dense[0, 0].item() == 1
    assert dense[0, 7].item() == 0


def test_rasterizer_discards_out_of_bounds_points():
    rasterizer = opus_rasterizer.OPUSRasterizer(
        pc_range=[0, 0, 0, 1, 1, 1], grid_size=1, grid_shape=[1, 1, 1])
    dense = rasterizer(torch.tensor([[[2.0, 0.0, 0.0]]]), torch.tensor([[[1.0, 0.0]]]))
    assert dense[0, 0].item() == 17


def test_rasterizer_discards_padded_points():
    rasterizer = opus_rasterizer.OPUSRasterizer(
        pc_range=[0, 0, 0, 1, 1, 1], grid_size=1, grid_shape=[1, 1, 1])
    dense = rasterizer(
        torch.tensor([[[0.1, 0.1, 0.1]]]), torch.tensor([[[0.0, 3.0]]]),
        point_valid_mask=torch.tensor([[False]]))
    assert dense[0, 0].item() == 17
