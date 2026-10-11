import multiprocessing as mp

import torch
from torch import nn

from dataset import _loader_runtime_kwargs
from dataset.sparseworld_trajectory import NuScenesSparseWorldTrajectoryDataset
from dataset.sampler import CustomDistributedSampler
from model.segmentor.forecasting_sparseworld_segmentor import (
    ForecastingSparseWorld,
)


class _Backbone(nn.Module):
    def forward(self, images):
        return (images * 2., images[:, :, ::2, ::2] * 3.)


class _Neck(nn.Module):
    def forward(self, features):
        return tuple(feature + 1. for feature in features)


def test_loader_runtime_options_are_forwarded():
    options = _loader_runtime_kwargs(dict(
        batch_size=1, num_workers=6, pin_memory=True,
        persistent_workers=True, prefetch_factor=2))
    assert options == dict(
        batch_size=1, num_workers=6, pin_memory=True,
        persistent_workers=True, prefetch_factor=2)
    assert _loader_runtime_kwargs(dict(batch_size=1, num_workers=0)) == dict(
        batch_size=1, num_workers=0, pin_memory=True)


def test_single_gpu_iter_resume_sampler_skips_consumed_items():
    sampler = CustomDistributedSampler(
        list(range(10)), num_replicas=1, rank=0,
        shuffle=False, drop_last=False)
    sampler.set_last_iter(4)
    assert list(sampler) == list(range(4, 10))
    # Only the first iterator is shortened; later epochs are complete.
    assert list(sampler) == list(range(10))


def test_single_future_curriculum_is_epoch_visible():
    dataset = object.__new__(NuScenesSparseWorldTrajectoryDataset)
    dataset.future_steps = 6
    dataset.forecast_curriculum_start_epoch = 0
    dataset.forecast_curriculum_epochs_per_horizon = 2
    dataset._shared_epoch = mp.Value('i', 0, lock=False)
    assert {dataset._training_horizon(i) for i in range(8)} == {0}
    dataset.set_epoch(4)
    assert {dataset._training_horizon(i) for i in range(12)} == {0, 1, 2}
    dataset.set_epoch(10)
    assert {dataset._training_horizon(i) for i in range(12)} == set(range(6))


def test_chunked_observed_encoding_preserves_frame_order():
    network = object.__new__(ForecastingSparseWorld)
    nn.Module.__init__(network)
    network.img_backbone = _Backbone()
    network.img_neck = _Neck()
    network.freeze_img_backbone = True
    network.num_cams = 2
    network.observed_backbone_chunk_frames = 2
    images = torch.randn(2, 3, 2, 3, 8, 8)
    indices = [2, 1, 0]

    chunked = network._observed_features(images, indices)
    sequential = []
    for index in indices:
        pyramid = network._frame_features(images[:, index])
        sequential.append(pyramid)
    for level, values in enumerate(chunked):
        expected = torch.stack(
            [frame[level] for frame in sequential], dim=1)
        torch.testing.assert_close(values, expected)
