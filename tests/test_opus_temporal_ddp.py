"""Shared temporal backbone checkpointing must work with ordinary DDP."""
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from mmengine import Config

from model.segmentor.opus_temporal_segmentor import OPUSTemporalSegmentor


def _make_temporal(checkpoint=True):
    # A real mmseg ResNet/FPN exercises legacy per-block checkpoint flags;
    # intentionally request with_cp=True to verify the segmentor's protection.
    return OPUSTemporalSegmentor(
        image_chunk_size=6, image_checkpoint=checkpoint,
        image_amp_dtype='float32', img_backbone_out_indices=[0, 1],
        img_backbone=dict(
            type='ResNet', depth=18, stem_channels=8, base_channels=8,
            num_stages=2, strides=(1, 2), dilations=(1, 1), out_indices=(0, 1),
            frozen_stages=-1, norm_eval=True, with_cp=True),
        img_neck=dict(type='FPN', in_channels=[8, 16], out_channels=4,
                      num_outs=2))


class _TemporalFeatures(nn.Module):
    def __init__(self, checkpoint=True):
        super().__init__()
        self.temporal = _make_temporal(checkpoint)

    def forward(self, images):
        features = self.temporal.extract_img_feat(images)['ms_img_feats']
        return sum(feature.square().mean() for feature in features)


@pytest.mark.parametrize('checkpoint', [False, True])
def test_temporal_segmentor_disables_legacy_resnet_block_checkpoints(checkpoint):
    model = _make_temporal(checkpoint)
    assert model.image_checkpoint == checkpoint
    flags = [module.with_cp for module in model.img_backbone.modules()
             if hasattr(module, 'with_cp')]
    assert flags and not any(flags)


@pytest.mark.parametrize('size', ['tiny', 'small', 'middle', 'large'])
def test_eight_frame_configs_keep_only_nonreentrant_checkpointing(size):
    cfg = Config.fromfile(
        'config/full_occ3d_oups_guassian/'
        f'opusv1_gaussian_softplus_learned_bias_{size}_8f.py')
    assert cfg.model.img_backbone.with_cp is False
    assert cfg.model.image_checkpoint
    assert cfg.model.encoder.decoder_checkpoint


def _ddp_worker(rank, rendezvous, checkpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        backend='gloo', init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(42)
        network = _TemporalFeatures(checkpoint)
        network.train()
        ddp = nn.parallel.DistributedDataParallel(
            network, broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.01)
        torch.manual_seed(100 + rank)

        # Eight frames x six cameras: the same backbone is reused eight times.
        for step in range(2):
            images = torch.randn(1, 48, 3, 16, 16, requires_grad=True)
            reference = deepcopy(network)
            reference.temporal.image_checkpoint = False
            reference.train()
            reference_images = images.detach().clone().requires_grad_(True)
            reference_loss = reference(reference_images)
            reference_loss.backward()

            optimizer.zero_grad()
            loss = ddp(images)
            torch.testing.assert_close(loss, reference_loss)
            loss.backward()
            assert torch.isfinite(images.grad).all()
            assert torch.all(images.grad.flatten(2).abs().sum(dim=-1) > 0)
            torch.testing.assert_close(images.grad, reference_images.grad)

            # DDP gradients must equal the mean of independently computed
            # eager gradients, rather than merely avoiding the ready-twice error.
            for (name, parameter), (reference_name, reference_parameter) in zip(
                    network.named_parameters(), reference.named_parameters()):
                assert name == reference_name
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                expected = reference_parameter.grad.clone()
                dist.all_reduce(expected)
                expected /= 2
                torch.testing.assert_close(parameter.grad, expected)

            optimizer.step()
            for parameter in network.parameters():
                rank_zero = parameter.detach().clone()
                dist.broadcast(rank_zero, src=0)
                torch.testing.assert_close(parameter, rank_zero)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason='Gloo distributed backend is unavailable')
@pytest.mark.parametrize('checkpoint', [False, True])
def test_eight_frame_backbone_backward_with_two_ddp_ranks(tmp_path, checkpoint):
    rendezvous = (tmp_path / 'gloo_rendezvous').as_uri()
    torch.multiprocessing.spawn(
        _ddp_worker, args=(rendezvous, checkpoint), nprocs=2, join=True)
