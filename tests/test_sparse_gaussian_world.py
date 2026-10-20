from pathlib import Path

import pytest
import torch

from loss.sparseworld_gaussian_point_occupancy_loss import (
    SparseWorldGaussianPointOccupancyLoss)
from model.head.point_anchored_semantic_gaussian import (
    PointAnchoredSemanticGaussian, PointAnchoredSemanticGaussianCUDA)


def test_point_anchored_gaussian_applies_child_valid_mask_and_zero_init():
    module = PointAnchoredSemanticGaussian(
        embed_dims=8, num_children=2, num_classes=3,
        pc_range=(0., 0., 0., 2., 2., 2.), num_neighbors=2)
    token = torch.randn(1, 2, 8)
    points = torch.tensor([[[[.1, .1, .1], [.2, .2, .2]],
                            [[.3, .3, .3], [.4, .4, .4]]]])
    output = module(
        token, points, torch.tensor([[True, True]]),
        torch.tensor([[[True, False], [True, True]]]))
    assert output['valid_mask'].tolist() == [[True, False, True, True]]
    assert output['residual'].shape == (1, 4, 3)
    assert torch.equal(output['residual'], torch.zeros_like(output['residual']))
    assert torch.all(output['scales'] >= .1) and torch.all(output['scales'] <= .6)
    assert torch.all(output['opacities'] >= 0.) and torch.all(output['opacities'] <= 1.)


def test_sparseworld_gaussian_loss_uses_future_horizon_and_detaches_base():
    criterion = SparseWorldGaussianPointOccupancyLoss(
        empty_label=2, class_weights=(1., 1.), horizon_weights=(1.,) * 7)
    current_base = torch.zeros(1, 1, 2, requires_grad=True)
    current_residual = torch.zeros(1, 1, 2, requires_grad=True)
    future_base = torch.zeros(1, 1, 2, requires_grad=True)
    future_residual = torch.zeros(1, 1, 2, requires_grad=True)
    xyz = torch.tensor([[[[[.5, .5, .5]]]]])
    future_labels = torch.full((1, 6, 1, 1, 1), 2, dtype=torch.long)
    future_labels[:, 1] = 1  # horizon=2 must select index 1.
    loss, metrics = criterion({
        'current_gaussian_point_logits': current_base.detach() + current_residual,
        'current_gaussian_point_points': torch.tensor([[[.5, .5, .5]]]),
        'current_gaussian_point_valid_mask': torch.tensor([[True]]),
        'future_gaussian_point_logits': [future_base.detach() + future_residual],
        'future_gaussian_point_points': [torch.tensor([[[.5, .5, .5]]])],
        'future_gaussian_point_valid_masks': [torch.tensor([[True]])],
        'future_gaussian_horizons': [2],
        'sparseworld_pretrain': False,
        'global_iter': 1,
        'metas': {
            'occ_xyz': xyz, 'occ_label': torch.zeros(1, 1, 1, 1, dtype=torch.long),
            'future_occ_labels': future_labels,
        }})
    loss.backward()
    assert current_base.grad is None and future_base.grad is None
    assert current_residual.grad is not None and current_residual.grad.abs().sum() > 0
    assert future_residual.grad is not None and future_residual.grad.abs().sum() > 0
    assert metrics['gaussian_active_horizons'].item() == 2


def test_sparse_gaussian_world_warmstart_allows_only_adapter_keys():
    checkpoint = Path('../SparseWorld/ckpts/epoch_56.pth')
    if not checkpoint.is_file():
        pytest.skip('official SparseWorld checkpoint is not present')
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model  # noqa: F401
    from misc.sparseworld_compat_checkpoint import (
        load_sparse_gaussian_world_warmstart)
    config = Config.fromfile('config/opus/sparse_gaussian_world.py')
    network = build_segmentor(config.model)
    report = load_sparse_gaussian_world_warmstart(network, checkpoint)
    assert report['loaded'] == 679
    assert report['missing']
    assert all(key.startswith('semantic_gaussian.') for key in report['missing'])
    assert network.pretrain is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_semantic_gaussian_cuda_forward_backward():
    module = PointAnchoredSemanticGaussianCUDA(
        embed_dims=8, num_children=2, num_classes=3,
        pc_range=(0., 0., 0., 2., 2., 2.), cell_size=1.).cuda()
    with torch.no_grad():
        module.residual_head.bias.fill_(.1)
    token = torch.randn(1, 2, 8, device='cuda', requires_grad=True)
    points = torch.tensor(
        [[[[.1, .1, .1], [.2, .2, .2]],
          [[.3, .3, .3], [2.1, .4, .4]]]], device='cuda')
    point_mask = ((points >= 0.) & (points < 2.)).all(-1)
    output = module(
        token, points, torch.ones(1, 2, dtype=torch.bool, device='cuda'),
        point_mask)
    loss = output['residual'].square().sum()
    loss.backward()
    assert output['valid_mask'].tolist() == [[True, True, True, False]]
    assert torch.isfinite(loss)
    assert module.residual_head.weight.grad is not None
    assert torch.isfinite(module.residual_head.weight.grad).all()
