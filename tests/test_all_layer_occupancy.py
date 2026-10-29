"""Check actual per-layer objective coefficients, values and gradients."""
import pytest
import torch
from loss.all_layer_occupancy_loss import AllLayerOccupancyLoss
from loss.occupancy_loss import OccupancyLoss


def parameters():
    return dict(empty_label=17, num_classes=18, balance_cls_weight=True,
        manual_class_weight=[1.] * 17 + [.5], use_sem_geo_scal_loss=False,
        use_lovasz_loss=True, lovasz_ignore=17,
        multi_loss_weights=dict(loss_voxel_ce_weight=10., loss_voxel_lovasz_weight=1.))


def test_all_six_occupancy_terms_equal_independent_original_full_weight_losses():
    torch.manual_seed(928)
    predictions = [torch.randn(1, 18, 49, requires_grad=True) for _ in range(6)]
    target = (torch.arange(49) % 18).view(1, -1)
    mask = (torch.arange(49) % 4 != 0).view(1, -1)
    inputs = dict(pred_occ=predictions, sampled_xyz=torch.randn(1, 49, 3),
                  sampled_label=target, occ_mask=mask)
    objective = AllLayerOccupancyLoss(**parameters())
    actual, metrics = objective(inputs)
    native = OccupancyLoss(**parameters())
    terms = [native(dict(inputs, pred_occ=[p])) for p in predictions]
    torch.testing.assert_close(actual, sum(terms), rtol=0, atol=0)
    assert actual > native(inputs) * 5.9  # legacy loss would average all six
    for i, term in enumerate(terms):
        torch.testing.assert_close(metrics[f'layer_{i+1}'], term)
    gradients = torch.autograd.grad(actual, predictions, retain_graph=True)
    for p, term, gradient in zip(predictions, terms, gradients):
        reference, = torch.autograd.grad(term, p)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_occupancy_rejects_missing_rendered_layers():
    objective = AllLayerOccupancyLoss(**parameters())
    with pytest.raises(ValueError, match='rendered decoder layers'):
        objective(dict(pred_occ=[torch.zeros(1, 18, 2)], sampled_xyz=torch.zeros(1, 2, 3),
                       sampled_label=torch.zeros(1, 2, dtype=torch.long), occ_mask=None))


def test_experiment_config_preserves_each_original_loss_coefficient():
    from mmengine import Config
    baseline = Config.fromfile('config/mini_all_layer_render/small_baseline_1ep.py')
    treatment = Config.fromfile('config/mini_all_layer_render/small_all_layers_1ep.py')
    assert treatment.model.encoder.attribute_stages == list(range(6))
    assert baseline.model.encoder.attribute_stages == [5]
    assert treatment.model.head.apply_loss_type == 'all'
    assert treatment.loss.loss_cfgs[0] == baseline.loss.loss_cfgs[0]
    new = dict(treatment.loss.loss_cfgs[1])
    assert new.pop('stage_weights') == [1.] * 6
    new['type'] = 'OccupancyLoss'
    assert new == baseline.loss.loss_cfgs[1]
