import torch

from misc.checkpoint_util import select_model_state_dict
from misc.model_ema import ModelEMA


def test_model_ema_averages_parameters_and_copies_buffers():
    model = torch.nn.BatchNorm1d(2)
    ema = ModelEMA(model, decay=0.75, update_buffers=True)

    with torch.no_grad():
        model.weight.fill_(5.0)
        model.bias.fill_(2.0)
        model.running_mean.fill_(3.0)
        model.num_batches_tracked.fill_(7)
    ema.update(model)

    assert torch.allclose(ema.module.weight, torch.full((2,), 2.0))
    assert torch.allclose(ema.module.bias, torch.full((2,), 0.5))
    assert torch.equal(ema.module.running_mean, model.running_mean)
    assert torch.equal(ema.module.num_batches_tracked, model.num_batches_tracked)
    assert ema.num_updates == 1
    assert not ema.module.training
    assert all(not parameter.requires_grad for parameter in ema.module.parameters())


def test_model_ema_checkpoint_round_trip():
    model = torch.nn.Linear(2, 1)
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(4.0)
    ema.update(model)

    restored = ModelEMA(torch.nn.Linear(2, 1), decay=0.9)
    restored.load_state_dict(ema.state_dict())
    assert torch.equal(restored.module.weight, ema.module.weight)


def test_checkpoint_selection_prefers_ema_with_legacy_fallback():
    raw = {'weight': torch.tensor([1.0])}
    ema = {'weight': torch.tensor([2.0])}
    selected, key = select_model_state_dict(
        {'state_dict': raw, 'ema_state_dict': ema}, prefer_ema=True)
    assert selected is ema
    assert key == 'ema_state_dict'

    selected, key = select_model_state_dict(
        {'state_dict': raw}, prefer_ema=True)
    assert selected is raw
    assert key == 'state_dict'
