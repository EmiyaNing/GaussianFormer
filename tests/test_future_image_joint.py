"""Regression guards for trainable sampling and live joint geometry."""
import torch
from torch import nn

from model.encoder.sparseworld_strict_encoder import _Layer
from model.future_image_joint import FutureImageJointRefiner
from model.future_image_probe import AdaptiveOPUSImageReader


def source():
    torch.manual_seed(11)
    student = nn.Module()
    student.encoder = nn.Module()
    student.pc_range = (-40., -40., -1., 40., 40., 5.4)
    student.num_refines = 48
    layer = _Layer(256, 5, 6, 4, 4, 4, 17, 32, 48, 8, 512, 0., .5,
                   student.pc_range)
    student.encoder.layers = nn.ModuleList([layer])
    student.semantic = nn.Linear(256, 48 * 17)
    student.reg = nn.Linear(256, 48 * 3)
    student.velocity = nn.Linear(256, 48 * 2)
    student.requires_grad_(False)
    return student


def test_adaptive_sampling_does_not_inherit_source_freezing():
    student = source()
    reader = AdaptiveOPUSImageReader(student.encoder.layers[-1])
    assert all(p.requires_grad for p in reader.parameters())
    assert all(not p.requires_grad for p in student.parameters())


def test_joint_identity_and_trainable_geometry():
    student = source()
    model = FutureImageJointRefiner(student).eval()
    query = torch.randn(1, 5, 256)
    points = torch.rand(1, 5, 48, 3)
    initial_logits = student.semantic(query).view(1, 5 * 48, 17)
    predictions = model(query, points)
    for prediction in predictions:
        torch.testing.assert_close(prediction['points'], model.world(points).flatten(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(prediction['logits'], initial_logits, rtol=0, atol=0)
    loss = predictions[-1]['points'].square().mean() + predictions[-1]['logits'].square().mean()
    loss.backward()
    assert model.reg.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in student.parameters())
    assert all(p.grad is None for p in model.reference_reg.parameters())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    assert not torch.equal(model(query, points)[-1]['points'], predictions[-1]['points'])


def test_joint_rejects_invalid_rounds():
    import pytest
    with pytest.raises(ValueError):
        FutureImageJointRefiner(source(), rounds=0)
