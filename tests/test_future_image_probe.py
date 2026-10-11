"""Guard the oracle probe's identity, geometry and gradient contracts."""
import torch
from torch import nn

from model.future_image_probe import (
    FutureImageFusionProbe, decoded_offset, probe_prediction)
from tools.diagnose_future_image_features import select_extra


def fixture():
    torch.manual_seed(13)
    student = nn.Module()
    student.num_refines = 2
    student.semantic = nn.Linear(8, 34)
    student.reg = nn.Linear(8, 6)
    student.velocity = nn.Linear(8, 4)
    student.requires_grad_(False)
    query = torch.randn(1, 7, 8)
    logits = student.semantic(query).view(1, 7, 2, 17)
    state = dict(query=query, points=torch.randn(1, 7, 2, 3),
                 logits=logits.flatten(1, 2),
                 offset=decoded_offset(student, query, logits),
                 future_extra=torch.randn_like(query),
                 current_extra=torch.randn_like(query),
                 future_visible=torch.tensor([[True, False, True, True, False, True, True]]))
    return student, FutureImageFusionProbe(8), state


def test_zero_adapter_preserves_queries_and_predictions():
    student, probe, state = fixture()
    for mode in ('semantic', 'joint'):
        points, logits, delta = probe_prediction(
            student, probe, state, state['future_extra'], mode)
        torch.testing.assert_close(points, state['points'].flatten(1, 2))
        torch.testing.assert_close(logits, state['logits'], rtol=0, atol=0)
        assert torch.count_nonzero(delta) == 0
        assert points.shape[1] == state['query'].shape[1] * student.num_refines


def test_semantic_intervention_cannot_change_geometry():
    student, probe, state = fixture()
    nn.init.constant_(probe.fusion[-1].bias, .3)
    points, logits, _ = probe_prediction(student, probe, state, state['future_extra'], 'semantic')
    torch.testing.assert_close(points, state['points'].flatten(1, 2), rtol=0, atol=0)
    assert not torch.equal(logits, state['logits'])
    joint_points, _, _ = probe_prediction(student, probe, state, state['future_extra'], 'joint')
    assert not torch.equal(joint_points, points)


def test_frozen_decoder_passes_task_gradient_to_adapter_only():
    student, probe, state = fixture()
    points, logits, _ = probe_prediction(student, probe, state, state['future_extra'], 'joint')
    (points.square().mean() + logits.square().mean()).backward()
    assert probe.fusion[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in student.parameters())


def test_shuffle_changes_correspondence_but_preserves_visibility_pattern():
    _, _, state = fixture()
    mask = state['future_visible']
    state['future_extra'] *= mask.unsqueeze(-1)
    shuffled = select_extra(state, 'future_image', 'shuffled')
    assert torch.equal(shuffled[~mask], state['future_extra'][~mask])
    assert not torch.equal(shuffled[mask], state['future_extra'][mask])
    torch.testing.assert_close(shuffled[mask].sum(0), state['future_extra'][mask].sum(0))
    assert torch.count_nonzero(select_extra(state, 'query_only')) == 0
