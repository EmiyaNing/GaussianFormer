"""Small CPU checks for the feature-identity experiment's controls."""
import numpy as np
import torch

from tools.experiment_future2current_identity import (
    ResidualAdapter, _miou_from_hist, magnitude_match, schedule_for,
    semantic_kl_3s,
)
from tools.experiment_future2current import SharedReadout


def test_residual_adapter_starts_as_exact_identity():
    adapter = ResidualAdapter()
    features = torch.randn(2, 7, 256)
    torch.testing.assert_close(adapter(features), features, rtol=0, atol=0)


def test_one_epoch_schedule_uses_every_sample_once():
    schedule = schedule_for(count=17, updates=17, seed=42)
    assert sorted(schedule) == list(range(17))


def test_miou_excludes_free_class_and_uses_confusion_matrix():
    hist = np.zeros((18, 18), dtype=np.int64)
    hist[0, 0] = 5
    hist[1, 1] = 3
    hist[17, 17] = 100
    assert _miou_from_hist(hist) == 100.0


def test_semantic_kd_updates_student_without_updating_frozen_readout():
    torch.manual_seed(4)
    readout = SharedReadout().eval().requires_grad_(False)
    student = torch.randn(1, 3, 256, requires_grad=True)
    teacher = student.detach() + .25 * torch.randn_like(student)
    states = {6: dict(student=student, teacher=teacher,
                      valid=torch.tensor([True, True, False]))}
    objective = semantic_kl_3s(readout, states, temperature=2.)
    assert objective.item() > 0
    objective.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in readout.parameters())


def test_shuffled_loss_matches_scalar_strength_without_correct_gradient():
    shuffled = torch.tensor(3., requires_grad=True)
    correct = torch.tensor(.5, requires_grad=True)
    adjusted = magnitude_match(shuffled, correct)
    torch.testing.assert_close(adjusted.detach(), correct.detach())
    adjusted.backward()
    assert shuffled.grad is not None and shuffled.grad > 0
    assert correct.grad is None
