"""Small CPU checks for the isolated H3 predictable-information experiment."""

import torch

from tools.experiment_future_predictable_information import (
    feature_distance, masks_for_query, teacher_feature_loss,
)


def test_query_groups_and_teacher_helpful_use_same_48_points():
    labels = torch.cat((torch.full((48,), 4), torch.full((48,), 11)))
    base = torch.zeros(96, 17)
    teacher = torch.zeros_like(base)
    teacher[:48, 4] = 5.0
    valid = torch.tensor([True, True])
    masks = masks_for_query(labels, 2, valid, base, teacher)
    assert masks['dynamic'].tolist() == [True, False]
    assert masks['static'].tolist() == [False, True]
    assert masks['teacher_helpful'].tolist() == [True, False]


def test_feature_loss_has_student_gradient_only_on_valid_query():
    student = torch.randn(1, 3, 256, requires_grad=True)
    teacher = torch.randn(1, 3, 256)
    states = {h: {'teacher': teacher, 'valid': torch.tensor([True, False, True])}
              for h in (2, 4, 6)}
    features = {h: student for h in (2, 4, 6)}
    value = teacher_feature_loss(states, features)
    value.backward()
    assert torch.isfinite(value)
    assert student.grad[:, 0].abs().sum() > 0
    assert student.grad[:, 1].abs().sum() == 0
    assert student.grad[:, 2].abs().sum() > 0
    assert torch.allclose(feature_distance(student.detach(), student.detach()),
                          torch.zeros(1, 3), atol=1e-6)
