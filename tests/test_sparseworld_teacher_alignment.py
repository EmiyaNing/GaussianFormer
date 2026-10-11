"""Contract tests for the isolated six-horizon alignment implementation."""

import torch

from model.multi_horizon_teacher import MultiHorizonFrozenOPUSV1Teacher


def _teacher_shell():
    teacher = object.__new__(MultiHorizonFrozenOPUSV1Teacher)
    teacher.history_frames = 6
    teacher.num_horizons = 6
    teacher.num_cams = 1
    return teacher


def test_teacher_windows_are_causal_and_have_eight_frames():
    teacher = _teacher_shell()
    assert teacher.teacher_frame_indices(1) == [7, 0, 1, 2, 3, 4, 5, 6]
    assert teacher.teacher_frame_indices(2) == [8, 7, 0, 1, 2, 3, 4, 5]
    assert teacher.teacher_frame_indices(6) == [12, 11, 10, 9, 8, 7, 0, 1]


def test_projection_rebase_uses_target_native_ego_frame():
    teacher = _teacher_shell()
    projection = torch.eye(4).view(1, 1, 4, 4).repeat(1, 13, 1, 1)
    transforms = torch.eye(4).view(1, 1, 4, 4).repeat(1, 6, 1, 1)
    for horizon in range(1, 7):
        transforms[0, horizon - 1, 0, 3] = float(horizon)
    metas = dict(
        projection_mat=projection,
        image_wh=torch.ones(1, 13, 2),
        future_ego_to_current=transforms)
    frames = teacher.teacher_frame_indices(2)
    selected = teacher._select_metas(metas, 2, frames, frames)
    # Target t+2 is translated +2 in current coordinates.
    # Its own camera needs identity; t+1 needs +1; current/history need +2.
    assert torch.allclose(selected['projection_mat'][0, 0], torch.eye(4))
    plus_one = torch.eye(4)
    plus_one[0, 3] = 1.
    assert torch.allclose(selected['projection_mat'][0, 1], plus_one)
    plus_two = torch.eye(4)
    plus_two[0, 3] = 2.
    assert torch.allclose(selected['projection_mat'][0, 2], plus_two)


def test_matching_losses_backpropagate_to_student_only():
    import loss.sparseworld_teacher_alignment_loss  # noqa: F401
    from loss.sparseworld_teacher_alignment_loss import (
        SparseWorldTeacherAlignmentLoss)
    objective = SparseWorldTeacherAlignmentLoss(
        base_loss=dict(
            type='SparseWorldStrictLoss',
            current_loss=dict(
                type='OPUSSetLoss', stage_weights=[1.],
                loss_mode='official_v1', pc_range=[-1., -1., -1., 1., 1., 1.]),
            pc_range=[-1., -1., -1., 1., 1., 1.]),
        matcher=dict(position_weight=1., semantic_weight=0., max_cost=10.))
    student_points = torch.tensor(
        [[[[0., 0., 0.], [0.1, 0., 0.]],
          [[1., 0., 0.], [1.1, 0., 0.]]]], requires_grad=True)
    student_features = torch.randn(1, 2, 4, requires_grad=True)
    student_logits = torch.randn(1, 2, 3, requires_grad=True)
    student = dict(
        query_points_metric=student_points,
        query_features=student_features,
        aligned_features=student_features,
        semantic_logits=student_logits)
    teacher_features = torch.randn(1, 2, 4)
    teacher = dict(
        query_points_metric=student_points.detach() + .2,
        query_features=teacher_features,
        semantic_logits=torch.randn(1, 2, 3))
    geometry, feature, semantic, metrics = objective._matched_losses(
        student, teacher)
    (geometry + feature + semantic).backward()
    assert student_points.grad is not None
    assert student_features.grad is not None
    assert student_logits.grad is not None
    assert metrics['matched_ratio'].item() == 1.
