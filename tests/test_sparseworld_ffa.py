import torch
from types import SimpleNamespace

from eval_ffa import _seconds_to_horizons
from model.ffa_teacher import FrozenOPUSV1Teacher
from model.future_feature_alignment import FutureFeatureAlignment
from model.segmentor.sparseworld_ffa_segmentor import SparseWorldFFASegmentor
from model.sparse_latent_dynamics import HungarianSparseMatcher
from train_ffa import _find_nonfinite_gradients


def test_teacher_window_ends_at_h6_and_uses_opus_history_order():
    teacher = object.__new__(FrozenOPUSV1Teacher)
    teacher.target_horizon_index = 6
    teacher.num_student_frames = 5
    assert teacher.teacher_frame_indices == [10, 9, 8, 7, 6, 5, 0, 1]


def test_teacher_projections_are_rebased_to_target_future_ego():
    teacher = object.__new__(FrozenOPUSV1Teacher)
    teacher.target_horizon_index = 6
    teacher.num_student_frames = 5
    teacher.num_cams = 1
    transforms = torch.eye(4).repeat(1, 6, 1, 1)
    transforms[0, :, 0, 3] = torch.arange(1., 7.)
    metas = dict(
        projection_mat=torch.eye(4).repeat(1, 11, 1, 1),
        image_wh=torch.ones(1, 11, 2),
        future_ego_to_current=transforms)
    frames = [10, 9, 5, 0, 1]
    selected = teacher._select_metas(metas, frames, frames)
    torch.testing.assert_close(
        selected['projection_mat'][0, :, 0, 3],
        torch.tensor([0., 1., 5., 6., 6.]))


def test_zero_initialized_ffa_is_ego_warped_current_query():
    torch.manual_seed(0)
    module = FutureFeatureAlignment(
        embed_dims=16, num_points=4, num_layers=1, num_heads=4,
        ffn_dims=32, alignment_dims=8, dropout=0.,
        checkpoint_layers=False, shape_start_epoch=8)
    features = torch.randn(2, 5, 16)
    points = torch.randn(2, 5, 4, 3)
    transform = torch.eye(4).repeat(2, 1, 1)
    transform[:, :3, 3] = torch.tensor([[1., 2., 0.], [-2., 0., 1.]])
    output = module(features, points, transform)
    expected = points + transform[:, None, None, :3, 3]
    torch.testing.assert_close(output['points_metric'], expected)
    torch.testing.assert_close(output['query_features'], features)
    assert output['shape_scale'].item() == 0.


def test_geometry_semantic_matcher_does_not_require_feature_basis():
    matcher = HungarianSparseMatcher(
        position_weight=1., semantic_weight=1., feature_weight=0.,
        max_cost=100.)
    points = torch.tensor([[[[0., 0., 0.]], [[5., 0., 0.]]]])
    semantic = torch.tensor([[[10., 0.], [0., 10.]]])
    permutation = torch.tensor([1, 0])
    matches = matcher(
        dict(points_metric=points, semantic_logits=semantic),
        dict(points_metric=points[:, permutation],
             semantic_logits=semantic[:, permutation]))
    assert matches.prediction[0].tolist() == [0, 1]
    assert matches.target[0].tolist() == [1, 0]


def test_nonfinite_gradient_detection_reports_parameter_names():
    module = torch.nn.Linear(2, 1)
    module(torch.ones(1, 2)).sum().backward()
    assert _find_nonfinite_gradients(module) == []

    module.weight.grad[0, 0] = float('inf')
    assert _find_nonfinite_gradients(module) == ['weight']


def test_eval_seconds_map_to_keyframe_horizons():
    assert _seconds_to_horizons((1., 2., 3.), .5, 6) == [2, 4, 6]


def test_target_rollout_retains_only_future_born_suffix():
    points = [torch.zeros(1, 1, 3) for _ in range(6)]
    logits = [torch.zeros(1, 1, 17) for _ in range(6)]
    points[-1] = torch.arange(10 * 3.).view(1, 10, 3)
    logits[-1] = torch.arange(10 * 17.).view(1, 10, 17)
    owner = SimpleNamespace(target_horizon_index=6, num_refines=2)
    born_points, born_logits = \
        SparseWorldFFASegmentor._future_born_at_target(
            owner, dict(points=points, logits=logits),
            persistent_queries=3)
    torch.testing.assert_close(born_points, points[-1][:, 6:])
    torch.testing.assert_close(born_logits, logits[-1][:, 6:])
