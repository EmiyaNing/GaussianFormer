import torch

from model.sparse_latent_dynamics import (
    HungarianSparseMatcher, SharedSparseLatentPredictor, transform_points)


def test_transform_points_uses_column_vector_se3_convention():
    points = torch.tensor([[[[1., 2., 3.], [0., 0., 0.]]]])
    transform = torch.eye(4).unsqueeze(0)
    transform[:, :3, 3] = torch.tensor([[4., -2., 1.]])
    expected = points + torch.tensor([4., -2., 1.])
    assert torch.allclose(transform_points(points, transform), expected)


def test_hungarian_matcher_recovers_query_permutation():
    centers = torch.tensor([[[0., 0., 0.], [5., 0., 0.], [10., 0., 0.]]])
    points = centers[:, :, None]
    features = torch.eye(3).unsqueeze(0)
    semantics = torch.eye(3).mul(10.).unsqueeze(0)
    permutation = torch.tensor([2, 0, 1])
    predicted = dict(points_metric=points, query_feats=features,
                     semantic_logits=semantics)
    target = dict(points_metric=points[:, permutation],
                  query_feats=features[:, permutation],
                  semantic_logits=semantics[:, permutation])
    matches = HungarianSparseMatcher(max_cost=100.)(predicted, target)
    assert matches.prediction[0].tolist() == [0, 1, 2]
    assert matches.target[0].tolist() == [1, 2, 0]


def test_zero_initialized_predictor_starts_from_ego_warped_current_state():
    torch.manual_seed(0)
    predictor = SharedSparseLatentPredictor(
        embed_dims=16, point_dims=4, num_points=2, num_classes=3,
        num_horizons=2, num_layers=1, num_heads=4)
    state = dict(
        points_metric=torch.randn(1, 3, 2, 3),
        query_feats=torch.randn(1, 3, 16),
        point_feats=torch.randn(1, 3, 2, 4),
        semantic_logits=torch.randn(1, 3, 3))
    transform = torch.eye(4).unsqueeze(0)
    transform[:, :3, 3] = torch.tensor([[1., 2., 0.]])
    output = predictor([state], transform, horizon=0)
    assert torch.allclose(
        output['points_metric'], transform_points(
            state['points_metric'], transform))
    assert torch.allclose(output['query_feats'], state['query_feats'])
    assert torch.allclose(output['point_feats'], state['point_feats'])
    assert output['semantic_logits'].shape == (1, 3, 3)
    assert output['existence_logits'].shape == (1, 3)
