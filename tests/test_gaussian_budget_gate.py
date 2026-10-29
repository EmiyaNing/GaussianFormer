"""Numerical and rendering-semantic checks for budget selection."""
import torch
from torch.autograd import gradcheck

from model.encoder.gaussian_budget_gate import budget_sigmoid, gather_candidates, gather_gaussian


def test_threshold_implicit_derivative_matches_finite_difference():
    torch.manual_seed(10)
    scores = torch.randn(2, 9, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda value: budget_sigmoid(value, 6, .7), (scores,),
                     eps=1e-5, atol=2e-5, rtol=2e-4)


def test_budget_and_common_shift_invariance():
    scores = torch.tensor([[1., -2., .3, .9, 2., -1.], [.2] * 6], dtype=torch.float64, requires_grad=True)
    p = budget_sigmoid(scores, 4, .3)
    torch.testing.assert_close(p.sum(-1), torch.full((2,), 4., dtype=p.dtype), rtol=0, atol=1e-9)
    torch.testing.assert_close(p, budget_sigmoid(scores + 7, 4, .3), rtol=0, atol=1e-9)
    weights = torch.tensor([[2., -1., .5, -3., 1., 4.]], dtype=p.dtype)
    derivative = torch.autograd.grad((p * weights).sum(), scores)[0]
    torch.testing.assert_close(derivative.sum(-1), torch.zeros(2, dtype=p.dtype), rtol=0, atol=1e-12)
    assert torch.isfinite(derivative).all() and derivative.norm() > 0


def test_unhelpful_candidate_loses_probability_to_helpful_candidate():
    scores = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
    p = budget_sigmoid(scores, 4, .5)
    # This is an exact small additive-renderer task: enhancing candidate zero
    # increases loss, while enhancing candidate one decreases it.
    loss = p[0, 0] - p[0, 1]
    (gradient,) = torch.autograd.grad(loss, scores)
    updated = budget_sigmoid(scores.detach() - .1 * gradient, 4, .5)
    assert updated[0, 0] < p[0, 0] and updated[0, 1] > p[0, 1]


def test_partial_opacity_is_preserved_after_hard_selection():
    alpha = torch.tensor([[[[.3], [.8], [.6]], [[.7], [.4], [.9]]]])
    query = torch.tensor([[.4, .9]], requires_grad=True)
    log_priority = (query.log()[..., None] + alpha.squeeze(-1).log()).flatten(1)
    selected = torch.argsort(log_priority, descending=True, stable=True)[:, :4]
    hard_alpha = gather_candidates(alpha, selected)
    assert hard_alpha.shape == (1, 1, 4, 1)
    torch.testing.assert_close(hard_alpha.flatten(), alpha.flatten()[selected.flatten()])
    assert query.grad is None


def test_invalid_candidates_have_no_probability_or_gradient():
    scores = torch.tensor([[1., 0., -.5, -torch.inf, .2]], dtype=torch.float64, requires_grad=True)
    p = budget_sigmoid(scores, 3, .4)
    assert p[0, 3] == 0
    gradient = torch.autograd.grad((p * torch.arange(5, dtype=p.dtype)).sum(), scores)[0]
    assert gradient[0, 3] == 0 and torch.isfinite(gradient).all()


def test_cold_gate_and_binary_penalty_remain_finite():
    scores = torch.linspace(-8., 8., 120, dtype=torch.float64)[None].requires_grad_()
    p = budget_sigmoid(scores, 80, .05)
    gradient = torch.autograd.grad((p * (1 - p)).mean(), scores)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(p.sum(), torch.tensor(80., dtype=p.dtype), rtol=0, atol=1e-8)


def test_binary_schedule_warms_up_and_remains_differentiable():
    from loss.gaussian_budget_loss import GaussianBudgetLoss
    score = torch.tensor([[-1., .1, .2, .4, .7, 1.]], requires_grad=True)
    p = budget_sigmoid(score, 4, .3)
    allocation = dict(gates=p, query_scores=torch.tensor([[.5, .6]]),
        selected_per_query=torch.tensor([[1, 3]]), keep_budget=4,
        candidate_count=6, temperature=.3)
    objective = GaussianBudgetLoss(binary_weight=.1, total_iters=101)
    warm, details = objective(dict(allocation=allocation, global_iter=0))
    assert warm == 0 and details['binary_weight'] == 0
    late, details = objective(dict(allocation=allocation, global_iter=50))
    assert late > 0 and details['binary_weight'] == .1
    derivative = torch.autograd.grad(late, score)[0]
    assert torch.isfinite(derivative).all() and derivative.norm() > 0


def test_gaussian_selection_preserves_unset_optional_fields():
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    value = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    gaussian = GaussianPrediction(value, value, value, value[..., :1], value)
    selected = gather_gaussian(gaussian, torch.tensor([[4, 0, 2, 5]]))
    assert selected.means.shape == (1, 1, 4, 3)
    assert selected.original_means is None and selected.delta_means is None
    torch.testing.assert_close(selected.means[0, 0, 0], value[0, 1, 1])
