"""Numerical and output-contract regressions for the optimized renderer."""
import types

import pytest
import torch

from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.head.gaussian_opus_head import GaussianOPUSHead
from model.head.sparse_opus_gausisan_world_head import SparseOpusGaussianRenderer
from model.ops.sparse_localagg import CompactLocalAggregator, _C
from model.segmentor.sparse_opus_gausisan_world import SparseOpusGaussianWorld


def test_analytic_empty_matches_finite_support_and_scalar_gradient(monkeypatch):
    head = SparseOpusGaussianRenderer.__new__(SparseOpusGaussianRenderer)
    torch.nn.Module.__init__(head)
    head.analytic_empty = True
    head.aggregator = CompactLocalAggregator(3, 8, 8, 8, [0., 0., 0.], 1.)
    monkeypatch.setattr(GaussianOPUSHead, '_render_batched',
                        lambda self, xyz, *args: torch.zeros(xyz.shape[0], 18, xyz.shape[1]))
    xyz = torch.tensor([[[0.5, .5, .5], [1.5, .5, .5], [2.5, .5, .5]]])
    means = torch.tensor([[[.5, .5, .5], [.5, .5, .5]]])
    scales = torch.full((1, 2, 3), .2)  # radius=1, final sample outside support
    scalar = torch.tensor(2., requires_grad=True)
    semantics = torch.zeros(1, 2, 18)
    semantics[:, -1, 17] = scalar
    opacity = torch.ones(1, 2, 1)
    covariance = torch.eye(3).expand(1, 2, 3, 3)
    result = head._render_batched(xyz, means, opacity, semantics, scales, covariance)
    expected_weight = torch.tensor([[1., torch.exp(torch.tensor(-.5)), 0.]])
    torch.testing.assert_close(result[:, 17], scalar * expected_weight)
    assert result[:, :17].count_nonzero() == 0
    torch.testing.assert_close(torch.autograd.grad(result.sum(), scalar)[0], expected_weight.sum())


@pytest.mark.parametrize('pretrain,steps,expected', [(True, 1, [0]), (False, 5, [0, 2, 4]), (False, 6, [0, 2, 4, 6])])
def test_training_renders_only_loss_horizons_and_eval_keeps_all(pretrain, steps, expected):
    torch.set_num_threads(4)
    model = SparseOpusGaussianWorld.__new__(SparseOpusGaussianWorld)
    torch.nn.Module.__init__(model)
    model.pretrain = pretrain
    model.gaussian_loss_horizons = (0, 2, 4, 6)
    calls = []

    class Renderer(torch.nn.Module):
        def render_state(self, state, xyz):
            calls.append(torch.is_grad_enabled())
            return torch.zeros(1, 18, 1).expand(1, 18, 640000)

    model.gaussian_head = Renderer()
    encoder = types.SimpleNamespace(layers=[None], _make_gaussian=lambda *args: None)
    model.pts_bbox_head = types.SimpleNamespace(transformer=types.SimpleNamespace(encoder=encoder))
    gaussian = GaussianPrediction(*[torch.zeros(1, 2, 1, size) for size in (3, 3, 4, 1, 17)])
    outputs = {'representation': [{'gaussian': gaussian}]}
    points, classes, features = [torch.zeros(1, 2, 1, 3)] * steps, [None] * steps, [torch.zeros(1, 2, 4)] * steps
    args = (outputs, points, classes, features, torch.tensor([True, False]), torch.zeros(1, 640000, 3), 1)
    model.train()
    logits, grids = model._render_occupancy(*args, render_all=False)
    assert list(logits) == expected
    assert len(calls) == len(expected) and all(calls)
    assert len(grids) == 1
    calls.clear()
    model.eval()
    with torch.no_grad():
        _, grids = model._render_occupancy(*args, render_all=True)
    assert len(grids) == steps + 1
    assert len(calls) == steps + 1 and not any(calls)


CUDA = pytest.mark.skipif(not torch.cuda.is_available() or _C is None,
                          reason='requires CUDA and the repository-local sparse localagg extension')


def cuda_inputs(subset=False):
    torch.manual_seed(17)
    coordinates = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16), torch.arange(8), indexing='ij'), -1)
    pts = (coordinates.cuda().float().reshape(1, -1, 3) + .5) * .4
    if subset:
        pts = pts[:, torch.randperm(pts.shape[1], device='cuda')[:600]]
    means = torch.rand(1, 64, 3, device='cuda') * torch.tensor([6.3, 6.3, 3.1], device='cuda')
    opacity = torch.rand(1, 64, 1, device='cuda')
    semantics = torch.rand(1, 64, 18, device='cuda')
    scales = torch.full((1, 64, 3), .4, device='cuda')
    matrix = torch.randn(1, 64, 3, 3, device='cuda')
    covariance = matrix.transpose(-1, -2) @ matrix + torch.eye(3, device='cuda')
    return pts, means, opacity, semantics, scales, covariance


def execute(operator, inputs):
    pts, means, opacity, semantics, scales, covariance = inputs
    leaves = [x.detach().clone().requires_grad_() for x in (means, opacity, semantics, covariance)]
    result = operator(pts, leaves[0], leaves[1].reshape(1, -1), leaves[2], scales, leaves[3])
    gradient = torch.sin(torch.arange(result.numel(), device='cuda').reshape_as(result).float())
    gradients = torch.autograd.grad(result, leaves, gradient)
    return result, gradients


@CUDA
@pytest.mark.parametrize('subset', [False, True])
def test_compact_matches_legacy_outputs_gradients_and_nondefault_stream(subset):
    from local_aggregate import LocalAggregator
    arguments = dict(scale_multiplier=3, H=16, W=16, D=8, pc_min=[0., 0., 0.], grid_size=.4)
    legacy = LocalAggregator(**arguments).cuda()
    compact = CompactLocalAggregator(**arguments).cuda()
    inputs = cuda_inputs(subset)
    expected, expected_gradients = execute(legacy, inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual, actual_gradients = execute(compact, inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_grad, expected_grad in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-5)


@CUDA
def test_compact_saved_storage_is_less_than_third_of_legacy():
    from local_aggregate import LocalAggregator
    args = dict(scale_multiplier=3, H=16, W=16, D=8, pc_min=[0., 0., 0.], grid_size=.4)
    inputs = cuda_inputs()
    saved_bytes = []
    for cls in (LocalAggregator, CompactLocalAggregator):
        sizes = []
        def pack(tensor):
            sizes.append(tensor.untyped_storage().nbytes())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            execute(cls(**args).cuda(), inputs)
        saved_bytes.append(sum(sizes))
    assert saved_bytes[1] < saved_bytes[0] / 3, saved_bytes


@CUDA
def test_compact_empty_gaussian_set_has_zero_outputs_and_gradients():
    operator = CompactLocalAggregator(3, 16, 16, 8, [0., 0., 0.], .4).cuda()
    inputs = cuda_inputs()
    inputs = (inputs[0], *[x[:, :0] for x in inputs[1:]])
    result, gradients = execute(operator, inputs)
    assert result.count_nonzero() == 0
    assert all(gradient.numel() == 0 for gradient in gradients)
