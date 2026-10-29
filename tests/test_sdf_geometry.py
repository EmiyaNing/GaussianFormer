import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from tools.sdf_geometry.field import construct, face_seeds, volume, save, load
from loss.gaussian_sdf_geometry_loss import GaussianSDFGeometryLoss, OPUSSetClassificationLoss, sample_geometry


def test_face_distance_matches_exact_squares():
    rng = np.random.default_rng(928)
    labels = np.where(rng.random((5, 6, 4)) < .3, 4, 17)
    camera = rng.random(labels.shape) > .15
    sdf, valid = construct(labels, camera)
    _, known, _, faces = face_seeds(labels, camera)
    points = (np.indices(labels.shape).reshape(3, -1).T+.5)*.4
    distance2 = np.full(len(points), np.inf)
    for axis, active in enumerate(faces):
        for index in np.argwhere(active):
            center = (index+.5)*.4; center[axis] += .2
            delta = np.maximum(np.abs(points-center)-.2, 0)
            delta[:, axis] = points[:, axis]-center[axis]
            distance2 = np.minimum(distance2, (delta*delta).sum(1))
    np.testing.assert_allclose(np.abs(sdf[known]), np.sqrt(distance2).reshape(labels.shape)[known], atol=2e-7)
    assert not valid[~known].any()


def test_unknown_does_not_create_surface():
    labels = np.full((5, 5, 5), 4)
    camera = np.zeros_like(labels, dtype=bool); camera[2, 2, 2] = True
    sdf, valid = construct(labels, camera)
    assert not valid.any()
    assert np.isfinite(sdf).all()


def test_planar_half_voxel_distance_and_cache(tmp_path):
    labels = np.full((8, 8, 8), 17); labels[:4] = 4
    sdf, valid = construct(labels, np.ones_like(labels, dtype=bool))
    assert np.isclose(sdf[3, 4, 4], -.2)
    assert np.isclose(sdf[4, 4, 4], .2)
    assert valid[3, 4, 4] and valid[4, 4, 4]
    path = tmp_path/'field.npz'; save(path, sdf, valid, 'test')
    s, v = load(path)
    np.testing.assert_array_equal(sdf, s); np.testing.assert_array_equal(valid, v)
    result = volume(s, v)
    assert result.shape == (6, 8, 8, 8) and np.isfinite(result).all()


def test_fixed_normal_loss_gradient():
    a = torch.linspace(-1, 1, 9, dtype=torch.float64)
    z, y, x = torch.meshgrid(a, a, a, indexing='ij')
    fields = torch.stack([.8*x+.6*y, torch.full_like(x, .8), torch.full_like(x, .6),
                          torch.zeros_like(x), torch.ones_like(x), torch.ones_like(x)])[None]
    loss = GaussianSDFGeometryLoss(pc_range=(-1.125,)*3+(1.125,)*3)
    means = torch.tensor([[[.31, .27, .13]]], dtype=torch.float64, requires_grad=True)
    scales = torch.tensor([[[.4, .25, .2]]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[[1., .1, .2, .3]]], dtype=torch.float64, requires_grad=True)
    def objective(m, s, r):
        g = SimpleNamespace(means=m, scales=s, rotations=r)
        return loss({'gaussian': g, 'metas': {'sdf_fields': fields}})[0]
    assert torch.autograd.gradcheck(objective, (means, scales, q), atol=1e-5)
    gradients = torch.autograd.grad(objective(means, scales, q), (means, scales, q))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def test_normal_scale_matches_renderer_covariance():
    source = Path(__file__).resolve().parents[1]/'model/utils/utils.py'
    spec = importlib.util.spec_from_file_location('sdf_rotation_reference', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.manual_seed(928)
    normal = torch.nn.functional.normalize(torch.randn(1, 1, 3), dim=-1)
    fields = torch.ones(1, 6, 4, 4, 4)
    fields[:, 1:4] = normal.reshape(1, 3, 1, 1, 1)
    g = SimpleNamespace(means=torch.zeros(1, 7, 3), scales=torch.rand(1, 7, 3)+.1,
                        rotations=torch.randn(1, 7, 4))
    _, sigma, _, _ = sample_geometry(g, fields, pc_range=(-1.,)*3+(1.,)*3)
    r = module.get_rotation_matrix(g.rotations)
    cov = r.transpose(-1, -2) @ torch.diag_embed(g.scales.square()) @ r
    expected = torch.einsum('bni,bnij,bnj->bn', normal.expand_as(g.means), cov,
                            normal.expand_as(g.means)).sqrt()
    torch.testing.assert_close(sigma, expected)


def test_set_classification_identical_and_no_geometry_gradient():
    from loss.opus_set_loss import OPUSSetLoss
    torch.manual_seed(928)
    points = torch.randn(2, 7, 3, requires_grad=True)
    logits = torch.randn(2, 7, 17, requires_grad=True)
    labels = torch.tensor([[4, 4, 17, 2, 17], [17, 2, 4, 2, 4]])
    inputs = dict(opus_pred_points=[points], opus_pred_logits=[logits],
                  metas=dict(occ_label=labels, occ_xyz=torch.randn(2, 5, 3)))
    reference, _ = OPUSSetLoss(lambda_pts=0)(inputs)
    actual, _ = OPUSSetClassificationLoss()(inputs)
    torch.testing.assert_close(reference, actual)
    torch.testing.assert_close(torch.autograd.grad(reference, logits, retain_graph=True)[0],
                               torch.autograd.grad(actual, logits, retain_graph=True)[0])
    assert torch.autograd.grad(actual, points, allow_unused=True)[0] is None


def test_direct_boundary_normals_survive_cache_and_interpolation(tmp_path):
    labels = np.full((8, 8, 8), 17); labels[:4] = 4
    sdf, valid, normals = construct(labels, np.ones_like(labels, dtype=bool), return_normals=True)
    np.testing.assert_allclose(normals[0], 1.)
    np.testing.assert_allclose(normals[1:], 0.)
    path = tmp_path/'normals.npz'; save(path, sdf, valid, 'test', normals=normals)
    s, v, n = load(path, return_normals=True)
    data = volume(s, v, normals=n)
    assert data[5, 3:5, 3:5, 3:5].all()
    np.testing.assert_allclose(data[1], 1.)


def test_unknown_distance_filter_matches_exact_cubes():
    rng = np.random.default_rng(22)
    shape = (5, 6, 4)
    labels = np.where(rng.random(shape) < .3, 4, 17)
    known = rng.random(shape) > .2
    sdf, valid = construct(labels, known)
    p = (np.indices(shape).reshape(3,-1).T+.5)*.4
    unknown_centers = (np.argwhere(~known)+.5)*.4
    d = np.maximum(np.abs(p[:,None]-unknown_centers[None])-.2, 0)
    exact = np.sqrt((d*d).sum(-1)).min(-1)
    outside = np.minimum(p, np.asarray(shape)*.4-p).min(-1)
    exact = np.minimum(exact,outside).reshape(shape)
    np.testing.assert_array_equal(valid, known & (np.abs(sdf)+1e-6 < exact))
