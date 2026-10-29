"""Regression checks for model coordinates, label filtering and ellipsoid axes."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmengine import Config

from visualize import (
    COLORS, SceneRenderer, gaussian_data, gaussian_transforms,
    occupancy_data, parse_args, voxel_size_from_config,
)


def make_gaussian(count, grouped=False, classes=17):
    shape = (1, 1, count) if grouped else (1, count)
    semantics = np.zeros((*shape, classes))
    semantics[..., 0] = 1
    return SimpleNamespace(
        means=np.zeros((*shape, 3)), scales=np.ones((*shape, 3)) * .2,
        rotations=np.broadcast_to([1., 0, 0, 0], (*shape, 4)).copy(),
        opacities=np.ones((*shape, 1)) * .01, semantics=semantics,
    )


def test_occupancy_uses_model_coordinates_and_does_not_edit_labels():
    xyz = torch.tensor([[[[-39.8, -39.8, -.8], [-39.8, -39.8, -.4], [-39.8, -39.8, 0.]]]])
    pred = torch.tensor([[0, 17, 4]])
    truth = torch.tensor([[[0, 17, 4]]])
    metas = dict(occ_xyz=xyz, occ_label=truth, occ_cam_mask=torch.tensor([[[True, True, False]]]))
    output = dict(final_occ=pred, sampled_xyz=xyz.reshape(1, -1, 3), sampled_label=truth.flatten(1))
    data = occupancy_data(output, metas, 'camera')
    np.testing.assert_allclose(data['xyz'], xyz.numpy().reshape(-1, 3))
    np.testing.assert_array_equal(data['mask'], [True, True, False])
    np.testing.assert_array_equal(data['prediction'], [0, 17, 4])
    data['prediction'][0] = 9
    assert pred.tolist() == [[0, 17, 4]]
    assert truth.flatten().tolist() == [0, 17, 4]


def test_requested_mask_is_required():
    metas = dict(occ_xyz=torch.zeros(1, 1, 3), occ_label=torch.zeros(1, 1))
    with pytest.raises(ValueError, match='occ_cam_mask'):
        occupancy_data(dict(final_occ=torch.zeros(1, 1)), metas, 'camera')


def test_grouped_gaussians_keep_all_foreground_and_low_opacity():
    gaussian = make_gaussian(76800, grouped=True)
    gaussian.semantics[..., -1, :] = 0
    gaussian.semantics[..., -1, 16] = 1
    data = gaussian_data(gaussian)
    assert len(data['means']) == 76800
    assert data['labels'][0] == 0
    assert data['labels'][-1] == 16
    np.testing.assert_allclose(data['opacities'], .01)
    np.testing.assert_array_equal(data['source_indices'], np.arange(76800))


def test_gaussian_filter_removes_empty_and_padding_without_changing_inputs():
    gaussian = make_gaussian(4, classes=18)
    gaussian.semantics[0, 1, :] = 0
    gaussian.semantics[0, 1, 17] = 1
    gaussian.scales[0, 2, 0] = np.nan
    data = gaussian_data(gaussian, torch.tensor([[True, True, True, False]]))
    np.testing.assert_array_equal(data['source_indices'], [0])
    assert gaussian.means.shape == (1, 4, 3)
    assert np.isnan(gaussian.scales[0, 2, 0])


def test_ellipsoid_transform_matches_head_covariance_and_mahalanobis_surface():
    from model.utils.utils import get_rotation_matrix
    rng = np.random.default_rng(928)
    rotations = rng.normal(size=(5, 4))
    scales = rng.uniform(.08, .4, size=(5, 3))
    transform = gaussian_transforms(scales, rotations)
    expected = (torch.diag_embed(torch.tensor(scales)) @
                get_rotation_matrix(torch.tensor(rotations))).numpy()
    np.testing.assert_allclose(transform, expected, atol=1e-12)
    covariance = np.swapaxes(expected, -1, -2) @ expected
    unit = rng.normal(size=(20, 3))
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    offsets = 2 * np.einsum('vj,njk->nvk', unit, transform)
    distance = np.einsum('nvi,nij,nvj->nv', offsets, np.linalg.inv(covariance), offsets)
    np.testing.assert_allclose(distance, 4., atol=1e-10)


def test_occ3d_voxel_size_comes_from_configuration():
    cfg = Config(dict(model=dict(head=dict(cuda_kwargs=dict(grid_size=.4)))))
    np.testing.assert_allclose(voxel_size_from_config(cfg), [.4, .4, .4])


def test_voxel_geometry_keeps_class_zero_and_exact_centers():
    renderer = SceneRenderer.__new__(SceneRenderer)
    renderer.o3d = pytest.importorskip('open3d')
    data = dict(xyz=np.array([[-39.8, -39.8, -.8], [0, 0, 0], [10, 10, 1.]]),
                prediction=np.array([0, 17, 4]), mask=np.array([True, True, False]))
    mesh = renderer.occupancy(data, 'prediction', np.array([.4, .4, .4]))
    vertices = np.asarray(mesh.vertices)
    assert len(vertices) == 8
    np.testing.assert_allclose(vertices.mean(0), [-39.8, -39.8, -.8])
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [.4, .4, .4])
    np.testing.assert_allclose(np.asarray(mesh.vertex_colors), np.tile(COLORS[0], (8, 1)))


def test_empty_outputs_are_supported():
    data = gaussian_data(make_gaussian(0))
    assert data['means'].shape == (0, 3)
    assert gaussian_transforms(data['scales'], data['rotations']).shape == (0, 3, 3)


def test_vis_occ_opens_interactive_windows_by_default():
    assert parse_args(['--py-config', 'config.py', '--vis-occ']).show is True
    assert parse_args(['--py-config', 'config.py', '--vis-gaussian']).show is True


def test_batch_export_can_explicitly_disable_windows():
    assert parse_args(['--py-config', 'config.py', '--vis-occ', '--no-show']).show is False
    assert parse_args(['--py-config', 'config.py', '--vis-occ', '--data-only']).show is False
    assert parse_args(['--py-config', 'config.py']).show is False
