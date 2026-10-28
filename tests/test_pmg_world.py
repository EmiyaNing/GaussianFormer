"""Geometry and experiment controls that distinguish the proposed mechanisms."""

from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from model.encoder.pmg_shape_projection import (apply_rotation, centered, fit_yaw,
                                              rigid_tangent, shape_prox,
                                              structured_update, yaw_matrix)
from tools.pmg_world.regions import box_regions, histogram, region_result
from tools.sparse_opus_gausisan_world.runtime import model_fingerprint


def cloud():
    return centered(torch.randn(2, 3, 32, 3, generator=torch.Generator().manual_seed(17), dtype=torch.float64))


def test_rigid_transport_preserves_all_pair_distances_and_recovers_yaw():
    template = cloud()
    rotation = yaw_matrix(torch.full((2, 3), .7, dtype=torch.float64))
    candidate = apply_rotation(template, rotation) + torch.tensor([10., 2., 0.])
    torch.testing.assert_close(fit_yaw(template, centered(candidate)), torch.full((2, 3), .7, dtype=torch.float64))
    torch.testing.assert_close(torch.cdist(template, template), torch.cdist(candidate, candidate))


def test_proximal_residual_satisfies_tangent_and_total_budget_across_six_steps():
    reference = cloud()
    previous = reference
    for _ in range(6):
        offsets = torch.randn_like(previous) * .5
        updated, angle = structured_update(previous, reference, offsets, "prox", radius_rms=.1)
        canonical = apply_rotation(centered(updated), yaw_matrix(angle).transpose(-1, -2))
        residual = canonical - reference
        assert residual.square().sum(-1).mean(-1).sqrt().max() <= .1 + 1e-8
        projection = rigid_tangent(reference).transpose(-1, -2) @ residual.flatten(-2)[..., None]
        assert projection.abs().max() < 1e-8
        previous = updated


@pytest.mark.parametrize("mode", ["translation", "rigid", "free", "prox"])
def test_geometry_modes_have_finite_nonzero_motion_gradients(mode):
    reference = cloud()
    offsets = (torch.randn_like(reference) * .01).requires_grad_()
    updated, _ = structured_update(reference, reference, offsets, mode)
    gradient = torch.autograd.grad((updated * torch.tensor([1., 2., 3.])).sum(), offsets)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    torch.testing.assert_close(updated.mean(-2), offsets.mean(-2), atol=1e-10, rtol=1e-10)


def test_degenerate_shape_and_zero_radius_have_finite_backward():
    reference = torch.zeros(1, 2, 4, 3, dtype=torch.float64)
    candidate = torch.randn_like(reference).requires_grad_()
    result, _ = structured_update(reference, reference, candidate, "prox", radius_rms=0)
    gradient = torch.autograd.grad(result.square().sum(), candidate)[0]
    assert torch.isfinite(result).all() and torch.isfinite(gradient).all()


def test_fixed_proximal_map_does_not_amplify_candidate_perturbations():
    reference = cloud()
    a, b = torch.randn_like(reference), torch.randn_like(reference)
    change = shape_prox(a, reference) - shape_prox(b, reference)
    assert change.norm() <= (a-b).norm() / 3 + 1e-10


def test_free_residual_is_an_additive_coordinate_update():
    reference = cloud()
    previous, offsets = reference * 1.2, torch.randn_like(reference)
    actual, _ = structured_update(previous, reference, offsets, "free")
    torch.testing.assert_close(actual, previous + offsets)


def test_region_metrics_separate_occupied_semantics_and_false_positives():
    hist = histogram(np.array([4, 17, 4, 5]), np.array([4, 4, 17, 5]))
    result = region_result(hist)
    assert result["false_positive"] == 1 and result["false_negative"] == 1
    assert result["occupied_recall"] == pytest.approx(200/3)
    assert result["conditional_semantic_accuracy"] == 100.


def test_region_assignment_uses_rotated_box_and_excludes_overlaps():
    xyz = np.zeros((640000, 3))
    xyz[:4] = [[0., 0., 0.], [5., 0., 0.], [.1, 0., 0.], [0., 1.5, 0.]]
    target = np.full(640000, 17)
    target[:4] = 4
    instance = dict(label=4, translation=[0., 0., 0.], rotation=[2**-.5, 0., 0., 2**-.5],
                    size=[2., 4., 2.], speed=4., voxels=2)
    masks, _ = box_regions(target, xyz, [instance], np.eye(4))
    assert masks["fast"][:4].tolist() == [True, False, True, True]
    masks, denominator = box_regions(target, xyz, [instance, instance], np.eye(4))
    assert masks["fast"].sum() == 0 and denominator["ambiguous_voxels"] == 3


def test_configs_are_independent_and_preserve_all_nonarchitecture_training_values():
    paths = [Path(f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py")
             for arm in ("A0", "A1", "A2", "A3", "A4", "C1")]
    configs = []
    for path in paths:
        source = path.read_text()
        assert "_base_" not in source and "Config.fromfile" not in source and "runpy" not in source
        cfg = Config.fromfile(str(path))
        assert cfg.max_epochs == 15 and cfg.model.finetune_epoch == 0
        configs.append(cfg.to_dict())
    for cfg in configs:
        for key in ("experiment_name", "report_directory", "pmg_experiment"):
            cfg.pop(key)
        for key in ("geometry_mode", "persistent_content"):
            cfg["model"].pop(key)
    assert all(cfg == configs[0] for cfg in configs)


def test_initial_parameters_are_identical_to_the_baseline():
    torch.set_num_threads(4)
    fingerprints = []
    for arm in ("A0", "A1", "A3", "A4", "C1"):
        cfg = Config.fromfile(f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py")
        set_random_seed(17)
        model = build_segmentor(cfg.model)
        model.init_weights()
        fingerprints.append(model_fingerprint(model))
        del model
    assert len(set(fingerprints)) == 1


def test_content_covariance_transport_matches_the_existing_renderer_convention():
    from model.segmentor.pmg_sparse_opus_world import transport_renderer_quaternion
    from model.utils.utils import get_rotation_matrix
    quaternion = torch.nn.functional.normalize(torch.randn(2, 3, 8, 4, dtype=torch.float64), dim=-1)
    angles = torch.randn(2, 3, dtype=torch.float64)
    diagonal = torch.diag_embed(torch.tensor([.1, .2, .3], dtype=torch.float64) ** 2)
    q = get_rotation_matrix(quaternion)
    original = q.transpose(-1, -2) @ diagonal @ q
    moved_q = get_rotation_matrix(transport_renderer_quaternion(quaternion, angles))
    moved = moved_q.transpose(-1, -2) @ diagonal @ moved_q
    rotation = yaw_matrix(angles)[..., None, :, :]
    torch.testing.assert_close(moved, rotation @ original @ rotation.transpose(-1, -2))


def test_actual_forecast_keeps_birth_slots_and_has_gradients_at_all_six_horizons():
    from model.encoder.opus_encoder import _opus_decode_points
    from tools.pmg_world.evaluate import shape_statistics
    cfg = Config.fromfile("config/sparse_opus_gaussian_world/pmg_A4_4096_1024_direct15e.py")
    model = build_segmentor(cfg.model)
    model.init_weights()
    model.num_query = 2
    model.pts_bbox_head.ind_stamps_all = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    model.curr_epoch = 14
    model.eval()
    points = torch.randn(2, 8, 128, 3) * .005 + .5
    outputs = dict(query_feat=torch.randn(2, 8, 256), all_refine_pts=[points],
                   all_cls_scores=[torch.randn(2, 8, 128, 17)])
    ego = torch.randn(2, 1, 256)
    baseline_births = None
    for mode in ("baseline", "translation", "rigid", "free", "prox"):
        model.geometry_mode = mode
        model.zero_grad(set_to_none=True)
        future_points, classes, features, trajectories, masks = model._forecast(outputs, ego, {})
        assert [x.shape[1] for x in future_points] == [3, 4, 5, 6, 7, 8]
        assert len(classes) == len(features) == len(trajectories) == 6
        if mode == "baseline":
            baseline_births = future_points[0][:, 2:].detach().clone()
        else:
            # Before any recurrent feedback differs, new slots exactly retain
            # their native first-step construction.
            torch.testing.assert_close(future_points[0][:, 2:], baseline_births, rtol=0, atol=0)
        (future_points[-1].square().mean() + classes[-1].square().mean()).backward()
        assert model.reg_branch[-1].weight.grad.norm() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if mode in ("translation", "rigid", "prox"):
            diagnostic = shape_statistics(model, dict(outputs, forecast_points_list=future_points,
                                          query_stamps=model.pts_bbox_head.ind_stamps_all), 0)
            bound = .101 if mode == "prox" else .001
            assert max(row["canonical_shape_rms_max_m"] for row in diagnostic) < bound


@pytest.mark.parametrize("initial_opacity,coordinate,expected_future_opacity", [(1., 1.2, 0.), (0., .5, 1.)])
def test_persistent_content_clamps_only_the_renderer_view_of_out_of_roi_points(
        initial_opacity, coordinate, expected_future_opacity):
    from types import SimpleNamespace
    from model.segmentor.pmg_sparse_opus_world import PMGSparseOpusWorld
    from model.encoder.gaussian_encoder.utils import GaussianPrediction
    from model.encoder.opus_encoder import _opus_decode_points
    pc_range = torch.tensor([-40., -40., -1., 40., 40., 5.4])
    raw = torch.full((1, 2, 4, 3), coordinate)
    calls = []

    def render(state, xyz):
        assert (state.means > pc_range[:3]).all() and (state.means < pc_range[3:]).all()
        calls.append(state.opacities.clone())
        return torch.zeros(1, 18, 640000)

    rotations = torch.zeros(1, 2, 4, 4)
    rotations[..., 0] = 1
    gaussian = GaussianPrediction(torch.zeros(1, 2, 4, 3), torch.full((1, 2, 4, 3), .2),
                                  rotations, torch.full((1, 2, 4, 1), initial_opacity), torch.ones(1, 2, 4, 17))
    fake = SimpleNamespace(persistent_content=True, training=False, pretrain=False,
                           gaussian_loss_horizons=[0, 2, 4, 6], pc_range=pc_range,
                           gaussian_head=SimpleNamespace(render_state=render),
                           pts_bbox_head=SimpleNamespace(transformer=SimpleNamespace(
                               encoder=SimpleNamespace(render_eps=1e-4, layers=[None],
                                   attribute_branches={"0": lambda features: (None, None, torch.ones(1, 2, 4, 1))}))))
    outputs = dict(representation=[dict(gaussian=gaussian, query_valid_mask=torch.ones(1, 2, dtype=torch.bool))],
                   query_feat=torch.zeros(1, 2, 256), pmg_state_order=[torch.arange(2)], pmg_angles=[torch.zeros(1, 2)])
    PMGSparseOpusWorld._render_occupancy(fake, outputs, [raw], [], [], torch.tensor([True, True]),
                                       torch.zeros(1, 640000, 3), 1)
    assert calls[0].min() == initial_opacity and calls[1].max() == expected_future_opacity
    assert (raw == coordinate).all()


def test_patch_support_counts_unique_voxels_and_ignores_outside_points():
    from tools.pmg_world.diagnose_geometry import semantic_patch_support
    target = torch.full((200, 200, 16), 17, dtype=torch.long)
    target[100, 100, 3] = 4
    target[101, 100, 3] = 4
    target[102, 100, 3] = 7
    points = torch.tensor([[[.2, .2, .4], [.2, .2, .4], [.6, .2, .4], [100., 100., 100.]],
                           [[.2, .2, .4], [.6, .2, .4], [1., .2, .4], [1., .2, .4]]])
    result = semantic_patch_support(points, target, torch.tensor([-40., -40., -1., 40., 40., 5.4]))
    assert result["queries_with_3_unique_occupied_voxels"] == 1
    assert result["mean_semantic_purity"] == pytest.approx(2/3)
