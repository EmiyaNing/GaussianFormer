"""Numerical controls needed to interpret the scientific diagnostic."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from mmengine import Config

from loss.opus_set_loss import OPUSSetLoss
from loss.center_support_diagnostic_loss import CenterSupportDiagnosticSetLoss
from tools.center_support_common import contribution_diagnostic, gradient_summary, kernel_weights
from tools.summarize_center_support import interval, summarize


def batch():
    torch.manual_seed(9)
    p = [torch.randn(2, 7, 3, requires_grad=True) for _ in range(2)]
    z = [torch.randn(2, 7, 3, requires_grad=True) for _ in range(2)]
    y = torch.tensor([[0, 1, 2, 3, 3], [0, 1, 2, 0, 3]])
    return dict(opus_pred_points=p, opus_pred_logits=z,
                opus_point_valid_masks=[torch.tensor([[1, 1, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1]], dtype=torch.bool)] * 2,
                metas=dict(occ_label=y, occ_xyz=torch.randn(2, 5, 3),
                           occ_cam_mask=torch.tensor([[1, 0, 1, 1, 1], [1, 1, 0, 0, 1]], dtype=torch.bool)))


KWARGS = dict(stage_weights=[.7, 1.3], empty_label=3,
              class_weights=[2., 3., 4.], pc_range=[-2, -2, -2, 2, 2, 2])


def test_historical_scalar_and_all_output_gradients_match():
    inputs = batch()
    original, _ = OPUSSetLoss(**KWARGS)(inputs)
    diagnostic, _ = CenterSupportDiagnosticSetLoss(**KWARGS)(inputs)
    torch.testing.assert_close(original, diagnostic)
    targets = inputs['opus_pred_points'] + inputs['opus_pred_logits']
    a = torch.autograd.grad(original, targets, retain_graph=True)
    b = torch.autograd.grad(diagnostic, targets)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(expected, actual)


def test_only_final_terms_change_and_mask_alignment_is_explicit():
    inputs = batch()
    original, _ = CenterSupportDiagnosticSetLoss(**KWARGS).components(inputs)
    removed, _ = CenterSupportDiagnosticSetLoss(**KWARGS, final_geometry_weight=0,
                                               final_classification_weight=.25).components(inputs)
    for key in original:
        if key.startswith('stage_0'):
            torch.testing.assert_close(removed[key], original[key])
        elif key.endswith('classification'):
            torch.testing.assert_close(removed[key], original[key] * .25)
        else:
            assert float(removed[key].detach()) == 0
    _, counts = CenterSupportDiagnosticSetLoss(**KWARGS, final_gt_domain='camera').components(inputs)
    assert counts['stage_0/gt_count'] == 7
    assert counts['stage_1/gt_count'] == 4
    no_camera = deepcopy(inputs['metas'])
    no_camera.pop('occ_cam_mask')
    with pytest.raises(ValueError, match='requires occ_cam_mask'):
        CenterSupportDiagnosticSetLoss(**KWARGS, gt_domain='camera').components(dict(inputs, metas=no_camera))


def test_classification_has_no_direct_center_gradient():
    inputs = batch()
    terms, _ = CenterSupportDiagnosticSetLoss(**KWARGS).components(inputs)
    g = torch.autograd.grad(terms['stage_1/classification'], inputs['opus_pred_points'][-1], allow_unused=True)
    assert g[0] is None
    summary = gradient_summary(g[0], torch.ones(7, 3))
    assert summary['cosine'] is None and summary['missing_a']


def test_candidate_box_is_not_a_three_sigma_ellipsoid():
    xyz = torch.tensor([[.2, 0., 0.], [.6, 0., 0.], [1., 0., 0.]])
    means = torch.tensor([[.2, 0., 0.]], requires_grad=True)
    scales = torch.tensor([[.08, .08, .08]], requires_grad=True)
    inv = torch.diag_embed(scales.reciprocal().square())
    weights = kernel_weights(xyz, means, torch.ones(1), scales, inv, [0, 0, 0])
    # ceil(.08*3/.4) = 1. Neighbor bin survives despite being 5 sigma away.
    torch.testing.assert_close(weights[0], torch.ones(1))
    torch.testing.assert_close(weights[1], torch.tensor([torch.exp(torch.tensor(-12.5))]))
    assert weights[2].item() == 0


def test_responsibility_normalization_and_chunk_invariance():
    xyz = torch.tensor([[.2, 0., 0.], [.6, 0., 0.], [1., 0., 0.]])
    means = torch.tensor([[.2, 0., 0.], [.6, 0., 0.]])
    scales = torch.full((2, 3), .2)
    inv = torch.diag_embed(scales.reciprocal().square())
    opacity = torch.tensor([.2, .7])
    labels = torch.tensor([0, 1, 2])
    semantics = torch.tensor([[2., .1], [.2, 3.]])
    nearest = torch.tensor([0, 1])
    a, at = contribution_diagnostic(xyz, labels, means, opacity, scales, inv,
                                    semantics, nearest, 2, [0, 0, 0], voxel_chunk=1, gaussian_chunk=1)
    b, bt = contribution_diagnostic(xyz, labels, means, opacity, scales, inv,
                                    semantics, nearest, 2, [0, 0, 0], voxel_chunk=3, gaussian_chunk=2)
    weights = kernel_weights(xyz, means, opacity, scales, inv, [0, 0, 0])
    responsibility = weights[:2] / weights[:2].sum(-1, keepdim=True)
    expected_mass = responsibility.sum(0)
    expected_target = responsibility.T / expected_mass[:, None]
    torch.testing.assert_close(at['mass'], expected_mass)
    torch.testing.assert_close(at['target'], expected_target)
    torch.testing.assert_close(at['target'], bt['target'])
    assert a['empty_voxels'] == b['empty_voxels'] == 1
    assert a['occupied_wrong_semantic_contribution_fraction'] == pytest.approx(b['occupied_wrong_semantic_contribution_fraction'])


def test_training_controls_preserve_model_early_supervision_and_voxel_loss():
    root = Path(__file__).resolve().parents[1]
    baseline = Config.fromfile(str(root / 'config/mini_occ3d_4096_1024/opusv1_gaussian_softplus_learned_bias.py'))
    configs = sorted((root / 'config/center_support_contribution1').glob('a*.py'))
    assert len(configs) == 8
    for path in configs:
        cfg = Config.fromfile(str(path))
        assert cfg.model == baseline.model
        assert cfg.loss.loss_cfgs[1] == baseline.loss.loss_cfgs[1]
        assert cfg.loss.loss_cfgs[0].stage_weights == baseline.loss.loss_cfgs[0].stage_weights
        assert cfg.loss_input_convertion == baseline.loss_input_convertion
        assert cfg.optimizer == baseline.optimizer
        assert cfg.load_from == baseline.load_from


def test_scene_interval_and_blocked_status_do_not_fabricate_evidence():
    assert interval([.2])['ci95'] is None
    assert interval([.2] * 8)['ci95'] == pytest.approx([.2, .2])
    blocked = summarize({'status': 'blocked_no_cuda'})
    assert blocked['status'] == 'blocked_no_cuda'
    assert 'branches' not in blocked


def test_unknown_gt_is_excluded_from_aligned_loss():
    inputs = batch()
    inputs['metas']['occ_label'][0, 1] = 255
    _, counts = CenterSupportDiagnosticSetLoss(**KWARGS).components(inputs)
    assert counts['stage_0/gt_count'] == 6
