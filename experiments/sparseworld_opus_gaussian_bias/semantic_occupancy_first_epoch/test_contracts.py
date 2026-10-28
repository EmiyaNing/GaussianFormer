import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mmengine.config import Config

from ..conditional_first_epoch.loss import FullspaceConditionalSemanticOccupancyLoss, conditional_semantic_ce
from .diagnostics import conditional_statistics
from .frequencies import balance_multipliers
from .loss import BalancedConditionalSemanticOccupancyLoss
from .model import joint_log_probabilities, predict_factorized
from .protocol import ARMS, C_CONFIG, D, E


def options():
    value = Config.fromfile(str(C_CONFIG)).loss.occupancy_loss.to_dict()
    value.pop('type')
    return value


def test_all_one_multipliers_match_C_loss_and_gradient_exactly():
    old = FullspaceConditionalSemanticOccupancyLoss(**options())
    new = BalancedConditionalSemanticOccupancyLoss(conditional_class_multipliers=[1.] * 17, **options())
    logits = torch.randn(1, 18, 28, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = (torch.arange(28) % 18)[None]
    labels[:, -1] = 255
    xyz = torch.zeros(1, 28, 3)
    a, b = old.loss_voxel([logits], xyz, labels), new.loss_voxel([logits], xyz, labels)
    torch.testing.assert_close(a, b, rtol=0., atol=0.)
    torch.testing.assert_close(torch.autograd.grad(a, logits, retain_graph=True)[0], torch.autograd.grad(b, logits)[0], rtol=0., atol=0.)


def test_balancing_changes_only_auxiliary_and_keeps_free_gradients():
    old = FullspaceConditionalSemanticOccupancyLoss(**options())
    factors = np.linspace(.5, 3, 17).tolist()
    new = BalancedConditionalSemanticOccupancyLoss(conditional_class_multipliers=factors, **options())
    logits = torch.randn(1, 18, 8, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.tensor([[4, 11, 17, 13, 255, 17, 16, 4]])
    a, b = old.loss_voxel([logits], torch.zeros(1, 8, 3), labels), new.loss_voxel([logits], torch.zeros(1, 8, 3), labels)
    expected = a - conditional_semantic_ce(logits, labels, old.class_weights) + conditional_semantic_ce(logits, labels, new.conditional_class_weights)
    torch.testing.assert_close(b, expected)
    ga, gb = torch.autograd.grad(a, logits, retain_graph=True)[0], torch.autograd.grad(b, logits)[0]
    torch.testing.assert_close(ga[:, 17], gb[:, 17], rtol=0., atol=0.)
    torch.testing.assert_close(ga[:, :, [2, 4, 5]], gb[:, :, [2, 4, 5]], rtol=0., atol=0.)


def test_training_frequency_rule_is_finite_capped_and_supports_car():
    counts = np.array([320973, 452165, 35668, 833291, 5290948, 304646, 54523, 531219, 85807, 819694,
                       1989105, 30324828, 759435, 8676565, 10034484, 34946298, 38693521])
    factors, reference = balance_multipliers(counts)
    assert reference == pytest.approx(float(counts.mean()))
    assert factors.min() >= .5 and factors.max() <= 3
    assert factors[4] > factors[11] and factors[10] > factors[11]
    assert np.isfinite(factors).all()
    with pytest.raises(ValueError):
        balance_multipliers(np.zeros(17))


def test_joint_probabilities_and_binary_decision_are_independent_of_semantic_logits():
    foreground = torch.randn(2, 17, 11, generator=torch.Generator().manual_seed(17), requires_grad=True)
    gate = torch.linspace(-3, 3, 22).reshape(2, 11).requires_grad_(True)
    joint = joint_log_probabilities(foreground, gate)
    torch.testing.assert_close(joint.exp().sum(1), torch.ones_like(gate), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(joint[:, 17].exp(), (-gate).sigmoid())
    changed = joint_log_probabilities(foreground * 10 + 100, gate)
    torch.testing.assert_close(joint[:, 17], changed[:, 17], rtol=0., atol=0.)
    assert torch.equal(predict_factorized(joint) != 17, predict_factorized(changed) != 17)
    assert torch.equal(predict_factorized(joint) != 17, gate > 0)
    gradient = torch.autograd.grad(-joint[:, 17].mean(), foreground, allow_unused=True)[0]
    assert gradient is None or gradient.count_nonzero() == 0


def test_occupied_joint_ce_equals_binary_ce_plus_conditional_ce():
    raw = torch.randn(1, 17, 8, generator=torch.Generator().manual_seed(17), requires_grad=True)
    gate = torch.randn(1, 8, generator=torch.Generator().manual_seed(19), requires_grad=True)
    labels = torch.tensor([[4, 11, 17, 13, 255, 17, 16, 4]])
    joint = joint_log_probabilities(raw, gate)
    weights = torch.linspace(.1, 1.8, 18)
    valid, occupied = labels != 255, (labels < 17)
    selected = weights[labels.masked_fill(~valid, 0)] * valid
    binary = F.binary_cross_entropy_with_logits(gate, occupied.float(), reduction='none')
    semantic = F.cross_entropy(raw, labels.masked_fill(~occupied, 255), reduction='none', ignore_index=255)
    expected = ((binary + semantic) * selected).sum() / selected.sum()
    actual = F.cross_entropy(joint, labels, weight=weights, ignore_index=255)
    torch.testing.assert_close(actual, expected)
    auxiliary = conditional_semantic_ce(joint, labels, weights)
    gate_gradient = torch.autograd.grad(auxiliary, gate)[0]
    torch.testing.assert_close(gate_gradient, torch.zeros_like(gate_gradient), atol=1e-7, rtol=0.)


@pytest.mark.parametrize('arm', [D, E])
def test_conditional_gradient_diagnosis_matches_autograd(arm):
    cfg = Config.fromfile(str(ARMS[arm]))
    logits = torch.randn(1, 18, 9, generator=torch.Generator().manual_seed(17), requires_grad=True)
    labels = torch.tensor([[17, 4, 11, 13, 255, 17, 16, 4, 17]])
    camera = torch.tensor([[True, True, False, True, False, True, False, False, False]])
    weights = logits.new_tensor(cfg.loss.occupancy_loss.manual_class_weight)
    auxiliary_weights = weights[:17] * logits.new_tensor(cfg.loss.occupancy_loss.get('conditional_class_multipliers', [1.] * 17))
    auxiliary = conditional_semantic_ce(logits, labels, auxiliary_weights)
    base = 10 * F.cross_entropy(logits, labels, weight=weights, ignore_index=255)
    ag, bg = torch.autograd.grad(auxiliary, logits, retain_graph=True)[0], torch.autograd.grad(base, logits)[0]
    record = conditional_statistics(logits.detach(), labels, camera, cfg, 0)
    assert record['conditional_semantic_loss'] == pytest.approx(float(auxiliary.detach()), abs=1e-6)
    assert record['auxiliary_logit_gradient_norm'] == pytest.approx(float(ag.norm()), abs=1e-6)
    assert record['base_18class_ce_logit_gradient_norm'] == pytest.approx(float(bg.norm()), abs=1e-6)
    assert record['auxiliary_base_ce_logit_gradient_cosine'] == pytest.approx(float((ag*bg).sum()/(ag.norm()*bg.norm())), abs=1e-6)
    assert sum(v['auxiliary_loss_contribution'] for v in record['classes']) == pytest.approx(float(auxiliary.detach()), abs=1e-6)


def test_standalone_configs_preserve_schedule_geometry_and_all_other_losses():
    base = Config.fromfile(str(C_CONFIG))
    for name, path in ARMS.items():
        text = path.read_text()
        assert '_base_' not in text and 'Config.fromfile' not in text and 'runpy' not in text
        cfg = Config.fromfile(str(path))
        assert cfg.max_epochs == 15 and cfg.loss.mask_type == 'none'
        assert cfg.loss.occupancy_loss.manual_class_weight == base.loss.occupancy_loss.manual_class_weight
        assert cfg.loss.occupancy_loss.multi_loss_weights == base.loss.occupancy_loss.multi_loss_weights
        assert cfg.loss.strict_loss == base.loss.strict_loss
        assert cfg.optimizer == base.optimizer and cfg.model.gaussian_loss_horizons == base.model.gaussian_loss_horizons
        assert cfg.model.finetune_epoch == base.model.finetune_epoch
        assert cfg.model.pts_bbox_head.transformer.encoder.scale_max == .4
        if name == E:
            assert cfg.loss == base.loss and cfg.model.gaussian_head.occupancy_threshold == .5
            assert cfg.model.gaussian_head.occupancy_prior_offset == pytest.approx(math.log(17))


def test_paired_scene_bootstrap_and_GT_mismatch_guard():
    from .report import paired_bootstrap
    baseline = np.zeros((8, 18, 18), dtype=np.int64)
    candidate = baseline.copy()
    for c in range(17):
        baseline[:, c, c] = 20
        baseline[:, c, 17] = 80
        candidate[:, c, c] = 40
        candidate[:, c, 17] = 60
    baseline[:, 17, 17] = candidate[:, 17, 17] = 1000
    result = paired_bootstrap(candidate, baseline, draws=300)
    assert result['delta_miou_pp'] == pytest.approx(20.)
    assert result['meets_prespecified_first_epoch_screen']
    corrupt = candidate.copy()
    corrupt[0, 4, 4] += 1
    with pytest.raises(ValueError, match='GT differs'):
        paired_bootstrap(corrupt, baseline)


@pytest.mark.parametrize('arm', [D, E])
def test_entry_keeps_15_epoch_LR_and_pauses_at_complete_first_epoch(tmp_path, monkeypatch, arm):
    import sys
    import json
    from .. import train as runner
    from . import train as entry
    cfg = Config.fromfile(str(ARMS[arm]))
    cfg.report_directory = str(tmp_path / 'analysis')
    Path(cfg.report_directory).mkdir()
    path = tmp_path / 'standalone.py'
    cfg.dump(str(path))
    work = tmp_path / 'work'
    work.mkdir()
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([dict(params=[parameter], lr=.0002, base_lr=.0002)])
    advanced = []

    def simulated_runner():
        runner.set_lr(optimizer, 0, 15360, 500, .001)
        runner.atomic_save(work / 'epoch_01.pth', dict(epoch=1, next_iter=0, phase='train', updates=1024,
            config_sha256='config', manifest_sha256={'train':'train','val':'val'}))
        advanced.append(True)

    monkeypatch.setattr(runner, 'main', simulated_runner)
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path), '--work-dir', str(work), '--stop-after-epoch', '1'])
    entry.main()
    assert not advanced
    budget = json.loads((work / 'optimization_budget.json').read_text())
    boundary = json.loads((work / 'first_epoch_completed.json').read_text())
    assert budget['total_updates'] == 15360 and budget['planned_epochs'] == 15
    assert boundary['updates'] == 1024 and boundary['trained_epochs'] == 1
    assert (work / 'latest.pth').exists()
