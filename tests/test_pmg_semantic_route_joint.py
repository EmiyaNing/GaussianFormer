"""Joint training must update both sides and honor the epoch-6 LR boundary."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.segmentor.pmg_semantic_gaussian_route_world import select_children
from model.segmentor.pmg_semantic_route_joint_world import PMGSemanticRouteJointWorld
from tools.pmg_world_native_adapt.joint_optimization import (
    make_optimizer,
    parameter_role,
    set_learning_rates,
)


def test_selected_children_backpropagate_to_both_forecast_paths():
    reference = GaussianPrediction(
        *(torch.randn(1, 2, 3, d, requires_grad=True) for d in (3, 3, 4, 1, 17))
    )
    candidate = GaussianPrediction(
        *(torch.randn(1, 2, 3, d, requires_grad=True) for d in (3, 3, 4, 1, 17))
    )
    mask = torch.tensor([[[True, False, True], [False, True, False]]])
    routed = select_children(reference, candidate, mask)
    sum(value.square().sum() for value in routed if value is not None).backward()
    for ref, cand in zip(reference, candidate):
        if ref is None:
            continue
        assert ref.grad[~mask].norm() > 0
        assert cand.grad[mask].norm() > 0
        assert torch.count_nonzero(ref.grad[mask]) == 0
        assert torch.count_nonzero(cand.grad[~mask]) == 0


def test_training_mode_enables_perception_and_both_forecast_paths():
    model = PMGSemanticRouteJointWorld.__new__(PMGSemanticRouteJointWorld)
    nn.Module.__init__(model)
    model.img_backbone = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))
    model.reference_forecast_modules = nn.ModuleDict({"branch": nn.Dropout(0.5)})
    model.query_update_gate = nn.Dropout(0.5)
    model.eval()
    model.train()
    assert all(module.training for module in model.modules())


def test_optimizer_coverage_and_lr_ratio_change_at_epoch_six():
    model = nn.Module()
    model.img_backbone = nn.Linear(2, 2)
    model.gaussian_head = nn.Module()
    model.gaussian_head.empty_bias = nn.Parameter(torch.zeros(()))
    model.reg_branch = nn.Linear(2, 2)
    model.reference_forecast_modules = nn.ModuleDict({"reg_branch": nn.Linear(2, 2)})
    cfg = SimpleNamespace(
        base_lr=2e-4,
        weight_decay=0.01,
        warmup_iters=500,
        min_lr_ratio=0.001,
        perception_lr=SimpleNamespace(equal_epochs=5, late_ratio=0.1),
    )
    optimizer = make_optimizer(model, cfg)
    optimized = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(optimized) == len(set(optimized)) == len(list(model.parameters()))
    for epoch, update, ratio in (
        (0, 0, 1),
        (4, 24664, 1),
        (5, 24665, 0.1),
        (14, 73994, 0.1),
    ):
        rates = set_learning_rates(optimizer, update, 73995, epoch, cfg)
        assert rates["perception_lr"] / rates["prediction_lr"] == pytest.approx(ratio)
    # Check optimizer state round-trip immediately across the policy boundary.
    saved = optimizer.state_dict()
    restored = make_optimizer(model, cfg)
    restored.load_state_dict(saved)
    assert set_learning_rates(restored, 24665, 73995, 5, cfg) == set_learning_rates(
        optimizer, 24665, 73995, 5, cfg
    )
    with pytest.raises(ValueError, match="no declared"):
        parameter_role("unknown.weight")
    model.img_backbone.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="frozen"):
        make_optimizer(model, cfg)


def test_joint_reports_pair_ground_truth_without_requiring_same_current_predictions(tmp_path):
    import json
    import numpy as np

    from tools.pmg_world.regions import REGIONS
    from tools.pmg_world_native_adapt.semantic_route_joint_report import load_joint_sources
    from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics

    samples = np.array([7] * 124 + [6] * 26)
    scenes = np.array([f'scene-{i:03d}' for i in range(150)])
    sources = []
    for arm in ('baseline', 'joint'):
        directory = tmp_path / arm
        directory.mkdir()
        counts = np.zeros((150, 7, 18, 18), dtype=np.int64)
        counts[:, :, 4, 4] = samples[:, None]
        if arm == 'joint':
            # Current prediction changes, while sample order and GT stay fixed.
            counts[0, 0, 4, 17] = counts[0, 0, 4, 4]
            counts[0, 0, 4, 4] = 0
        regions = np.zeros((150, 3, len(REGIONS), 18, 18), dtype=np.int64)
        metric = HorizonMetrics()
        metric.samples = 1024
        metric.confusion = counts.sum(0)
        (directory / 'epoch_01_metrics.json').write_text(json.dumps(metric.results()))
        np.savez_compressed(directory / 'epoch_01_counts.npz', scenes=scenes,
                            sample_counts=samples, scene_confusion=counts,
                            confusion=counts.sum(0), scene_region_confusion=regions,
                            region_confusion=regions.sum(0))
        sources.append((arm, directory, 'epoch_01'))
    metrics, *_ = load_joint_sources(sources)
    assert metrics[0]['horizons'][0]['miou'] != metrics[1]['horizons'][0]['miou']
    # A different GT distribution must still fail, even when the totals agree.
    path = sources[-1][1] / 'epoch_01_counts.npz'
    with np.load(path) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays['scene_confusion'][0, 0, 3] = arrays['scene_confusion'][0, 0, 4]
    arrays['scene_confusion'][0, 0, 4] = 0
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        load_joint_sources(sources)
