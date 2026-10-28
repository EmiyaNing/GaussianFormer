"""Regression contracts against the frozen, completed D experiment."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from tools.sparse_opus_gausisan_world.protocol import (
    MODEL_NAME,
    canonical_training_config,
    check_resume_identity,
    checkpoint_epoch_index,
    training_contract_sha256,
)


NEW = Path(
    "config/sparse_opus_gausisan_world/sparse_opus_gausisan_world_4096_1024_15e.py"
)
OLD = Path(
    "config/sparseworld_opus_gaussian_bias/sparseworld_opus_gaussian_bias_balanced_conditional_D_continue_4096_1024_15e.py"
)


def configs():
    return Config.fromfile(str(OLD)), Config.fromfile(str(NEW))


def test_standalone_config_and_production_imports_do_not_load_experiments():
    source = NEW.read_text()
    assert (
        "_base_" not in source
        and "Config.fromfile" not in source
        and "runpy" not in source
    )
    code = f"""from mmengine.config import Config
from mmseg.models import SEGMENTORS
from mmengine.registry import MODELS
import sys
cfg=Config.fromfile({str(NEW)!r})
import tools.sparse_opus_gausisan_world.train
import tools.sparse_opus_gausisan_world.evaluate
assert cfg.model.type=={MODEL_NAME!r}
assert SEGMENTORS.get(cfg.model.type) is MODELS.get(cfg.model.type)
assert not any(k=='experiments' or k.startswith('experiments.') for k in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_every_training_configuration_value_matches_D_after_registry_rename():
    old, new = configs()
    assert canonical_training_config(old) == canonical_training_config(new)
    assert training_contract_sha256(new) == new.legacy_training_contract_sha256
    assert new.model.gaussian_loss_horizons == [0, 2, 4, 6]
    assert new.max_epochs == 15 and new.loss.mask_type == "none"


def test_initial_state_parameter_order_and_all_tensors_match_D_exactly():
    torch.set_num_threads(4)
    old, new = configs()
    set_random_seed(17)
    legacy = build_segmentor(old.model)
    legacy.init_weights()
    set_random_seed(17)
    current = build_segmentor(new.model)
    current.init_weights()
    assert list(dict(legacy.named_parameters())) == list(
        dict(current.named_parameters())
    )
    a, b = legacy.state_dict(), current.state_dict()
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)
    current.load_state_dict(a, strict=True)


@pytest.mark.parametrize(
    "masked,multiple", [(False, False), (True, False), (False, True)]
)
def test_D_voxel_loss_and_output_gradients_are_exactly_unchanged(masked, multiple):
    old, new = configs()
    from loss import OPENOCC_LOSS

    a = OPENOCC_LOSS.build(old.loss.occupancy_loss)
    b = OPENOCC_LOSS.build(new.loss.occupancy_loss)
    values = [
        torch.randn(
            1, 18, 36, generator=torch.Generator().manual_seed(seed), requires_grad=True
        )
        for seed in ([17, 19] if multiple else [17])
    ]
    labels = (torch.arange(36) % 18)[None]
    labels[:, -1] = 255
    mask = (torch.arange(36)[None] % 3 != 1) if masked else None
    xyz = torch.zeros(1, 36, 3)
    left = a.loss_voxel(values, xyz, labels, mask)
    right = b.loss_voxel(values, xyz, labels, mask)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    ga = torch.autograd.grad(left, values, retain_graph=True)
    gb = torch.autograd.grad(right, values)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_conditional_helper_excludes_free_ignore_and_free_logit_gradient():
    configs()
    from loss.sparse_opus_gausisan_world_loss import conditional_semantic_ce

    logits = torch.randn(
        1, 18, 5, generator=torch.Generator().manual_seed(17), requires_grad=True
    )
    labels = torch.tensor([[4, 17, 255, 12, 16]])
    loss = conditional_semantic_ce(logits, labels, torch.ones(17))
    gradient = torch.autograd.grad(loss, logits)[0]
    assert (
        gradient[:, 17].count_nonzero() == 0
        and gradient[:, :, [1, 2]].count_nonzero() == 0
    )
    free = conditional_semantic_ce(logits, torch.full_like(labels, 17), torch.ones(17))
    assert free.item() == 0


def test_legacy_resume_allows_only_verified_D_training_semantics():
    old, new = configs()
    manifests = {"train": "a", "val": "b"}
    saved = dict(
        config_sha256=new.legacy_checkpoint_config_sha256[-1], manifest_sha256=manifests
    )
    assert (
        check_resume_identity(saved, new, "new-config-hash", manifests)
        == "verified_legacy_D_registry_rename"
    )
    changed = copy.deepcopy(new)
    changed.optimizer.optimizer.lr *= 2
    with pytest.raises(ValueError, match="incompatible"):
        check_resume_identity(saved, changed, "changed-config", manifests)
    with pytest.raises(ValueError, match="manifests"):
        check_resume_identity(
            saved, new, "new-config-hash", {"train": "changed", "val": "b"}
        )


def test_integrated_checkpoint_accepts_report_changes_and_rejects_loss_changes():
    _, cfg = configs()
    manifests = {"train": "a", "val": "b"}
    saved = dict(
        training_contract_sha256=training_contract_sha256(cfg),
        manifest_sha256=manifests,
    )
    cfg.report_directory = "analysis/another_report"
    assert (
        check_resume_identity(saved, cfg, "changed-file-hash", manifests)
        == "matching_training_contract"
    )
    cfg.loss.occupancy_loss.conditional_semantic_weight = 2.0
    with pytest.raises(ValueError, match="training configuration"):
        check_resume_identity(saved, cfg, "changed-file-hash", manifests)


@pytest.mark.parametrize(
    "saved,expected",
    [
        ({"epoch_index": 14}, 14),
        ({"epoch": 15, "next_iter": 0, "phase": "train"}, 14),
        ({"epoch": 6, "next_iter": 100, "phase": "train"}, 6),
        ({"epoch": 6, "next_iter": 4096, "phase": "evaluate"}, 6),
    ],
)
def test_epoch_metadata_supports_legacy_selected_and_resumable_checkpoints(
    saved, expected
):
    assert checkpoint_epoch_index(saved) == expected


def test_dataset_token_order_and_augmentation_seed_match_D(tmp_path, monkeypatch):
    configs()
    from dataset.sparseworld_strict import NuScenesSparseWorldStrictDataset
    from dataset.sparse_opus_gausisan_world_dataset import (
        SparseOpusGaussianWorldDataset,
    )
    from experiments.sparseworld_opus_gaussian_bias.dataset import (
        SparseWorldBiasSubsetDataset,
    )

    def initialize(self, **kwargs):
        self.scene_infos = [[{"token": "a"}, {"token": "b"}, {"token": "c"}]]
        self.keyframes = [(0, 0), (0, 1), (0, 2)]

    monkeypatch.setattr(NuScenesSparseWorldStrictDataset, "__init__", initialize)
    monkeypatch.setattr(
        NuScenesSparseWorldStrictDataset,
        "__getitem__",
        lambda self, index: np.random.rand(6),
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(dict(num_samples=2, samples=[{"token": "c"}, {"token": "a"}]))
    )
    a, b = SparseWorldBiasSubsetDataset(str(path)), SparseOpusGaussianWorldDataset(
        str(path)
    )
    assert a.keyframes == b.keyframes == [(0, 2), (0, 0)]
    a.set_epoch(4)
    b.set_epoch(4)
    before = np.random.get_state()
    assert np.array_equal(a[1], b[1])
    after = np.random.get_state()
    assert (
        before[0] == after[0]
        and np.array_equal(before[1], after[1])
        and before[2:] == after[2:]
    )


def test_diagnostic_statistics_preserve_D_values_and_all_GT_occupied_voxels():
    old, new = configs()
    from tools.sparse_opus_gausisan_world import diagnostic_statistics as integrated
    from experiments.sparseworld_opus_gaussian_bias.ab_first_epoch.diagnostics import (
        classification_statistics,
    )
    from experiments.sparseworld_opus_gaussian_bias.semantic_occupancy_first_epoch.diagnostics import (
        conditional_statistics as balanced_semantic_statistics,
    )
    from experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.diagnostics import (
        conditional_confusion,
    )

    logits = torch.randn(1, 18, 72, generator=torch.Generator().manual_seed(17))
    labels = (torch.arange(72) % 18).reshape(1, 3, 6, 4)
    labels[:, -1, -1, -1] = 255
    camera = (torch.arange(72) % 3 != 0).reshape_as(labels)
    left = classification_statistics(logits, labels, camera, old, 4)
    right = integrated.classification_statistics(logits, labels, camera, new, 4)
    left.pop("scope")
    right.pop("scope")
    assert left == right
    left = balanced_semantic_statistics(logits, labels, camera, old, 4)
    right = integrated.conditional_statistics(logits, labels, camera, new, 4)
    left.pop("scope")
    right.pop("scope")
    assert left == right
    for region, hist in conditional_confusion(logits, labels, camera).items():
        assert np.array_equal(
            hist, integrated.conditional_confusion(logits, labels, camera)[region]
        )
    assert integrated.conditional_confusion(logits, labels, camera)[
        "full"
    ].sum() == int((labels < 17).sum())


def test_metrics_match_D_at_every_horizon_and_class():
    from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics
    from experiments.sparseworld_opus_gaussian_bias.metrics import (
        HorizonMetrics as LegacyMetrics,
    )

    rng = np.random.default_rng(17)
    a, b = LegacyMetrics(), HorizonMetrics()
    counts = rng.integers(0, 10000, size=(7, 18, 18))
    a.confusion[:] = counts
    b.confusion[:] = counts
    assert a.results() == b.results()


@pytest.mark.parametrize(
    "has_resume,next_iter,phase,expected",
    [
        (False, 0, "train", False),
        (True, 0, "train", False),
        (True, 2048, "train", True),
        (True, 4096, "evaluate", True),
    ],
)
def test_resume_clears_statistics_only_at_new_epoch_boundaries(
    has_resume, next_iter, phase, expected
):
    from tools.sparse_opus_gausisan_world.runtime import should_restore_query_statistics

    assert (
        should_restore_query_statistics({} if has_resume else None, next_iter, phase)
        is expected
    )


def test_optimizer_and_schedule_match_D_for_all_15360_updates():
    from torch import nn
    from tools.sparse_opus_gausisan_world import runtime
    from experiments.sparseworld_opus_gaussian_bias.balanced_D_full15e import (
        train as legacy,
    )

    old, new = configs()

    def tiny_model():
        model = nn.Module()
        model.img_backbone = nn.Linear(2, 2)
        model.sampling_offset = nn.Linear(2, 2)
        model.gaussian_head = nn.Module()
        model.gaussian_head.empty_bias = nn.Parameter(torch.zeros(()))
        return model

    a, b = legacy.make_optimizer(tiny_model(), old), runtime.make_optimizer(
        tiny_model(), new
    )
    for step in range(15360):
        legacy.set_lr(a, step, 15360, old.warmup_iters, old.min_lr_ratio)
        runtime.set_lr(b, step, 15360, new.warmup_iters, new.min_lr_ratio)
        for left, right in zip(a.param_groups, b.param_groups):
            assert left["lr"] == right["lr"]
            assert left["weight_decay"] == right["weight_decay"]
