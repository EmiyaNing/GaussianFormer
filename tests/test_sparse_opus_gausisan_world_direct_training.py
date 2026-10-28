"""The direct-training config must use the formal path from the first epoch."""

from pathlib import Path

import pytest
import torch
from mmengine import Config
from mmseg.models import build_segmentor


BASE = Path("config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_nuscenes_occ3d_15e.py")
DIRECT = BASE.with_name("sparse_opus_gaussian_world_nuscenes_occ3d_direct_15e.py")


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(4)
    cfg = Config.fromfile(str(DIRECT))
    return build_segmentor(cfg.model)


def test_direct_config_is_standalone_and_only_changes_stage_start_and_run_name():
    for forbidden in ("_base_", "Config.fromfile", "runpy"):
        assert forbidden not in DIRECT.read_text()
    base = Config.fromfile(str(BASE)).to_dict()
    direct = Config.fromfile(str(DIRECT)).to_dict()
    assert direct["model"]["finetune_epoch"] == 0
    assert direct["model"]["is_pretrain"] is False
    assert direct["model"]["pts_bbox_head"]["is_pretrain"] is False
    assert direct["train_load_from"] == base["train_load_from"]
    assert direct["load_from"] is None
    assert direct["max_epochs"] == 15
    direct["model"]["finetune_epoch"] = base["model"]["finetune_epoch"]
    direct["experiment_name"] = base["experiment_name"]
    assert direct == base


def test_first_epoch_has_formal_attention_and_no_warmup_mode(model):
    model.train()
    model.set_epoch(0)
    assert not model.pretrain and not model.pts_bbox_head.pretrain
    head = model.pts_bbox_head
    stamps = head.ind_stamps_all
    assert torch.bincount(stamps, minlength=7).tolist() == [720, 60, 60, 60, 60, 40, 40]
    for layer in head.transformer.decoder.decoder_layers:
        mask = layer.self_attn.ind_mask
        assert mask is not None
        assert (mask[stamps == 0][:, stamps > 0] < -1e4).all()
        assert (mask[stamps == 0][:, stamps == 0] == 0).all()


def test_query_assignment_stays_fixed_within_first_epoch_and_validation(model):
    model.train()
    model.set_epoch(0)
    head = model.pts_bbox_head
    before = head.ind_stamps_all.clone()
    statistics = head.num_stamps_all.clone()
    try:
        # A strong new preference must not trigger per-batch reassignment.
        head.num_stamps_all[:, -1] += 1000
        head.update_stamp_assignment(force=head.pretrain)
        assert torch.equal(head.ind_stamps_all, before)
        model.eval()
        head.update_stamp_assignment(force=head.pretrain)
        assert torch.equal(head.ind_stamps_all, before)
    finally:
        head.num_stamps_all.copy_(statistics)


@pytest.mark.parametrize("epoch,steps", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (14, 6)])
def test_actual_forecast_grows_from_half_second_to_three_seconds(model, monkeypatch, epoch, steps):
    model.train()
    model.set_epoch(epoch)
    assert not model.pretrain
    # Exercise the real world transition on a smaller query group; image
    # extraction and the dense Gaussian renderer are unnecessary here.
    monkeypatch.setattr(model, "num_query", 2)
    monkeypatch.setattr(model.pts_bbox_head, "ind_stamps_all", torch.tensor([0, 0, 1, 2, 3, 4, 5, 6]))
    outputs = dict(
        query_feat=torch.zeros(1, 8, 256),
        all_refine_pts=[torch.full((1, 8, 128, 3), 0.5)],
        all_cls_scores=[torch.zeros(1, 8, 128, 17)],
    )
    metas = dict(ego2lidar=torch.eye(4)[None], temporal_trajs=torch.zeros(1, 6, 2))
    with torch.no_grad():
        points, classes, features, trajectories, masks = model._forecast(
            outputs, torch.zeros(1, 1, 256), metas
        )
    assert len(points) == len(classes) == len(features) == len(masks) == steps
    assert len(trajectories) == min(steps + 1, 6)
    assert all(torch.isfinite(p).all() for p in points)
    # Evaluation still predicts the complete six-step sequence from epoch 1.
    model.eval()
    with torch.no_grad():
        evaluated = model._forecast(outputs, torch.zeros(1, 1, 256), metas)
    assert len(evaluated[0]) == 6
