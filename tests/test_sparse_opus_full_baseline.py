"""Scheduling contract and final incomplete accumulation-group checks."""

import pytest
import torch

from model.segmentor.sparse_opus_full_horizon_baseline import SparseOpusFullHorizonBaseline
from model.segmentor.sparse_opus_gausisan_world import SparseOpusGaussianWorld
from tools.pmg_world_full.train import accumulation_divisor


def model_shell(epoch=0, training=True):
    model = SparseOpusFullHorizonBaseline.__new__(SparseOpusFullHorizonBaseline)
    torch.nn.Module.__init__(model)
    model.curr_epoch, model.finetune_epoch, model.num_fu_frames = epoch, 0, 6
    model.train(training)
    return model


def test_first_epoch_unfolds_six_steps_without_changing_actual_epoch(monkeypatch):
    model = model_shell()
    monkeypatch.setattr(SparseOpusGaussianWorld, "_forecast",
                        lambda self, *args: min(self.curr_epoch + 1, self.num_fu_frames))
    assert model._forecast(None, None, None) == 6
    assert model.curr_epoch == 0


def test_rollout_exception_restores_actual_epoch(monkeypatch):
    model = model_shell(epoch=1)

    def failure(self, *args):
        raise RuntimeError("test failure")

    monkeypatch.setattr(SparseOpusGaussianWorld, "_forecast", failure)
    with pytest.raises(RuntimeError, match="test failure"):
        model._forecast(None, None, None)
    assert model.curr_epoch == 1


def test_final_unpaired_microbatch_keeps_full_gradient_weight():
    assert [accumulation_divisor(i, 5, 2) for i in range(5)] == [2, 2, 2, 2, 1]
    assert [accumulation_divisor(i, 6, 2) for i in range(6)] == [2] * 6
