"""Verify the shared-path probe actually sends task gradients to projection."""
import torch
from torch import nn
from tools.diagnose_teacher_alignment import shared_projection
from tools.diagnose_teacher_alignment import pilot_loss_config
from mmengine import Config


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.future_feature_mlp = nn.Linear(4, 4)
        self.student_projection = nn.Linear(4, 4, bias=False)


def test_shared_projection_changes_task_path_and_removes_hook():
    student = TinyStudent()
    mixed = torch.randn(1, 7, 4)
    plain = mixed + student.future_feature_mlp(mixed)
    with shared_projection(student, True):
        shared = mixed + student.future_feature_mlp(mixed)
    torch.testing.assert_close(shared, student.student_projection(plain))
    shared.square().sum().backward()
    assert student.student_projection.weight.grad.abs().sum() > 0
    assert student.future_feature_mlp.weight.grad.abs().sum() > 0
    torch.testing.assert_close(mixed + student.future_feature_mlp(mixed), plain)
    assert shared.shape == plain.shape  # Every query, including born slots, kept.


def test_shared_projection_restores_hook_after_exception():
    student = TinyStudent()
    try:
        with shared_projection(student, True):
            raise RuntimeError('synthetic failure')
    except RuntimeError:
        pass
    assert not student.future_feature_mlp._forward_hooks


def test_kl_ablation_changes_only_semantic_loss_weight():
    cfg = Config.fromfile(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    full = pilot_loss_config(cfg, 'C_full')
    off = pilot_loss_config(cfg, 'E_no_semantic_kl')
    assert [k for k in full if full[k] != off[k]] == ['semantic_weight']
    assert full.semantic_weight == .1 and off.semantic_weight == 0.
    assert full.matcher == off.matcher
    assert full.geometry_weight == off.geometry_weight == .25
    assert full.feature_weight == off.feature_weight == .1


def test_full_training_kl_ablation_preserves_model_and_recipe():
    full = Config.fromfile(
        'config/opus/sparseworld_teacher_aligned_r50_704x256_6h.py')
    off = Config.fromfile(
        'config/opus/sparseworld_teacher_aligned_no_semantic_kl_16e.py')
    differing = {k for k in full if full[k] != off[k]}
    assert differing == {'loss', 'alignment_contract_version'}
    assert {k for k in full.loss if full.loss[k] != off.loss[k]} == {
        'semantic_weight'}
    assert off.max_epochs == full.max_epochs == 16
    assert off.alignment_contract_version != full.alignment_contract_version
