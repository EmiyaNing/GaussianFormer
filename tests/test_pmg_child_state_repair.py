"""Behavioral checks for the two changes combined in M-repair."""

import torch

from model.segmentor.pmg_child_state_repair_world import ChildStateRepairHead
from tools.pmg_world_repair.train import loss_control


def test_stopping_distance_gradient_preserves_values_and_other_gradients():
    audit = loss_control()
    assert audit["forward_loss_exact"]
    assert audit["classification_logit_gradient_exact"]
    assert audit["point_regression_gradient_exact"]


def test_shared_head_identity_bound_and_child_conditioning():
    torch.manual_seed(17)
    head = ChildStateRepairHead()
    future, current = torch.randn(1, 2, 256), torch.randn(1, 2, 256)
    local = torch.randn(1, 2, 3, head.local_dimensions)
    assert torch.equal(head(future, current, local), torch.zeros(1, 2, 3, 3))
    torch.nn.init.normal_(head.child[-1].weight, std=.1)
    result = head(future, current, local)
    assert (result[..., :2].norm(dim=-1) <= 2.000001).all()
    assert torch.equal(result[..., 2], torch.zeros_like(result[..., 2]))
    assert not torch.equal(result[:, :, 0], result[:, :, 1])
    permuted = head(future, current, local[:, :, [2, 0, 1]])
    torch.testing.assert_close(permuted, result[:, :, [2, 0, 1]])
