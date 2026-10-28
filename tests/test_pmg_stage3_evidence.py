"""Bounded position corrections and rank-preserving semantic controls."""

from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model.segmentor.pmg_evidence_residual_world import PMGEvidenceResidualWorld
from tools.pmg_world_stage3.confirm_semantics import make_with_rank_controls


def shell(zero):
    model=PMGEvidenceResidualWorld.__new__(PMGEvidenceResidualWorld)
    nn.Module.__init__(model)
    model.num_refines=3
    model.correction_radius_m=2.
    model._correction_step=0
    model._local_evidence=torch.randn(1,2,256)
    model.motion_correction=nn.Sequential(nn.Linear(513,256),nn.ReLU(),nn.Linear(256,12))
    if zero:
        nn.init.zeros_(model.motion_correction[-1].weight)
        nn.init.zeros_(model.motion_correction[-1].bias)
    return model


def test_zero_initialization_and_xy_l2_bound():
    model=shell(True)
    assert torch.equal(model.correction(torch.randn(1,2,256)),torch.zeros(1,2,3,3))
    model=shell(False)
    result=model.correction(torch.randn(1,2,256)*1000)
    assert (result[...,:2].norm(dim=-1)<=2.00001).all()
    assert torch.equal(result[...,2],torch.zeros_like(result[...,2]))


def test_semantic_rank_control_preserves_complete_score_multiset():
    State=namedtuple("State","means scales rotations opacities semantics")
    state=State(torch.zeros(1,2,3,3),torch.ones(1,2,3,3)*.2,
        torch.tensor([1.,0.,0.,0.]).expand(1,2,3,4),torch.ones(1,2,3,1),
        torch.arange(1.,103.).reshape(1,2,3,17))
    mapping=dict(indices=np.array([0,2]),targets=np.array([[1.,0.,0.],[0.,1.,0.]]),labels=np.array([4,7]))
    states,_=make_with_rank_controls(state,mapping,1,17)
    swapped=states['D01_rank_swap']
    assert torch.equal(swapped.semantics.sort(-1).values,state.semantics.sort(-1).values)
    assert torch.equal(swapped.means,state.means)
    assert torch.equal(states['D11_r20_rank_swap'].means,states['D10_r20'].means)
