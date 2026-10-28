"""Gradient isolation and deterministic modes for trainable GS routing."""

import torch
from torch import nn

from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.segmentor.pmg_semantic_gaussian_route_world import select_children
from model.segmentor.pmg_semantic_route_train_world import PMGSemanticRouteTrainWorld


def test_render_selection_only_backpropagates_to_selected_candidate_children():
    dims=(3,3,4,1,17)
    reference=GaussianPrediction(*(torch.randn(1,2,3,d) for d in dims))
    candidate=GaussianPrediction(*(torch.randn(1,2,3,d,requires_grad=True) for d in dims))
    mask=torch.tensor([[[True,False,True],[False,False,True]]])
    selected=select_children(reference,candidate,mask)
    sum(value.square().sum() for value in selected if value is not None).backward()
    for value in candidate:
        if value is None:continue
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad[~mask])==0
        assert value.grad[mask].norm()>0
    assert all(not value.requires_grad for value in reference if value is not None)


def test_outer_training_flag_keeps_frozen_children_deterministic():
    # Exercise the actual train override without constructing a full backbone.
    model=PMGSemanticRouteTrainWorld.__new__(PMGSemanticRouteTrainWorld)
    nn.Module.__init__(model)
    model.block=nn.Sequential(nn.Dropout(.5),nn.BatchNorm1d(3))
    model.reference_forecast_modules=nn.ModuleDict({'branch':nn.Dropout(.5)})
    assert model.train() is model
    assert model.training and not any(child.training for child in model.children())
    assert all(not module.training for module in model.block.modules())
    model.eval()
    assert not model.training
