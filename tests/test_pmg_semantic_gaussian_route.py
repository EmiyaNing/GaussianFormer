"""Selection must retain every static-child attribute and use only reference semantics."""

import pytest
import torch

from model.encoder.gaussian_encoder.utils import GaussianPrediction
from model.segmentor.pmg_semantic_gaussian_route_world import semantic_mask, select_children


def pack(offset):
    shape = (1,2,3)
    return GaussianPrediction(*(torch.arange(6*d,dtype=torch.float32).reshape(*shape,d)+offset
                                for d in (3,3,4,1,17)))


def test_mixed_query_routes_whole_children_and_excludes_traffic_cones():
    logits = torch.full((1,2,3,17),-10.)
    labels = torch.tensor([[[4,11,8],[16,7,10]]])
    logits.scatter_(-1,labels[...,None],10.)
    expected = torch.tensor([[[True,False,False],[False,True,True]]])
    assert torch.equal(semantic_mask(logits),expected)
    ref,candidate = pack(0),pack(1000)
    selected = select_children(ref,candidate,expected)
    for actual,left,right in zip(selected,ref,candidate):
        if actual is None:
            assert left is None and right is None
            continue
        assert torch.equal(actual[~expected],left[~expected])
        assert torch.equal(actual[expected],right[expected])


def test_candidate_reclassification_does_not_change_reference_routing():
    ref,candidate = pack(0),pack(1000)
    logits = torch.zeros(1,2,3,17)
    logits[...,11] = 1
    candidate = candidate._replace(semantics=torch.zeros_like(candidate.semantics))
    candidate.semantics[...,4] = 100
    mask = semantic_mask(logits)
    assert not mask.any()
    result = select_children(ref,candidate,mask)
    assert all((left is None and right is None) or torch.equal(left,right) for left,right in zip(result,ref))
    all_candidate = select_children(ref,candidate,torch.ones_like(mask))
    assert all((left is None and right is None) or torch.equal(left,right) for left,right in zip(all_candidate,candidate))


def test_misaligned_child_slots_fail_instead_of_broadcasting():
    with pytest.raises(ValueError,match='align'):
        select_children(pack(0),pack(1),torch.ones(1,2,1,dtype=torch.bool))
