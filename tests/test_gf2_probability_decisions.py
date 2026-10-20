import numpy as np
import torch
from tools.gf2_probability_decisions import predict, temperature, risk_score


def test_joint_and_hierarchical_endpoints_have_different_risks():
    q=np.full((1,17),.8/16);q[0,0]=.2
    o=np.array([.7])
    assert predict(o,q,ratio=1).item()==17
    assert predict(o,q,ratio=0).item()==0
    assert .3 > .7*.2  # Joint MAP is optimal for flat0-1 category risk here.


def test_score_is_exact_bayes_action_for_all_registered_costs():
    rng=np.random.default_rng(928);q=rng.dirichlet(np.ones(17),200);o=rng.uniform(.01,.99,200)
    for ratio in [0,.25,.5,.75,1]:
        expected=np.where((1-o)+ratio*o*(1-q.max(-1)) <= o,q.argmax(-1),17)
        np.testing.assert_array_equal(predict(o,q,ratio),expected)


def test_temperature_keeps_order_and_hierarchical_decision():
    rng=np.random.default_rng(928);q=rng.dirichlet(np.ones(17),100);o=rng.uniform(0,1,100)
    for t in [.5,.75,1,1.5,2]:
        transformed=temperature(q,t)
        np.testing.assert_allclose(transformed.sum(-1),1)
        np.testing.assert_array_equal(transformed.argmax(-1),q.argmax(-1))
        np.testing.assert_array_equal(predict(o,transformed,0),predict(o,q,0))
    np.testing.assert_array_equal(temperature(q,1),q)


def test_probability_empty_nll_has_no_direct_semantic_gradient():
    raw=torch.tensor([-.3,.7,.2],dtype=torch.float64,requires_grad=True)
    o=torch.tensor(.6,dtype=torch.float64,requires_grad=True)
    q=raw.softmax(0)
    p=torch.cat([o*q,(1-o).reshape(1)])
    loss=-p[-1].log()
    gu,go=torch.autograd.grad(loss,(raw,o))
    torch.testing.assert_close(gu,torch.zeros_like(gu))
    torch.testing.assert_close(go,torch.tensor(2.5,dtype=torch.float64))


def test_unsupported_voxels_stay_empty_for_finite_thresholds():
    q=np.full((2,17),1/17);o=np.array([0.,1.])
    for ratio in [0,.5,1]:
        np.testing.assert_array_equal(predict(o,q,ratio),[17,0])


def test_grouped_softmax_reproduces_original_logit_argmax():
    rng=np.random.default_rng(928);z=rng.normal(size=(100,18))*4
    exp=np.exp(z-z.max(-1,keepdims=True));p=exp/exp.sum(-1,keepdims=True)
    o=p[:,:17].sum(-1);q=p[:,:17]/o[:,None]
    np.testing.assert_array_equal(predict(o,q),z.argmax(-1))


def test_float32_temperature_identity_has_zero_margin_correction():
    rng=np.random.default_rng(928)
    q=rng.dirichlet(np.ones(17),100).astype(np.float32)
    qt=temperature(q,1)
    delta=np.log(qt.max(-1))-np.log(q.astype(np.float64).max(-1))
    np.testing.assert_array_equal(delta,np.zeros(100))


def test_budget_threshold_is_reselected_for_each_scene_bootstrap():
    from tools.report_gf2_probability_decomposition import matched_budget
    from tools.calibrate_empty_mechanism_stage2 import EDGES,NB
    # Scene0 consumes its budget at score .5, scene1 at score -.5.
    # Reusing the pooled threshold for either resample would be incorrect.
    def hist(values):
        h=np.zeros((2,NB),np.int64)
        for scene,scores in enumerate(values):
            h[scene]=np.bincount(np.searchsorted(EDGES,scores,side='right'),minlength=NB)
        return h
    curves=dict(empty_hist=hist([[.5,.75],[-.5,-.25]]),
        rescue_hist=hist([[.6],[-.4]]),baseline_correct_hist=hist([[1.],[1.]]),
        baseline_fp=np.array([1,1]),cohort_size=np.array([1,1]))
    weights=np.array([[1.,1.],[2.,0.],[0.,2.]])
    result=matched_budget(curves,weights)
    assert result['feasible'].all()
    np.testing.assert_array_equal(result['fp'],[2,2,2])
    assert result['threshold'][1]>.5 and -.5<result['threshold'][2]<=-.25
    np.testing.assert_array_equal(result['gross'],[1,2,2])


def test_global_threshold_fit_uses_full_range_and_matches_direct_iou():
    from tools.eval_gf2_probability_decomposition import fit_global
    from tools.calibrate_empty_mechanism_stage2 import EDGES,NB,counts_from_hist
    from tools.eval_empty_decision_optimization import metrics
    h=np.zeros((18,17,NB),np.int64)
    # Optimal separating threshold lies outside the [-1,1] tune shifts.
    for y,score,n in [(0,-4.,20),(17,-5.,10),(0,-7.,1)]:
        h[y,0,np.searchsorted(EDGES,score,side='right')]+=n
    threshold,choice=fit_global(h)
    assert -5.<threshold<=-4. and not choice['boundary']
    direct=metrics(counts_from_hist(h,np.full(17,threshold))[None])
    np.testing.assert_allclose(choice['miou'],direct['mIoU'][0])


def test_temperature_keeps_original_maximizer_when_float64_creates_a_tie():
    from tools.gf2_probability_decisions import temperature_preserving_winner
    q=np.full((1,17),0.0588235294117647)
    q[0,4]=0.058823529411764705
    assert q.argmax(-1).item()==4
    plain=temperature(q,.5)
    assert plain.argmax(-1).item()==0
    transformed,winner,ties=temperature_preserving_winner(q,.5)
    assert winner.item()==4 and ties==1
    assert transformed[0,4]==transformed.max()
    unchanged,winner,ties=temperature_preserving_winner(q,1)
    np.testing.assert_array_equal(unchanged,q)
    assert winner.item()==4 and ties==0
