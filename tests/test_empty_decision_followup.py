"""Boundary preservation and separation of deployable and oracle inputs."""
import numpy as np
import pytest
import torch
from tools.fit_empty_decision_followup import Dense,Policy,Probability,probability_features


def scale():return dict(cols=list(range(8)),mean=np.zeros(8,np.float32),std=np.ones(8,np.float32))


def test_zero_residual_preserves_predicted_class_thresholds_including_ties(tmp_path):
    network=Dense(25,40,1,zero_output=True)
    thresholds=[0.]*17;thresholds[4]=-.46875;thresholds[11]=1.03125
    path=tmp_path/'policy.pt'
    torch.save(dict(probability_path=None,scale=scale(),hidden=40,anchor_thresholds=thresholds,
                    state_dict=network.state_dict()),path)
    model=Policy(path,device='cpu')
    x=torch.zeros(5,12);x[:,0]=torch.tensor([-.5,-.46875,-.3,.5,1.03125])
    fg=torch.tensor([4,4,4,11,11])
    prediction=torch.where(model(x,fg)>=0,fg,17)
    assert prediction.tolist()==[17,4,4,17,11]


def test_joint_probability_does_not_read_actual_region_but_oracle_requires_it(tmp_path):
    x=torch.randn(12,12);fg=torch.arange(12)%17
    for mode,outputs,extra in [('joint6',6,0),('mask_oracle',3,1)]:
        network=Dense(25+extra,32,outputs)
        path=tmp_path/(mode+'.pt')
        torch.save(dict(flavour=mode,scale=scale(),hidden=32,outputs=outputs,state_dict=network.state_dict()),path)
        model=Probability(path,device='cpu')
        if mode=='joint6':
            a=model(x,fg,torch.zeros(12));b=model(x,fg,torch.ones(12))
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(a.sum(-1),torch.ones(12))
        else:
            with pytest.raises(AssertionError,match='oracle-only'):model(x,fg)


def test_conditional_features_preserve_event_information_when_region_prior_changes():
    camera=torch.tensor([.6,.3,.1]);outside=torch.tensor([.1,.8,.1])
    values=torch.stack([torch.cat([.2*camera,.8*outside]),torch.cat([.8*camera,.2*outside])])
    saved=values.clone()
    conditional=probability_features(values,'Conditional')
    torch.testing.assert_close(conditional[0],conditional[1])
    torch.testing.assert_close(values,saved,rtol=0,atol=0)
    assert not torch.equal(values[0],values[1])
