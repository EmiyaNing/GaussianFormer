"""Read-only dependence diagnostics for already frozen decision networks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.empty_decision_followup import RUN,OLD,read,write_json,samples
from tools.fit_empty_decision_followup import Policy,Probability,encode,probability_features
from tools.fit_empty_decision_optimization import Dense


def main():
    data=samples('tune');x=torch.as_tensor(data['X'],device='cuda');fg=torch.as_tensor(data['fg'].astype(np.int64),device='cuda')
    w=data['weight'];stratum=data['fg'].astype(int)*32+np.floor((data['X'][:,0]+2)*4).astype(int)
    permutation=np.arange(len(fg));rng=np.random.default_rng(928)
    for key in np.unique(stratum):
        ids=np.where(stratum==key)[0];permutation[ids]=rng.permutation(ids)
    ids=torch.as_tensor(permutation,device='cuda');result={}
    points=read(RUN/'operating_points.json')['operating_points']
    with torch.inference_mode():
        for family in ['OldP3','H','Mix','Conditional','Support','Safe']:
            if family=='OldP3':
                r=torch.load(OLD/'models/P3_seed928.pt',map_location='cpu',weights_only=False)
                a=encode(x,fg,r['scale']);prob=Probability(ROOT/r['probability_path']);q=prob(x,fg)
                net=Dense(a.shape[1]+3,r['hidden'],1).cuda().eval();net.load_state_dict(r['state_dict'])
                anchor=x[:,0];amplitude=2.
                threshold=next(p for p in read(OLD/'operating_points.json')['operating_points'] if p['family']=='P3' and p['seed']==928 and p['budget']==.005)['threshold']
            else:
                point=next(p for p in points if p['family']==family and p['seed']==928 and p['budget']==.005)
                if point['source']=='anchor':result[family]={'zero_residual':True};continue
                policy=Policy(RUN/'models'/f'{point["source"]}.pt');net=policy.net;a=encode(x,fg,policy.record['scale'])
                q=probability_features(policy.prob(x,fg),family) if policy.prob else None
                anchor=x[:,0]-policy.anchor[fg];amplitude=1.;threshold=point['threshold']
            def residual(prob):return amplitude*net(torch.cat([a,prob],-1) if prob is not None else a)[:,0].tanh()
            raw=residual(q);original=anchor+raw;rr=raw.cpu().numpy()
            class_means=np.array([np.average(rr[data['fg']==c],weights=w[data['fg']==c]) if (data['fg']==c).any() else 0 for c in range(17)])
            entry=dict(mean_residual=float(np.average(rr,weights=w)),residual_rms=float(np.sqrt(np.average(rr**2,weights=w))),
                within_class_residual_std=float(np.sqrt(np.average((rr-class_means[data['fg']])**2,weights=w))),
                saturation_fraction_abs95=float(np.average(np.abs(rr)>.95*amplitude,weights=w)),
                mean_effective_bias_after_threshold=float(np.average(rr-threshold,weights=w)))
            if q is not None:
                changed=anchor+residual(q[ids]);delta=(changed-original).cpu().numpy();flip=((changed>=threshold)!=(original>=threshold)).cpu().numpy()
                entry['probability_permutation']={}
                for index,name in [(0,'camera'),(1,'complement'),(None,'all')]:
                    m=np.ones(len(w),bool) if index is None else data['region']==index
                    entry['probability_permutation'][name]=dict(decision_flip_fraction=float(np.average(flip[m],weights=w[m])),
                        score_change_rms=float(np.sqrt(np.average(delta[m]**2,weights=w[m]))))
            result[family]=entry
    write_json(RUN/'decision_dependence.json',dict(scope='200k weighted training-tune samples; labels not used to form permutations',
        perturbation='permute probability vectors within predicted foreground class and 0.25-wide native margin bins',
        limitation='off-manifold sensitivity diagnostic; not a retrained ablation or causal performance guarantee',models=result))
    print(result)


if __name__=='__main__':main()
