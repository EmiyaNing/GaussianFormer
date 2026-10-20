"""Exploratory endpoint factor swaps; distinct from causal training attribution."""
from __future__ import annotations
import itertools
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage3 import RUN,initialize,read,write_json,file_sha256
from tools.empty_mechanism_stage1 import build_network
from tools.eval_empty_mechanism_stage3 import numpy_counts
from tools.eval_empty_decision_optimization import metrics

FACTORS=['geometry','amplitude','composition','empty']


def shapley(values):
    """Values keyed by four 0/1 factors, each containing identically paired arrays."""
    result={}
    for index,name in enumerate(FACTORS):
        effect=np.zeros_like(values[(0,0,0,0)],dtype=float)
        for bits in itertools.product([0,1],repeat=4):
            if bits[index]:continue
            n=sum(bits);other=list(bits);other[index]=1
            weight=math.factorial(n)*math.factorial(3-n)/math.factorial(4)
            effect+=weight*(values[tuple(other)]-values[bits])
        result[name]=effect
    np.testing.assert_allclose(sum(result.values()),values[(1,1,1,1)]-values[(0,0,0,0)],atol=1e-10,rtol=1e-10)
    return result


def run():
    p=initialize();dest=RUN/'mediation';dest.mkdir(exist_ok=True)
    net,_=build_network(5);net.eval();start=time.monotonic()
    for meta in p['scene_samples']:
        index=meta['index'];path=dest/f'{index:04d}.json'
        if path.exists():continue
        loaded=[np.load(RUN/'evaluation'/name/'gaussians'/f'{index:04d}.npz') for name in ['B0','SE']]
        np.testing.assert_array_equal(loaded[0]['y'],loaded[1]['y'])
        np.testing.assert_array_equal(loaded[0]['xyz'],loaded[1]['xyz'])
        args=[[torch.as_tensor(r[f'args{j}'],device='cuda') for j in range(5)] for r in loaded]
        xyz=torch.as_tensor(loaded[0]['xyz'],device='cuda');y=loaded[0]['y']
        fgsem=[a[2][:,:-1,:17] for a in args]
        amplitude=[s.double().sum(-1,keepdim=True) for s in fgsem]
        composition=[s.double()/a for s,a in zip(fgsem,amplitude)]
        baseline=np.load(RUN/'evaluation/B0/frames'/f'{index:04d}.npz')
        cohort=(y<17)&(baseline['fg']==y)&(baseline['margin']<0)
        rows=[];parity=[]
        with torch.inference_mode():
            for bits in itertools.product([0,1],repeat=4):
                geom,amp,comp,empty=bits
                sem=fgsem[amp] if amp==comp else (amplitude[amp]*composition[comp]).float()
                sem=torch.cat([sem,torch.zeros_like(sem[:,:,:1])],-1)
                sem=torch.cat([sem,args[empty][2][:,-1:]],1).contiguous()
                means,opa,_,scales,inverse=args[geom]
                scores=net.head.aggregator(xyz.float().contiguous(),means.contiguous(),
                    opa.reshape(1,-1).contiguous(),sem,scales.contiguous(),inverse.contiguous()).T
                bias=float(loaded[empty]['bias']);scores[17]-=bias
                top,fg=scores[:17].max(0);margin=top-scores[17]
                pred=torch.where(margin>=0,fg,17).cpu().numpy()
                rows.append(dict(bits=list(bits),counts=numpy_counts(y,pred).tolist(),
                    rescued=int((cohort&(pred==y)).sum()),cohort=int(cohort.sum())))
                if bits in [(0,0,0,0),(1,1,1,1)]:
                    name='B0' if sum(bits)==0 else 'SE'
                    native=np.load(RUN/'evaluation'/name/'frames'/f'{index:04d}.npz')
                    expected=np.where(native['margin']>=0,native['fg'],17)
                    error=float(np.abs(margin.cpu().numpy()-native['margin']).max())
                    mismatch=int((expected!=pred).sum())
                    if error>5e-4:raise RuntimeError(f'Endpoint renderer parity failed: {error}')
                    parity.append(dict(model=name,max_margin_error=error,prediction_disagreements=mismatch))
        write_json(path,dict(frame=meta,conditions=rows,parity=parity,source_sha256=file_sha256(__file__)))
        print('factor swaps',index,'elapsed',round(time.monotonic()-start,1),flush=True)
    records=[read(dest/f'{m["index"]:04d}.json') for m in p['scene_samples']]
    n=len(records);rng=np.random.default_rng(928)
    w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000)])
    quantities={k:{} for k in ['miou','fnr','fpr','rescued_fraction']}
    for bits in itertools.product([0,1],repeat=4):
        selected=[next(r for r in f['conditions'] if tuple(r['bits'])==bits) for f in records]
        a=np.asarray([r['counts'] for r in selected],dtype=float)
        c=(w@a.reshape(n,-1)).reshape(-1,18,18);m=metrics(c)
        for key,metric in [('miou','mIoU'),('fnr','occupied_FNR'),('fpr','empty_FPR')]:quantities[key][bits]=m[metric]
        den=w@np.array([r['cohort'] for r in selected]);num=w@np.array([r['rescued'] for r in selected])
        quantities['rescued_fraction'][bits]=100*np.divide(num,den,out=np.zeros(len(w)),where=den>0)
    summary=dict(status='complete',frames=n,factor_combinations=16,
        interpretation='Exploratory cross-model endpoint output-factor accounting; not natural causal mediation or whole-history attribution',
        parity=[r['parity'] for r in records],effects={})
    for key,values in quantities.items():
        changes=shapley(values)
        summary['effects'][key]={factor:dict(value=float(v[0]),ci95=np.percentile(v[1:],[2.5,97.5]).tolist()) for factor,v in changes.items()}
        summary['effects'][key]['total']=dict(value=float((values[(1,1,1,1)]-values[(0,0,0,0)])[0]))
    write_json(dest/'summary.json',summary)


if __name__=='__main__':run()
