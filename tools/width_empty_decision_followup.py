"""Supplementary matched-width control, registered before new policy validation."""
from pathlib import Path
from datetime import datetime
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tools.empty_decision_followup import RUN,SEEDS,LAMBDAS,BUDGETS,initialize,read,write_json,file_sha256
from tools.fit_empty_decision_followup import Policy,fit_policy
from tools.eval_empty_decision_optimization import EDGES,hist_tensor,histogram_curve_counts,histogram_counts_torch,metrics,region_metrics,chosen_index
DEST=RUN/'width_control'


def fit_select():
    DEST.mkdir(exist_ok=True)
    if (DEST/'selection.json').exists():return
    assert not (RUN/'validation/summary.json').exists(),'Register control before new policy validation'
    if not (DEST/'protocol.json').exists():
        write_json(DEST/'protocol.json',dict(at=datetime.now().astimezone().isoformat(),role='supplementary matched-decoder-width control',
            reason='Main H40 vs Mix16 matches total parameters; this control matches decoder depth and width',
            main_selection_sha256=file_sha256(RUN/'operating_points.json'),hidden=16,decision_parameters=705,
            probability_inputs=False,seeds=SEEDS,lambda_grid=LAMBDAS,budgets=BUDGETS,
            new_policy_validation_not_started=True))
    for seed in SEEDS:
        for lam in LAMBDAS:fit_policy('H16',seed,lam)
    paths={p.stem:p for p in (DEST/'models').glob('*.pt')};models={k:Policy(p) for k,p in paths.items()}
    anchor=torch.as_tensor(initialize()['anchor_thresholds'],device='cuda')
    shape=(2,18,17,2,len(EDGES)+1)
    hist={k:torch.zeros(shape,device='cuda',dtype=torch.int64) for k in models}
    frames=[f for f in read(RUN/'train_cache.json')['frames'] if f['full_candidates']]
    with torch.inference_mode():
        for n,meta in enumerate(frames):
            with np.load(ROOT/meta['path']) as z:
                x=torch.as_tensor(z['X'],device='cuda');fg=torch.as_tensor(z['fg'].astype(np.int64),device='cuda')
                gt=torch.as_tensor(z['gt'].astype(np.int64),device='cuda');r=torch.as_tensor(z['region'],device='cuda')
            for name,model in models.items():hist[name]+=hist_tensor(model(x,fg),gt,fg,r,x[:,0]>=0)
            if (n+1)%256==0:print('width tuning',n+1,flush=True)
    with np.load(RUN/'tuning/histograms.npz') as z:
        fixed,native=z['fixed'],z['native'];base=z['anchor']
    zero=int(np.searchsorted(EDGES,0))+1;allowed=np.where((EDGES>=-1)&(EDGES<=1))[0]+1
    from tools.eval_empty_decision_optimization import histogram_counts
    b2=histogram_counts(base,fixed,zero)[0];reference=region_metrics(native)['camera_complement']['empty_FPR']/100
    options={seed:[dict(source='anchor',threshold=0.,cuts=zero,counts=b2)] for seed in SEEDS}
    for name,h in hist.items():
        record=read(paths[name].with_suffix('.json'));curve=histogram_curve_counts(h.cpu().numpy(),fixed,allowed)
        options[record['seed']]+=[dict(source=name,threshold=float(EDGES[c-1]),cuts=int(c),counts=cc) for c,cc in zip(allowed,curve)]
    points=[]
    for seed,choices in options.items():
        cc=np.stack([p['counts'] for p in choices]);cam=metrics(cc[:,0])['mIoU'];risk=metrics(cc[:,1])['empty_FPR']/100
        for budget in BUDGETS:
            i,feasible=chosen_index(cam,risk,budget,reference);tag='unconstrained' if budget is None else f'eps{budget:g}'
            points.append(dict(id=f'H16_{tag}_seed{seed}',seed=seed,budget=budget,**{k:v for k,v in choices[i].items() if k!='counts'},
                tuning=region_metrics(cc[i]),feasible=feasible))
    write_json(DEST/'selection.json',dict(status='frozen_before_validation',at=datetime.now().astimezone().isoformat(),
        points=points,models={k:dict(path=str(p.relative_to(ROOT)),sha256=file_sha256(p)) for k,p in paths.items()},
        main_selection_sha256=file_sha256(RUN/'operating_points.json'),source_sha256=file_sha256(__file__)))


def validate():
    selection=read(DEST/'selection.json');sha=file_sha256(DEST/'selection.json')
    assert selection['main_selection_sha256']==file_sha256(RUN/'operating_points.json')
    for name,r in selection['models'].items():assert file_sha256(ROOT/r['path'])==r['sha256']
    models={k:Policy(ROOT/v['path']) for k,v in selection['models'].items()};points=selection['points']
    anchor=torch.as_tensor(initialize()['anchor_thresholds'],device='cuda');counts=[]
    with torch.inference_mode():
        for n,meta in enumerate(read(RUN/'val_cache.json')['frames']):
            with np.load(ROOT/meta['path']) as z:
                x=torch.as_tensor(z['X'],device='cuda');fg=torch.as_tensor(z['fg'].astype(np.int64),device='cuda')
                gt=torch.as_tensor(z['gt'].astype(np.int64),device='cuda');r=torch.as_tensor(z['region'],device='cuda')
                fixed=torch.as_tensor(z['fixed_confusion'],device='cuda')
            values=[];cache={}
            for p in points:
                source=p['source']
                if source not in cache:
                    score=x[:,0]-anchor[fg] if source=='anchor' else models[source](x,fg)
                    cache[source]=hist_tensor(score,gt,fg,r,x[:,0]>=0)
                c,_=histogram_counts_torch(cache[source],fixed,[p['cuts']]);values.append(c[0].cpu().numpy())
            counts.append(np.stack(values))
            if (n+1)%256==0:print('width validation',n+1,flush=True)
    cc=np.stack(counts);np.savez_compressed(DEST/'per_frame_counts.npz',confusion=cc)
    result=dict(status='complete',selection_sha256=sha,points={p['id']:{**p,'validation':region_metrics(cc[:,i].sum(0))} for i,p in enumerate(points)})
    write_json(DEST/'validation.json',result)


def summarize():
    validation=read(RUN/'validation/summary.json');width=read(DEST/'validation.json')
    with np.load(RUN/'validation/per_frame_counts.npz') as z:c=z['confusion']
    with np.load(DEST/'per_frame_counts.npz') as z:h=z['confusion']
    primary=[p for p in width['points'].values() if p['budget']==.005];hnames=list(width['points']);names=list(validation['operating_points'])
    a=next(p for p in primary if p['seed']==928)
    scene=np.asarray([f['scene'] for f in validation['frames']]);unique,inverse=np.unique(scene,return_inverse=True)
    rng=np.random.default_rng(928);mult=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),500)
    result=dict(primary={p['seed']:p for p in primary},comparisons={},points=len(width['points']),parameters=705)
    for family in ['H','Mix','Conditional']:
        b=next(p for p in validation['operating_points'].values() if p['family']==family and p['budget']==.005 and p['seed']==928)
        pair=np.stack([c[:,names.index(b['id'])],h[:,hnames.index(a['id'])]],1)
        grouped=np.zeros((len(unique),2,2,18,18),np.float64);np.add.at(grouped,inverse,pair)
        with threadpool_limits(8):summed=(mult@grouped.reshape(len(unique),-1)).reshape(500,2,2,18,18)
        camera=metrics(summed[:,:,0])['mIoU'];risk=metrics(summed[:,:,1])['empty_FPR']
        result['comparisons'][family+'__minus__H16']=dict(camera_delta=b['validation']['camera']['mIoU']-a['validation']['camera']['mIoU'],
            risk_delta=b['validation']['camera_complement']['empty_FPR']-a['validation']['camera_complement']['empty_FPR'],
            camera_ci95=np.quantile(camera[:,0]-camera[:,1],[.025,.975]).tolist(),risk_ci95=np.quantile(risk[:,0]-risk[:,1],[.025,.975]).tolist())
    write_json(DEST/'results.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','validate','summarize']);a=p.parse_args()
    {'fit':fit_select,'validate':validate,'summarize':summarize}[a.stage]()
