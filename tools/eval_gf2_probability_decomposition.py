"""Cache full mini validation and fit every decision rule on training only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.gf2_probability_experiment import RUN,CONFIG,initialize,read,write,sha
from tools.gf2_probability_decisions import temperature,temperature_preserving_winner,risk_score,confusion,event_counts,TEMPERATURES,RISK_RATIOS
from tools.calibrate_empty_mechanism_stage2 import EDGES,NB,counts_from_hist,fitting_thresholds
from tools.eval_empty_decision_optimization import metrics

MODELS=['baseline','probability']
RULES=['joint','hierarchical','joint_class','hierarchical_class','risk_class','joint_global']


def checkpoint(name):
    return ROOT/initialize()['baseline'] if name=='baseline' else RUN/'train/probability/epoch_5.pth'


def build(name):
    import model
    from mmengine import Config
    from mmseg.models import build_segmentor
    from tools.empty_decision_optimization import seed_all
    seed_all(928)
    config=Config.fromfile('config/center_support_contribution1/a0_original.py' if name=='baseline' else CONFIG)
    net=build_segmentor(config.model).cuda().eval().requires_grad_(False)
    import torch
    state=torch.load(checkpoint(name),map_location='cpu',weights_only=False)
    assert state['epoch']==5 and state['global_iter']==20480
    net.load_state_dict(state['state_dict'],strict=True)
    return net


def extract_fields(name,out,batch):
    import torch
    labels=out['sampled_label'][0].long()
    mask=batch['occ_cam_mask'][0].bool().flatten()&(labels>=0)&(labels<18)
    z=out['pred_occ'][-1][0,:,mask]
    native=z.argmax(0)
    if name=='baseline':
        q=z[:17].double().softmax(0).T
        logit=torch.logsumexp(z[:17].double(),0)-z[17].double()
        o=logit.sigmoid()
        margin=z[:17].max(0).values.double()-z[17].double()
    else:
        q=out['conditional_semantics'][-1][0,:,mask].T
        o=out['bin_logits'][-1][0,mask]
        with torch.no_grad():
            margin=z[:17].max(0).values.double().log()-z[17].double().log()
        assert float((q.sum(-1)-1).abs().max())<2e-5
        assert float((z.sum(0)-1).abs().max())<2e-5
    assert torch.isfinite(q).all() and torch.isfinite(o).all()
    assert (o>=0).all() and (o<=1).all()
    return dict(y=labels[mask].cpu().numpy().astype(np.uint8),q=q.cpu().numpy(),
        o=o.cpu().numpy(),native=native.cpu().numpy().astype(np.uint8),margin=margin.cpu().numpy())


def scores(frame,ratio):
    # At the joint endpoint retain the original forward's exact tie behavior.
    if ratio==1:return frame['margin']
    return risk_score(frame['o'],frame['q'].max(-1),ratio)


def hist_add(hist,frame,ratio):
    y=frame['y'].astype(np.int64);fg=frame['q'].argmax(-1)
    bucket=np.searchsorted(EDGES,scores(frame,ratio),side='right')
    hist+=np.bincount((y*17+fg)*NB+bucket,minlength=18*17*NB).reshape(18,17,NB)


def tune(hist,thresholds):
    best=None;candidates=[]
    for shift in np.arange(-32,33)/32:
        cuts=np.clip(thresholds-shift,EDGES[0],EDGES[-1])
        cm=counts_from_hist(hist,cuts)
        m=metrics(cm[None]);row=dict(shift=float(shift),miou=float(m['mIoU'][0]),fpr=float(m['empty_FPR'][0]))
        candidates.append(row)
        key=(-row['miou'],row['fpr'],abs(shift),shift)
        if best is None or key<best[0]:best=(key,row,cuts)
    return best[2],best[1],candidates


def fit_global(hist):
    tail=np.cumsum(hist[...,::-1],axis=-1,dtype=np.int64)[...,::-1][...,1:]
    truth=hist.sum((1,2));tp=tail[np.arange(17),np.arange(17)]
    fp=tail.sum(0)-tp
    miou=(tp/np.maximum(truth[:17,None]+fp,1)).mean(0)*100
    fpr=tail[17].sum(0)/max(truth[17],1)*100
    choices=np.flatnonzero(miou==miou.max())
    index=min(choices,key=lambda i:(fpr[i],abs(EDGES[i]),EDGES[i]))
    return float(EDGES[index]),dict(miou=float(miou[index]),fpr=float(fpr[index]),
        boundary=bool(index in [0,len(EDGES)-1]))


def extract(name):
    import torch
    from tools.empty_decision_optimization import build_loader,to_cuda
    from tools.empty_mechanism_stage1 import network_output
    p=initialize();dest=RUN/'evaluation'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'extraction_complete.json').exists():return
    cksha=sha(checkpoint(name));record=dest/'checkpoint.json'
    if record.exists():assert read(record)['sha256']==cksha
    else:write(record,dict(path=str(checkpoint(name).relative_to(ROOT)),sha256=cksha))
    net=build(name);start=time.monotonic()
    histpath=dest/'calibration_hist.npz'
    if not histpath.exists():
        dataset,_,_=build_loader('train');chosen={}
        for i,(scene,_) in enumerate(dataset.keyframes):chosen.setdefault(scene,i)
        assert set(chosen)==set(p['fit_scenes'])|set(p['tune_scenes'])
        indices=[chosen[s] for s in sorted(chosen)]
        _,loader,_=build_loader('train',indices)
        hist=np.zeros((len(RISK_RATIOS),2,18,17,NB),dtype=np.int64)
        fitting=set(p['fit_scenes'])
        with torch.inference_mode():
            for n,(index,batch) in enumerate(zip(indices,loader)):
                batch=to_cuda(batch);out=network_output(net,batch);a=extract_fields(name,out,batch)
                group=0 if dataset.keyframes[index][0] in fitting else 1
                for j,ratio in enumerate(RISK_RATIOS):hist_add(hist[j,group],a,ratio)
                del out,batch,a
                if (n+1)%128==0:print(name,'train calibration',n+1,len(indices),flush=True)
        np.savez_compressed(histpath,hist=hist,indices=indices,checkpoint_sha256=np.array(cksha))
    stored=np.load(histpath);assert str(stored['checkpoint_sha256'])==cksha
    hist=stored['hist'];selections=[]
    for j,ratio in enumerate(RISK_RATIOS):
        initial,boundary=fitting_thresholds(hist[j,0])
        if any(boundary):raise RuntimeError('Calibration optimum at registered search boundary')
        cuts,choice,candidates=tune(hist[j,1],initial)
        selections.append(dict(ratio=ratio,thresholds=cuts.tolist(),boundary=boundary,
            selection=choice,candidates=candidates))
    selected=min(selections,key=lambda x:(-x['selection']['miou'],x['selection']['fpr'],abs(x['ratio']-1)))
    # Fit the1-parameter global shift on the fitting cohort, then evaluate its
    # training-tune score without another parameter search.
    j=RISK_RATIOS.index(1.)
    global_cut,global_choice=fit_global(hist[j,0])
    if global_choice['boundary']:raise RuntimeError('Global calibration reached search boundary')
    write(dest/'selection.json',dict(by_ratio=selections,risk=selected,
        joint=selections[j],hierarchical=selections[0],global_threshold=global_cut,
        global_selection=global_choice,fit_frames=523,tune_frames=175))
    frames=dest/'frames';frames.mkdir(exist_ok=True)
    missing=[i for i in range(1024) if not (frames/f'{i:04d}.npz').exists()]
    dataset,loader,indices=build_loader('val',missing)
    with torch.inference_mode():
        for n,(index,batch) in enumerate(zip(indices,loader)):
            batch=to_cuda(batch);out=network_output(net,batch);a=extract_fields(name,out,batch)
            fg=a['q'].argmax(-1);pred=np.where(a['margin']>=0,fg,17)
            np.testing.assert_array_equal(pred,a['native'])
            np.savez(frames/f'{index:04d}.npz',**a)
            del out,batch,a
            if (n+1)%128==0:print(name,'validation',n+1,len(missing),'elapsed',round(time.monotonic()-start,1),flush=True)
    meta=[dict(index=i,scene=dataset.keyframes[i][0],scene_frame=int(dataset.keyframes[i][1])) for i in range(1024)]
    write(dest/'validation_frames.json',meta)
    write(dest/'extraction_complete.json',dict(status='complete',validation_frames=1024,
        calibration_frames=698,checkpoint_sha256=cksha,source_sha256=sha(__file__),elapsed=time.monotonic()-start))


def rule_parameters(selection):
    return dict(joint=(1.,np.zeros(17)),hierarchical=(0.,np.zeros(17)),
        joint_class=(1.,np.array(selection['joint']['thresholds'])),
        hierarchical_class=(0.,np.array(selection['hierarchical']['thresholds'])),
        risk_class=(selection['risk']['ratio'],np.array(selection['risk']['thresholds'])),
        joint_global=(1.,np.full(17,selection['global_threshold'])))


def statistics(name):
    dest=RUN/'evaluation'/name
    if (dest/'statistics_complete.json').exists():return
    assert (dest/'extraction_complete.json').exists()
    meta=read(dest/'validation_frames.json');scenes=sorted({a['scene'] for a in meta});lookup={s:i for i,s in enumerate(scenes)}
    params=rule_parameters(read(dest/'selection.json'))
    curve_rules=['joint','hierarchical','joint_class','hierarchical_class','risk_class']
    arrays={r:{k:np.zeros((len(scenes),NB),dtype=np.int64) for k in ['empty_hist','rescue_hist','baseline_correct_hist']} for r in curve_rules}
    basefp=np.zeros(len(scenes),np.int64);cohort_size=np.zeros(len(scenes),np.int64)
    rows=[]
    for i in range(1024):
        a=np.load(dest/'frames'/f'{i:04d}.npz');y=a['y'];q=a['q'];fg=q.argmax(-1);o=a['o'];native=a['native']
        cohort=(y<17)&(fg==y)&(native==17);correct=(y<17)&(native==y);scene=lookup[meta[i]['scene']]
        basefp[scene]+=int(((y==17)&(native!=17)).sum());cohort_size[scene]+=int(cohort.sum())
        gap=(scores(a,0)>=0)&(native==17)
        gap_counts=dict(gap=int(gap.sum()),gap_true_foreground=int((gap&(y<17)).sum()),
            gap_true_empty=int((gap&(y==17)).sum()),gap_correct_winner=int((gap&(y<17)&(fg==y)).sum()),
            gap_wrong_winner=int((gap&(y<17)&(fg!=y)).sum()))
        row=dict(index=i,foreground=int((y<17).sum()),empty=int((y==17).sum()),
            native_missed=int(((y<17)&(native==17)).sum()),cohort=int(cohort.sum()),
            rank_correct=int(((y<17)&(fg==y)).sum()),saturated_zero=int((o==0).sum()),
            saturated_one=int((o==1).sum()),gap=gap_counts,rules={},temperatures={})
        for rule,(ratio,cuts) in params.items():
            score=scores(a,ratio)-cuts[fg];pred=np.where(score>=0,fg,17)
            row['rules'][rule]=dict(counts=confusion(y,pred).tolist(),rescued=int((cohort&(pred==y)).sum()),
                newly_empty=int((correct&(pred==17)).sum()),newly_wrong=int((correct&(pred!=17)&(pred!=y)).sum()))
            if rule in arrays:
                bucket=np.searchsorted(EDGES,score,side='right')
                for key,mask in [('empty_hist',y==17),('rescue_hist',cohort),('baseline_correct_hist',correct)]:
                    arrays[rule][key][scene]+=np.bincount(bucket[mask],minlength=NB)
        hierarchical_native=np.where(scores(a,0)>=0,fg,17)
        for t in TEMPERATURES:
            qt,temperature_fg,numerical_ties=temperature_preserving_winner(q,t)
            np.testing.assert_array_equal(temperature_fg,fg)
            # Fix occupied/empty evidence; change only the conditional class
            # share. T1 has exactly zero correction to the original margin.
            with np.errstate(divide='ignore'):
                delta=np.log(qt.max(-1))-np.log(q.astype(np.float64).max(-1))
            margin=a['margin']+delta
            pred=np.where(margin>=0,fg,17)
            if t==1:np.testing.assert_array_equal(pred,native)
            hierarchical_temperature=np.where(scores(a,0)>=0,temperature_fg,17)
            np.testing.assert_array_equal(hierarchical_temperature,hierarchical_native)
            row['temperatures'][str(t)]=dict(counts=confusion(y,pred).tolist(),
                correct_to_empty=int((correct&(pred==17)).sum()),
                missed_to_correct=int(((y<17)&(native==17)&(pred==y)).sum()),
                empty_to_foreground=int(((y==17)&(native==17)&(pred!=17)).sum()),
                foreground_to_empty=int(((native!=17)&(pred==17)).sum()),
                rank_changes=int((temperature_fg!=fg).sum()),empty_probability_changes=0,
                hierarchical_changes=int((hierarchical_temperature!=hierarchical_native).sum()),
                numerical_argmax_ties=numerical_ties)
        rows.append(row)
        if (i+1)%256==0:print(name,'statistics',i+1,flush=True)
    for rule,values in arrays.items():
        np.savez_compressed(dest/f'curves_{rule}.npz',**values,baseline_fp=basefp,
            cohort_size=cohort_size,scenes=np.asarray(scenes))
    write(dest/'statistics.json',dict(status='complete',records=rows,frames=1024,scenes=scenes))
    write(dest/'statistics_complete.json',dict(status='complete',frames=1024))


def profile(name):
    import torch
    from tools.empty_decision_optimization import build_loader,to_cuda
    from tools.empty_mechanism_stage1 import network_output
    net=build(name);_,loader,_=build_loader('val',[0]);batch=to_cuda(next(iter(loader)))
    samples=[];post=[]
    with torch.inference_mode():
        for i in range(40):
            torch.cuda.synchronize();start=time.perf_counter();out=network_output(net,batch);torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if i>=10:samples.append(elapsed)
            if name=='probability':
                q=out['conditional_semantics'][-1];o=out['bin_logits'][-1]
                torch.cuda.synchronize();start=time.perf_counter()
                maximum,fg=q.max(1)
                # The additional risk-family gate only; learned class
                # thresholds do not alter this operation's tensor shapes.
                pred=torch.where((1-o)+.5*o*(1-maximum)<=o,fg,17)
                torch.cuda.synchronize()
                if i>=10:post.append(time.perf_counter()-start)
            del out
    write(RUN/'evaluation'/name/'timing.json',dict(device=torch.cuda.get_device_name(),
        batch_size=1,amp=False,warmup=10,repetitions=30,input='preloaded fixed256x704 camera tensors; no image loading/preprocessing',
        mean_seconds=float(np.mean(samples)),fps=float(1/np.mean(samples)),
        additional_risk_gate_seconds=float(np.mean(post)) if post else None))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['extract','statistics','profile'])
    p.add_argument('--model',choices=MODELS,default='probability');a=p.parse_args()
    globals()[a.stage](a.model)


if __name__=='__main__':main()
