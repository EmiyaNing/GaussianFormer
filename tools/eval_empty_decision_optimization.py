"""Full-frame training-only operating-point selection and independent evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.empty_decision_optimization import (RUN, REPORT, BAND, SEEDS, BUDGETS, PRIMARY_BUDGET,
    BINS, SCORE_MIN, SCORE_MAX, initialize, read, seed_all, build_loader, build_network,
    to_cuda, support_field, candidate_features, confusion, write_json, file_sha256)
from tools.fit_empty_decision_optimization import Predictor

NB = BINS+2
SHAPE = (2,18,17,2,NB)
EDGES = np.linspace(SCORE_MIN,SCORE_MAX,BINS+1)
THRESHOLDS = EDGES.copy()


def hist_tensor(score,gt,fg,region,native):
    edges = torch.as_tensor(EDGES,device=score.device,dtype=score.dtype)
    bins = torch.bucketize(score.contiguous(),edges,right=True)
    code = ((((region.long()*18+gt.long())*17+fg.long())*2+native.long())*NB+bins)
    return torch.bincount(code,minlength=int(np.prod(SHAPE))).reshape(SHAPE)


def histogram_counts(hist,fixed,cuts):
    """GT x predicted counts and six directional transition counts, exact at edges."""
    h = np.asarray(hist)
    reverse = np.flip(np.cumsum(np.flip(h,-1),axis=-1),-1)
    cut = np.broadcast_to(np.asarray(cuts,dtype=np.int64),(17,))
    chosen = np.take_along_axis(reverse,cut[None,None,:,None,None],axis=-1)[...,0]
    total = h.sum(-1)
    counts = np.array(fixed,dtype=np.int64,copy=True)
    counts[:,:,:17] += chosen.sum(-1)
    counts[:,:,17] += (total-chosen).sum((2,3))
    events = []
    for mass in [chosen[:,:,:,0],total[:,:,:,1]-chosen[:,:,:,1]]:
        correct = np.einsum('rii->r',mass[:,:17,:])
        other = mass[:,:17,:].sum((1,2))-correct
        empty = mass[:,17,:].sum(-1)
        events.extend([correct,other,empty])
    return counts,np.stack(events,-1)


def histogram_counts_torch(hist,fixed,cuts):
    reverse = hist.flip(-1).cumsum(-1).flip(-1)
    total = hist.sum(-1)
    outputs,events = [],[]
    for cut in cuts:
        c = torch.as_tensor(np.broadcast_to(np.asarray(cut,dtype=np.int64),(17,)).copy(),device=hist.device)
        gather = c[None,None,:,None,None].expand(2,18,17,2,1)
        chosen = reverse.gather(-1,gather)[...,0]
        cc = fixed.clone()
        cc[:,:,:17] += chosen.sum(-1)
        cc[:,:,17] += (total-chosen).sum((2,3))
        ev = []
        for mass in [chosen[:,:,:,0],total[:,:,:,1]-chosen[:,:,:,1]]:
            corr = mass[:,:17,:].diagonal(dim1=1,dim2=2).sum(-1)
            ev.extend([corr,mass[:,:17,:].sum((1,2))-corr,mass[:,17,:].sum(-1)])
        outputs.append(cc)
        events.append(torch.stack(ev,-1))
    return torch.stack(outputs),torch.stack(events)


def histogram_curve_counts(hist,fixed,cuts):
    """All scalar thresholds with one cumulative sum, avoiding repeated scans."""
    reverse = np.flip(np.cumsum(np.flip(hist,-1),axis=-1),-1)
    occupied = reverse[...,cuts].sum(-2).transpose(3,0,1,2)
    result = np.broadcast_to(fixed,(len(cuts),2,18,18)).copy()
    result[:,:,:,:17] += occupied
    result[:,:,:,17] += hist.sum((-1,-2,-3))[None]-occupied.sum(-1)
    return result


def metrics(cc):
    c = np.asarray(cc,dtype=np.float64)
    diag = np.diagonal(c,axis1=-2,axis2=-1)
    union = c.sum(-2)+c.sum(-1)-diag
    iou = np.divide(diag,union,out=np.zeros_like(diag),where=union>0)
    tp = c[...,:17,:17].sum((-2,-1))
    fp = c[...,17,:17].sum(-1)
    fn = c[...,:17,17].sum(-1)
    empty = c[...,17,:].sum(-1)
    return dict(mIoU=iou[...,:17].mean(-1)*100,
        IoU=np.divide(tp,tp+fp+fn,out=np.zeros_like(tp),where=(tp+fp+fn)>0)*100,
        empty_FPR=np.divide(fp,empty,out=np.zeros_like(fp),where=empty>0)*100,
        per_class_IoU=iou[...,:17]*100,
        occupied_FNR=np.divide(fn,c[...,:17,:].sum((-2,-1)),out=np.zeros_like(fn),where=c[...,:17,:].sum((-2,-1))>0)*100)


def region_metrics(cc):
    c = np.concatenate([cc,cc.sum(0,keepdims=True)],axis=0)
    results = {}
    for i,name in enumerate(['camera','camera_complement','all_valid']):
        results[name] = {k:v.tolist() if np.ndim(v) else float(v) for k,v in metrics(c[i]).items()}
    return results


def chosen_index(camera,risk,budget,reference):
    feasible = np.ones(len(camera),bool) if budget is None else risk <= reference+budget+1e-12
    choices = np.where(feasible)[0]
    if not len(choices):
        return int(np.argmin(risk)),False
    # Prefer smaller risk among exactly equal camera scores.
    i = min(choices,key=lambda j:(-camera[j],risk[j],j))
    return int(i),True


def model_manifest():
    fit = read(RUN/'fit_results.json')
    result = {k:v for k,v in fit['models'].items() if k.startswith(('B3_','P1_','P2_','P3_','P4_'))}
    return result


def score_for(name,predictor,x,fg,cache):
    raw = predictor(x,fg,cache)
    return torch.logit(raw[:,0].clamp(1e-6,1-1e-6)) if name.startswith('P1_') else raw


def run_tuning(limit=0):
    protocol = initialize()
    dest = RUN/('tuning_smoke' if limit else 'tuning')
    if not limit and (dest/'summary.json').exists() and (dest/'histograms.npz').exists():
        cached = read(dest/'summary.json')
        assert cached['status'] == 'complete' and cached['models'] == model_manifest()
        if (RUN/'operating_points.json').exists():
            frozen = read(RUN/'operating_points.json')
            assert frozen['histogram_sha256'] == file_sha256(dest/'histograms.npz')
            for name,meta in cached['models'].items():
                assert file_sha256(ROOT/meta['path']) == frozen['model_sha256'][name]
        print('Reusing completed full tuning census',flush=True)
        return cached
    seed_all(928)
    dataset,_,_ = build_loader('train')
    indices = [i for i,(scene,_) in enumerate(dataset.keyframes) if scene in protocol['tune_scenes']]
    if limit:
        indices = indices[:limit]
    dataset,loader,indices = build_loader('train',indices)
    net = build_network()
    manifest = model_manifest()
    predictors = {k:Predictor(ROOT/v['path']) for k,v in manifest.items()}
    names = ['margin']+list(predictors)
    hists = {name:torch.zeros(SHAPE,device='cuda',dtype=torch.int64) for name in names}
    fixed = np.zeros((2,18,18),np.int64)
    native_total = np.zeros_like(fixed)
    frames,started = [],time.monotonic()
    with torch.inference_mode():
        for n,(idx,batch) in enumerate(zip(indices,loader)):
            batch = to_cuda(batch)
            out = net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k != 'img'})
            logits = out['pred_occ'][-1][0]
            fgmax,fg = logits[:17].max(0)
            margin = fgmax-logits[17]
            gt = out['sampled_label'][0].long()
            region = (~batch['occ_cam_mask'][0].reshape(-1).bool()).long()
            native = logits.argmax(0)
            valid = (gt >= 0)&(gt < 18)
            ids = torch.where(valid&(margin.abs() <= BAND))[0]
            support,_ = support_field(net.head,out,check=False)
            x = candidate_features(out,support,ids)
            fi,gi,ri,ni = fg[ids],gt[ids],region[ids],native[ids] != 17
            cache = {}
            for name in names:
                score = x[:,0] if name == 'margin' else score_for(name,predictors[name],x,fi,cache)
                hists[name] += hist_tensor(score,gi,fi,ri,ni)
            outside = valid.clone()
            outside[ids] = False
            fixed += confusion(gt[outside],native[outside],region[outside])
            native_total += confusion(gt,native,region)
            frames.append(dict(index=idx,scene=dataset.keyframes[idx][0]))
            if n == 0 or (n+1)%128 == 0:
                print('full tuning',n+1,'/',len(indices),'elapsed',time.monotonic()-started,flush=True)
    dest = RUN/('tuning_smoke' if limit else 'tuning')
    dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'histograms.npz',**{k:v.cpu().numpy() for k,v in hists.items()},fixed=fixed,native=native_total)
    base,_ = histogram_counts(hists['margin'].cpu().numpy(),fixed,int(np.searchsorted(EDGES,0))+1)
    assert np.array_equal(base,native_total),'Native zero-threshold histogram must be exact'
    result = dict(status='complete',frames=frames,samples=len(frames),models=manifest,
        native_parity=True,elapsed_seconds=time.monotonic()-started,source_sha256=file_sha256(__file__))
    write_json(dest/'summary.json',result)
    return result


def select_points():
    summary = read(RUN/'tuning/summary.json')
    assert summary['status'] == 'complete'
    with np.load(RUN/'tuning/histograms.npz') as z:
        native = z['native']
        fixed = z['fixed']
        histograms = {k:z[k] for k in ['margin']+list(summary['models'])}
    ref = region_metrics(native)['camera_complement']['empty_FPR']/100
    records = []
    zero_cut = int(np.searchsorted(EDGES,0))+1
    records.append(dict(id='B0',family='B0',seed=None,budget=None,source='margin',cuts=zero_cut,
        threshold=0.,tuning=region_metrics(native),feasible=True))
    for name,h in histograms.items():
        if name == 'margin':
            family,seed = 'B1',None
        else:
            family,seed = name.rsplit('_seed',1)
            seed = int(seed)
        cuts = np.arange(1,BINS+2)
        if name == 'margin':
            cuts = cuts[(THRESHOLDS >= -BAND)&(THRESHOLDS <= BAND)]
        ccs = histogram_curve_counts(h,fixed,cuts)
        ms = metrics(ccs[:,0])['mIoU']
        risks = metrics(ccs[:,1])['empty_FPR']/100
        for budget in (BUDGETS if seed in [None,928] else [PRIMARY_BUDGET]):
            i,feasible = chosen_index(ms,risks,budget,ref)
            tag = 'unconstrained' if budget is None else f'eps{budget:g}'
            records.append(dict(id=f'{family}_{tag}'+('' if seed is None else f'_seed{seed}'),family=family,
                seed=seed,budget=budget,source=name,cuts=int(cuts[i]),threshold=float(EDGES[cuts[i]-1]),
                feasible=feasible,tuning=region_metrics(ccs[i])))
        if family == 'P1' and seed == 928:
            cc,_ = histogram_counts(h,fixed,zero_cut)
            records.append(dict(id='P1_q05_seed928',family='P1_q05',seed=928,budget=None,source=name,
                cuts=zero_cut,threshold=0.,feasible=True,tuning=region_metrics(cc)))
    # Class-specific scalar margin thresholds, optimized with dense dual search.
    h = histograms['margin']
    allowed = np.where((EDGES >= -BAND)&(EDGES <= BAND))[0]+1
    above = np.flip(np.cumsum(np.flip(h,-1),axis=-1),-1).sum(-2)
    n_empty = native[1,17].sum()
    camera_gt = native[0].sum(-1)
    tp = np.stack([fixed[0].diagonal()[:17]+above[0,np.arange(17),np.arange(17),cut] for cut in allowed])
    fp = np.stack([fixed[0].sum(0)[:17]-fixed[0].diagonal()[:17]+above[0,:,:,cut].sum(0)-above[0,np.arange(17),np.arange(17),cut] for cut in allowed])
    ious = tp/(camera_gt[:17]+fp).clip(1)
    out_fp = np.stack([fixed[1,17,:17]+above[1,17,:,cut] for cut in allowed])
    proposals = [np.full(17,zero_cut,dtype=int)]
    for lam in np.r_[0.,np.logspace(-5,4,181)]:
        utility = ious/17-lam*out_fp/max(n_empty,1)
        proposals.append(allowed[utility.argmax(0)])
    unique = np.unique(np.stack(proposals),axis=0)
    ccs = np.stack([histogram_counts(h,fixed,c)[0] for c in unique])
    ms,risks = metrics(ccs[:,0])['mIoU'],metrics(ccs[:,1])['empty_FPR']/100
    for budget in BUDGETS:
        i,feasible = chosen_index(ms,risks,budget,ref)
        tag = 'unconstrained' if budget is None else f'eps{budget:g}'
        records.append(dict(id=f'B2_{tag}',family='B2',seed=None,budget=budget,source='margin',
            cuts=unique[i].tolist(),threshold=EDGES[unique[i]-1].tolist(),feasible=feasible,
            tuning=region_metrics(ccs[i])))
    result = dict(status='frozen_before_validation',created_at=datetime.now().astimezone().isoformat(),
        tuning_samples=summary['samples'],reference_out_fpr=ref,operating_points=records,
        histogram_sha256=file_sha256(RUN/'tuning/histograms.npz'),
        model_sha256={k:file_sha256(ROOT/v['path']) for k,v in summary['models'].items()},
        class_bias_proposals=len(unique),source_sha256=file_sha256(__file__),
        selection='full tuning scenes only; exact integer counts at predeclared score edges')
    write_json(RUN/'operating_points.json',result)
    print('Frozen',len(records),'operating points; class bias proposals',len(unique),flush=True)
    return result


def validation(limit=0):
    initialize()
    selection = read(RUN/'operating_points.json')
    selection_sha = file_sha256(RUN/'operating_points.json')
    if not limit and (RUN/'validation/summary.json').exists():
        cached = read(RUN/'validation/summary.json')
        assert cached['status'] == 'complete' and cached['samples'] == 1024
        assert cached['selection_sha256'] == selection_sha
        for name,meta in read(RUN/'tuning/summary.json')['models'].items():
            assert file_sha256(ROOT/meta['path']) == selection['model_sha256'][name]
        print('Reusing completed validation with frozen selection',flush=True)
        return cached
    summary = read(RUN/'tuning/summary.json')
    seed_all(928)
    dataset,loader,indices = build_loader('val',range(limit) if limit else None)
    assert not set(s for s,_ in dataset.keyframes)&set(initialize()['fit_scenes']+initialize()['tune_scenes'])
    manifest = summary['models']
    for k,v in manifest.items():
        assert file_sha256(ROOT/v['path']) == selection['model_sha256'][k]
    predictors = {k:Predictor(ROOT/v['path']) for k,v in manifest.items()}
    net = build_network()
    points = selection['operating_points']
    grouped = {}
    for i,p in enumerate(points):
        grouped.setdefault(p['source'],[]).append((i,p))
    dest = RUN/('validation_smoke' if limit else 'validation')
    dest.mkdir(exist_ok=True)
    primary = [i for i,p in enumerate(points) if p['id'] == 'B0' or
        (p['budget'] == PRIMARY_BUDGET and p['seed'] in [None,928]) or p['family'] == 'P1_q05']
    frame_counts,frame_events,frames = [],[],[]
    calibration = {}
    started = time.monotonic()
    with torch.inference_mode():
        for n,(idx,batch) in enumerate(zip(indices,loader)):
            stem = dest/f'frame_{idx:04d}'
            if stem.with_suffix('.npz').exists() and stem.with_suffix('.json').exists():
                meta = read(stem.with_suffix('.json'))
                assert meta['selection_sha256'] == selection_sha
                with np.load(stem.with_suffix('.npz')) as z:
                    cc,ev = z['confusion'],z['events']
                for k,v in meta['calibration'].items():
                    if k not in calibration:
                        calibration[k] = np.zeros_like(v,dtype=float)
                    calibration[k] += v
            else:
                batch = to_cuda(batch)
                out = net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k != 'img'})
                logits = out['pred_occ'][-1][0]
                fgmax,fg = logits[:17].max(0)
                margin = fgmax-logits[17]
                native = logits.argmax(0)
                gt = out['sampled_label'][0].long()
                region = (~batch['occ_cam_mask'][0].reshape(-1).bool()).long()
                valid = (gt >= 0)&(gt < 18)
                ids = torch.where(valid&(margin.abs() <= BAND))[0]
                support,audit = support_field(net.head,out,check=n < 2)
                x = candidate_features(out,support,ids)
                fi,gi,ri,ni = fg[ids],gt[ids],region[ids],native[ids] != 17
                outside = valid.clone()
                outside[ids] = False
                fixed = torch.as_tensor(confusion(gt[outside],native[outside],region[outside]),device='cuda')
                cc = np.empty((len(points),2,18,18),np.int64)
                ev = np.empty((len(points),2,6),np.int64)
                cache,scores,local_cal = {},{},{}
                for name,items in grouped.items():
                    score = x[:,0] if name == 'margin' else score_for(name,predictors[name],x,fi,cache)
                    scores[name] = score
                    h = hist_tensor(score,gi,fi,ri,ni)
                    c,e = histogram_counts_torch(h,fixed,[p['cuts'] for _,p in items])
                    rows = [i for i,_ in items]
                    cc[rows],ev[rows] = c.cpu().numpy(),e.cpu().numpy()
                native_cc = confusion(gt,native,region)
                assert np.array_equal(cc[0],native_cc),'B0 must equal native predictions'
                rng = np.random.default_rng(928+idx)
                sampled,weights = [],[]
                for r in [0,1]:
                    for sign in [False,True]:
                        members = torch.where((ri == r)&(ni == sign))[0].cpu().numpy()
                        count = len(members)
                        if count:
                            chosen = members if count <= 256 else rng.choice(members,256,replace=False)
                            sampled.append(chosen)
                            weights.append(np.full(len(chosen),count/len(chosen),np.float64))
                sample_ids = torch.as_tensor(np.concatenate(sampled),device='cuda')
                sampled_probabilities,probability_names = [],[]
                for support_flag in [True,False]:
                    for seed in SEEDS:
                        key = f'prob_{"support" if support_flag else "no_support"}_seed{seed}'
                        path = ROOT/('work_dirs/empty_decision_optimization/models/'+key+'.pt')
                        cache_key = str(path)
                        probabilities = cache.get(cache_key)
                        if probabilities is None:
                            continue
                        target = torch.where(gi == 17,1,torch.where(gi == fi,0,2)).long()
                        stats = calibration_statistics(probabilities,target,ri)
                        sampled_probabilities.append(probabilities[sample_ids,:2].cpu().numpy())
                        probability_names.append(key)
                        local_cal[key] = stats.tolist()
                        if key not in calibration:
                            calibration[key] = np.zeros_like(stats)
                        calibration[key] += stats
                predictions = {}
                for i in primary:
                    p = points[i]
                    score = scores[p['source']]
                    threshold = torch.as_tensor(p['threshold'],device='cuda',dtype=score.dtype)
                    if threshold.ndim:
                        threshold = threshold[fi]
                    pred = native.clone()
                    pred[ids] = torch.where(score >= threshold,fi,17)
                    assert np.array_equal(confusion(gt,pred,region),cc[i])
                    predictions[p['id']] = pred.cpu().numpy().astype(np.uint8)
                scene,sf = dataset.keyframes[idx]
                meta = dict(index=idx,scene=scene,scene_frame=int(sf),selection_sha256=selection_sha,
                    calibration=local_cal,numerical_audit=audit,primary_predictions=list(predictions),
                    probability_sample_models=probability_names)
                np.savez_compressed(stem.with_suffix('.npz'),confusion=cc,events=ev,
                    gt=gt.cpu().numpy().astype(np.uint8),**predictions,
                    probability_sample_qe=np.stack(sampled_probabilities,axis=1),
                    probability_sample_outcome=target[sample_ids].cpu().numpy().astype(np.int8),
                    probability_sample_region=ri[sample_ids].cpu().numpy().astype(np.int8),
                    probability_sample_weight=np.concatenate(weights),
                    probability_sample_voxel=ids[sample_ids].cpu().numpy().astype(np.int32))
                write_json(stem.with_suffix('.json'),meta)
            frame_counts.append(cc)
            frame_events.append(ev)
            frames.append(dict(index=idx,scene=dataset.keyframes[idx][0],path=str(stem.relative_to(ROOT))))
            if n == 0 or (n+1)%64 == 0:
                print('validation',n+1,'/',len(indices),'elapsed',time.monotonic()-started,flush=True)
                write_json(RUN/'status.json',dict(status='running',stage='validation',frames=n+1,expected=len(indices)))
    counts = np.stack(frame_counts)
    events = np.stack(frame_events)
    np.savez_compressed(dest/'per_frame_counts.npz',confusion=counts,events=events)
    results = {p['id']:dict(**p,validation=region_metrics(counts[:,i].sum(0)),
        transition_counts=events[:,i].sum(0).tolist()) for i,p in enumerate(points)}
    output = dict(status='complete',samples=len(frames),frames=frames,operating_points=results,
        primary_predictions=[points[i]['id'] for i in primary],selection_sha256=selection_sha,
        calibration={k:calibration_metrics(v) for k,v in calibration.items()},
        raw_calibration={k:v.tolist() for k,v in calibration.items()},
        native_zero_parity=True,elapsed_seconds=time.monotonic()-started,source_sha256=file_sha256(__file__))
    write_json(dest/'summary.json',output)
    assert file_sha256(RUN/'operating_points.json') == selection_sha
    return output


def calibration_statistics(p,target,region,bins=15):
    # per region: count,NLL,Brier; followed by q/e bin(count,probability sum,positive sum).
    result = np.zeros((2,3+bins*3*2),np.float64)
    with torch.no_grad():
        truth = torch.nn.functional.one_hot(target,3).to(p)
        nll = -p.gather(1,target[:,None])[:,0].clamp_min(1e-20).log()
        brier = (p-truth).square().sum(-1)
        for r in [0,1]:
            keep = region == r
            result[r,:3] = [int(keep.sum()),float(nll[keep].sum()),float(brier[keep].sum())]
            for event in [0,1]:
                index = (p[keep,event]*bins).long().clamp(max=bins-1)
                arrays = [torch.bincount(index,minlength=bins),
                    torch.bincount(index,weights=p[keep,event],minlength=bins),
                    torch.bincount(index,weights=truth[keep,event],minlength=bins)]
                result[r,3+event*bins*3:3+(event+1)*bins*3] = torch.stack(arrays,-1).cpu().numpy().reshape(-1)
    return result


def calibration_metrics(stats):
    result = {}
    for r,name in enumerate(['camera','camera_complement']):
        n = stats[r,0]
        row = dict(candidate_voxels=int(n),NLL=float(stats[r,1]/max(n,1)),Brier=float(stats[r,2]/max(n,1)))
        for event,label in enumerate(['q','e']):
            values = stats[r,3+event*45:3+(event+1)*45].reshape(15,3)
            row[label+'_ECE'] = float(np.abs(values[:,1]-values[:,2]).sum()/max(n,1))
        result[name] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['tune','select','validate'])
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    if args.stage == 'tune':
        run_tuning(args.limit)
    elif args.stage == 'select':
        select_points()
    else:
        validation(args.limit)


if __name__ == '__main__':
    main()
