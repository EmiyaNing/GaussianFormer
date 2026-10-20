"""Full validation and equal calibration for stage-three endpoint models."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage3 import RUN,initialize,read,write_json,file_sha256,seed_all,build_loader,to_cuda,network_output
from tools.empty_mechanism_stage1 import build_network
from tools.calibrate_empty_mechanism_stage2 import EDGES,NB,counts_from_hist,fitting_thresholds,tuning_thresholds,fields,counts
from tools.eval_empty_decision_optimization import metrics

MODELS=['B0','S','E','SE','original5','no_final_set5']


def evaluation_protocol():
    initialize();path=RUN/'evaluation_protocol.json'
    if path.exists():return read(path)
    prior=read(ROOT/'work_dirs/empty_decision_optimization/protocol.json')
    dataset,_,_=build_loader('train');chosen={}
    for index,(scene,frame) in enumerate(dataset.keyframes):chosen.setdefault(scene,index)
    indices=[chosen[s] for s in sorted(chosen)]
    fit=set(prior['fit_scenes']);tune=set(prior['tune_scenes'])
    assert set(chosen)==fit|tune and not fit&tune
    p=dict(at=datetime.now().astimezone().isoformat(),models=MODELS,
           fitting_train_indices=indices,fit_scenes=sorted(fit),tune_scenes=sorted(tune),
           validation_frames=1024,threshold_grid=dict(min=-256,max=256,step=1/32),
           matched_fpr='Lowest grid threshold with actual validation empty FP count <= B0 native FP count; B0 remains at its native threshold zero',
           native_cohort='GT foreground AND B0 foreground winner correct AND B0 native final empty',
           cohorts_used_for_evaluation_only=True,
           source_sha256=file_sha256(__file__))
    write_json(path,p);return p


def checkpoint_path(name):
    if name in ['B0','S','E','SE']:return RUN/'train'/name/'resume.pth'
    return ROOT/read(RUN/'execution_addendum.json')['archived_checks'][name]['path']


def cache_frame(dest,index,y,fg,margin):
    path=dest/'frames'/f'{index:04d}.npz';path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez(temp,y=y.cpu().numpy().astype(np.uint8),fg=fg.cpu().numpy().astype(np.uint8),
             margin=margin.cpu().numpy().astype(np.float32))
    temp.replace(path)


def extract(name):
    p=evaluation_protocol();dest=RUN/'evaluation'/name;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'extraction_complete.json').exists():return
    checkpoint=checkpoint_path(name);sha=file_sha256(checkpoint)
    metadata=dest/'checkpoint.json'
    if metadata.exists():assert read(metadata)['sha256']==sha
    else:write_json(metadata,dict(path=str(checkpoint.relative_to(ROOT)),sha256=sha))
    seed_all(31928);net,_=build_network(5)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if name in ['B0','S','E','SE']:
        if p.get('endpoint_kind')=='full_history':assert state['epoch']==5 and state['global_iter']==20480
        else:assert state['step']==1024
    net.load_state_dict(state['state_dict'],strict=True);del state
    net.eval();start=time.monotonic();histpath=dest/'train_hist.npz'
    fit=set(p['fit_scenes']);tune=set(p['tune_scenes'])
    if histpath.exists():
        h=np.load(histpath);assert str(h['checkpoint_sha256'])==sha
        hist=h['hist'];native=h['native']
    else:
        hist=torch.zeros(2,18,17,NB,device='cuda',dtype=torch.int64)
        native=torch.zeros(2,18,18,device='cuda',dtype=torch.int64)
        edge=torch.as_tensor(EDGES,device='cuda',dtype=torch.float32)
        dataset,loader,indices=build_loader('train',p['fitting_train_indices'])
        with torch.inference_mode():
            for number,(index,batch) in enumerate(zip(indices,loader)):
                scene=dataset.keyframes[index][0];cohort=0 if scene in fit else 1
                assert scene in fit or scene in tune
                batch=to_cuda(batch);out=network_output(net,batch);y,fg,margin=fields(out,batch)
                bucket=torch.bucketize(margin.contiguous(),edge,right=True)
                code=(y*17+fg)*NB+bucket
                hist[cohort].add_(torch.bincount(code,minlength=18*17*NB).reshape(18,17,NB))
                native[cohort].add_(counts(y,torch.where(margin>=0,fg,17)))
                del out,batch,y,fg,margin,code,bucket
                if (number+1)%128==0:print(name,'train calibration',number+1,len(indices),flush=True)
        hist=hist.cpu().numpy();native=native.cpu().numpy()
        np.savez_compressed(histpath,hist=hist,native=native,checkpoint_sha256=np.array(sha))
    for group in range(2):np.testing.assert_array_equal(counts_from_hist(hist[group],np.zeros(17)),native[group])
    thresholds,boundary=fitting_thresholds(hist[0])
    chosen,selection,candidates=tuning_thresholds(hist[1],thresholds)
    if any(boundary):raise RuntimeError('Calibration fit optimum reached the registered search boundary')
    write_json(dest/'selection.json',dict(thresholds=chosen.tolist(),fitting_thresholds=thresholds.tolist(),
        boundary=boundary,selection=selection,candidates=candidates,training_frames=len(p['fitting_train_indices'])))
    del hist,native
    selected={m['index'] for m in initialize()['scene_samples']}
    missing=[i for i in range(1024) if not (dest/'frames'/f'{i:04d}.npz').exists() or
             (name in ['B0','SE'] and i in selected and not (dest/'gaussians'/f'{i:04d}.npz').exists())]
    dataset,loader,indices=build_loader('val',missing)
    with torch.inference_mode():
        for number,(index,batch) in enumerate(zip(indices,loader)):
            batch=to_cuda(batch);out=network_output(net,batch);y,fg,margin=fields(out,batch)
            cache_frame(dest,index,y,fg,margin)
            if name in ['B0','SE'] and index in selected:
                args=net.head.prepare_gaussian_args(out['gaussian'])
                labels=out['sampled_label'][0].long()
                mask=batch['occ_cam_mask'][0].bool().flatten()&(labels>=0)&(labels<18)
                folder=dest/'gaussians';folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{index:04d}.npz',**{f'args{j}':v.cpu().numpy() for j,v in enumerate(args)},
                    xyz=out['sampled_xyz'][:,mask].cpu().numpy(),y=y.cpu().numpy(),bias=np.array(float(net.head.empty_bias)))
            del out,batch,y,fg,margin
            if (number+1)%128==0:print(name,'validation',number+1,len(missing),'elapsed',round(time.monotonic()-start,1),flush=True)
    metadata=[dict(index=i,scene=dataset.keyframes[i][0],scene_frame=int(dataset.keyframes[i][1])) for i in range(1024)]
    write_json(dest/'validation_frames.json',metadata)
    write_json(dest/'extraction_complete.json',dict(status='complete',name=name,train_frames=len(p['fitting_train_indices']),
        validation_frames=1024,elapsed_seconds=time.monotonic()-start,checkpoint_sha256=sha,
        source_sha256=file_sha256(__file__),protocol_sha256=file_sha256(RUN/'evaluation_protocol.json')))


def numpy_counts(y,pred):
    return np.bincount(y.astype(np.int64)*18+pred,minlength=324).reshape(18,18)


def matched_threshold(hist,target_fp):
    tail=np.cumsum(hist[17,:,::-1],axis=-1,dtype=np.int64)[:,::-1]
    fp=tail[:,1:].sum(0)
    candidates=np.flatnonzero(fp<=target_fp)
    if not len(candidates):raise RuntimeError('Registered margin grid cannot reach the reference false-positive budget')
    index=int(candidates[0])
    return float(EDGES[index]),int(fp[index])


def make_statistics(name):
    dest=RUN/'evaluation'/name
    assert (dest/'extraction_complete.json').exists()
    selected=read(dest/'selection.json')['thresholds']
    base=RUN/'evaluation'/'B0'
    if (dest/'statistics_complete.json').exists():return
    metadata=read(dest/'validation_frames.json')
    scenes=sorted({m['scene'] for m in metadata});lookup={s:i for i,s in enumerate(scenes)}
    empty_hist=np.zeros((len(scenes),NB),dtype=np.int64)
    rescue_hist=np.zeros_like(empty_hist)
    baseline_correct_hist=np.zeros_like(empty_hist)
    baseline_fp=np.zeros(len(scenes),dtype=np.int64)
    cohort_size=np.zeros(len(scenes),dtype=np.int64)
    hist=np.zeros((18,17,NB),dtype=np.int64);target_fp=0
    for index in range(1024):
        a=np.load(dest/'frames'/f'{index:04d}.npz')
        y,fg,margin=a['y'],a['fg'],a['margin']
        bucket=np.searchsorted(EDGES,margin,side='right')
        code=(y.astype(np.int64)*17+fg)*NB+bucket
        hist+=np.bincount(code,minlength=18*17*NB).reshape(18,17,NB)
        b=np.load(base/'frames'/f'{index:04d}.npz')
        scene=lookup[metadata[index]['scene']]
        nf=int(((b['y']==17)&(b['margin']>=0)).sum());target_fp+=nf;baseline_fp[scene]+=nf
        cohort=(y<17)&(b['fg']==y)&(b['margin']<0)
        baseline_correct=(y<17)&(b['fg']==y)&(b['margin']>=0)
        cohort_size[scene]+=int(cohort.sum())
        empty_hist[scene]+=np.bincount(bucket[y==17],minlength=NB)
        rescue_hist[scene]+=np.bincount(bucket[cohort&(fg==y)],minlength=NB)
        baseline_correct_hist[scene]+=np.bincount(bucket[baseline_correct],minlength=NB)
    threshold,fp=matched_threshold(hist,target_fp) if name!='B0' else (0.,target_fp)
    np.savez_compressed(dest/'validation_hist.npz',hist=hist)
    np.savez_compressed(dest/'operating_curves.npz',empty_hist=empty_hist,rescue_hist=rescue_hist,
                        baseline_correct_hist=baseline_correct_hist,
                        baseline_fp=baseline_fp,cohort_size=cohort_size,scenes=np.asarray(scenes))
    rows=[];cohortrows=[]
    for index in range(1024):
        a=np.load(dest/'frames'/f'{index:04d}.npz');b=np.load(base/'frames'/f'{index:04d}.npz')
        y,fg,margin=a['y'],a['fg'],a['margin']
        np.testing.assert_array_equal(y,b['y'])
        basepred=np.where(b['margin']>=0,b['fg'],17)
        cohort=(y<17)&(b['fg']==y)&(basepred==17)
        correct=(y<17)&(basepred==y)
        val=dict(index=index,rank_correct=int(((y<17)&(fg==y)).sum()),modes={})
        group=dict(index=index,cohort=int(cohort.sum()),baseline_correct=int(correct.sum()),modes={})
        for mode,cut in [('native',0.),('calibrated',np.asarray(selected)[fg]),('matched',threshold)]:
            pred=np.where(margin>=cut,fg,17)
            val['modes'][mode]=dict(counts=numpy_counts(y,pred).tolist(),
                recoverable=int(((y<17)&(fg==y)&(pred==17)).sum()))
            group['modes'][mode]=dict(rescued=int((cohort&(pred==y)).sum()),
                remaining_empty=int((cohort&(pred==17)).sum()),
                wrong_foreground=int((cohort&(pred!=17)&(pred!=y)).sum()),
                rank_lost=int((cohort&(fg!=y)).sum()),
                newly_empty=int((correct&(pred==17)).sum()),
                newly_wrong_foreground=int((correct&(pred!=17)&(pred!=y)).sum()))
        rows.append(val);cohortrows.append(group)
    native=np.asarray([r['modes']['native']['counts'] for r in rows]).sum(0)
    np.testing.assert_array_equal(counts_from_hist(hist,np.zeros(17)),native)
    matched=np.asarray([r['modes']['matched']['counts'] for r in rows]).sum(0)
    assert int(matched[17,:17].sum())==fp<=target_fp
    write_json(dest/'statistics.json',dict(name=name,threshold_matched=threshold,target_fp=target_fp,achieved_fp=fp,
        records=rows,cohort_records=cohortrows))
    write_json(dest/'statistics_complete.json',dict(status='complete',frames=1024))
    print(name,'statistics complete matched threshold',threshold,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['initialize','extract','statistics'])
    parser.add_argument('--model',choices=MODELS,default='B0')
    a=parser.parse_args()
    if a.stage=='initialize':print(evaluation_protocol())
    elif a.stage=='extract':extract(a.model)
    else:make_statistics(a.model)


if __name__=='__main__':main()
