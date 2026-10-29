"""Immutable data preparation and exact sufficient statistics for class bias."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.empty_decision_optimization import (
    RUN as ORIGINAL, CONFIG, CHECKPOINT, read, write_json, file_sha256,
    build_loader, build_network, to_cuda, seed_all)
from tools.empty_decision_followup import RUN as CACHE
from tools.eval_empty_decision_optimization import region_metrics, metrics

RUN = ROOT/'work_dirs/learnable_class_bias'
OUT = ROOT/'analysis/empty_decision/class_bias'
SHRINK = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
OUT_COST = [0., 10., 100., 1000.]
ALL_COST = [10., 100., 1000.]
BUDGETS = [0., .005, .02, None]
SHIFTS = np.arange(-32, 33)/32


def initialize():
    RUN.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    path = RUN/'protocol.json'
    if path.exists(): return read(path)
    old = read(ORIGINAL/'protocol.json')
    record = dict(created_at=datetime.now().astimezone().isoformat(),
        config=CONFIG, checkpoint=CHECKPOINT, checkpoint_sha256=old['checkpoint_sha256'],
        original_bias=read(ORIGINAL/'train/frame_0000.json')['learned_bias'],
        original_protocol_sha256=file_sha256(ORIGINAL/'protocol.json'),
        original_selection_sha256=file_sha256(ORIGINAL/'operating_points.json'),
        cache_manifest_sha256={s:file_sha256(CACHE/(s+'_cache.json')) for s in ['train','val']},
        fit_scenes=old['fit_scenes'], tune_scenes=old['tune_scenes'], validation_scenes=old['validation_scenes'],
        train_frames=4096, fit_frames=3102, tuning_frames=994, validation_frames=1024,
        fit_rows=400000, tune_rows=200000, band=2.,
        parameterization='17 total biases; report beta=mean(b), delta=b-beta; optimize additional a=b-b0',
        fitting_bounds=[-2.,2.], postfit_shift_range=[-1.,1.], postfit_shift_step=1/32,
        ce=dict(loss='exact 18-way CE on IPW sampled candidates, parameters independent of fixed voxels',
                optimizer='bounded L-BFGS-B; Torch float64 autograd', maxiter=300,ftol=1e-12,gtol=1e-8,
                shrink_grid=SHRINK,shrink_selection='minimum IPW tuning candidate CE; no validation labels'),
        metric=dict(steps=1600,lr=.03,lr_final=.003,temperature_start=.4,temperature_end=.05,
                    margin_bins=1024,bin_width=4/1024,optimizer='Adam, no weight decay',
                    rho_out=OUT_COST,rho_all=ALL_COST,starts=['CE_shrunk','zero'],
                    selected_checkpoint='minimum fitting J at final temperature, checked every 40 updates',
                    shrink='same coefficient as C3 chosen using training tuning CE'),
        budgets=BUDGETS, primary_budget=.005, all_miou_tolerance=.0005,
        constraint_reference='risk: native B0; all-mIoU: same-data B2 at corresponding training risk budget',
        risk_matching_tolerance_pp=.25,
        same_data_B2='same 400k IPW fitting rows and fixed counts; class threshold step1/32; dense dual proposals; full tune selects proposal and common shift',
        primary='C5 primary-budget rule selected on full tuning census; fallback explicitly identified',
        bootstrap=dict(replicates=10,seed=9280,unit='523 fitting scenes sampled with replacement',
                       models=['C2','C3','C5'],C5_config='main training-selected config, refit without reselection of its costs',
                       calibration='same full tuning scenes, same shift search',purpose='fitting-scene stability, not backbone seeds'),
        evaluation='exact hard decisions, all1024 frames; all choices frozen before new method evaluation',
        ray='official OPUS, all1024, main methods only',benchmark='8frames,2warmups,5repeats,random order,GPU resident full network',
        no_joint_backbone_training=True,exploratory='existing validation informed research; not untouched confirmation',
        source_sha256=file_sha256(__file__))
    write_json(path,record)
    return record


def load_samples(cohort):
    with np.load(CACHE/f'samples_{cohort}.npz') as z:return {k:z[k] for k in z.files}


def prepare_maps():
    p=initialize();dest=RUN/'sample_maps.npz'
    if dest.exists():return
    frames=read(ORIGINAL/'train/summary.json')['frames'];result={}
    for cohort in ['fit','tune']:
        chosen=[f for f in frames if (f['scene'] in p['fit_scenes']) == (cohort=='fit')]
        voxel=[];fr=[]
        for f in chosen:
            with np.load((ROOT/f['path']).with_suffix('.npz')) as z:
                voxel.append(z['voxel_id']);fr.append(np.full(len(z['voxel_id']),f['index'],np.int16))
        voxel=np.concatenate(voxel);fr=np.concatenate(fr);n=len(fr)
        cap=p[cohort+'_rows'];ids=np.random.default_rng(928 if cohort=='fit' else 929).choice(n,cap,replace=False)
        data=load_samples(cohort);assert np.array_equal(fr[ids],data['frame'])
        result[cohort+'_voxel']=voxel[ids];result[cohort+'_frame']=fr[ids]
    np.savez_compressed(dest,**result)


def extract_ce(limit=0):
    p=initialize();prepare_maps();seed_all(928)
    assert file_sha256(ROOT/CHECKPOINT)==p['checkpoint_sha256']
    dest=RUN/'ce_frames';dest.mkdir(exist_ok=True)
    with np.load(RUN/'sample_maps.npz') as z:maps={k:z[k] for k in z.files}
    arrays={cohort:{'lse':np.zeros(p[cohort+'_rows'],np.float64),
                    'target':np.zeros(p[cohort+'_rows'],np.float64)} for cohort in ['fit','tune']}
    data={c:load_samples(c) for c in arrays}
    _,loader,indices=build_loader('train',range(limit) if limit else None)
    net=build_network();frames=[];started=time.monotonic()
    with torch.inference_mode():
        for index,batch in zip(indices,loader):
            path=dest/f'frame_{index:04d}.npz';cohort='fit' if (maps['fit_frame']==index).any() else 'tune'
            positions=np.where(maps[cohort+'_frame']==index)[0]
            voxel=maps[cohort+'_voxel'][positions]
            if path.exists():
                with np.load(path) as z:
                    assert np.array_equal(z['positions'],positions)
                    values={k:z[k] for k in ['lse','target']}
            else:
                batch=to_cuda(batch)
                out=net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k!='img'})
                scores=out['pred_occ'][-1][0,:,torch.as_tensor(voxel,device='cuda').long()].T.double()
                top,fg=scores[:,:17].max(-1);margin=top-scores[:,17]
                assert np.array_equal(fg.cpu().numpy(),data[cohort]['fg'][positions])
                assert np.allclose(margin.cpu().numpy(),data[cohort]['X'][positions,0],atol=2e-6,rtol=2e-6)
                gt=torch.as_tensor(data[cohort]['gt'][positions].astype(np.int64),device='cuda')
                values=dict(lse=(scores[:,:17].logsumexp(-1)-scores[:,17]).cpu().numpy(),
                            target=(scores.gather(1,gt[:,None])[:,0]-scores[:,17]).cpu().numpy())
                # The exact sufficient-statistic formula reproduces full CE.
                check=torch.logaddexp(torch.as_tensor(values['lse'],device='cuda'),torch.zeros(len(gt),device='cuda'))-torch.as_tensor(values['target'],device='cuda')
                torch.testing.assert_close(check,torch.nn.functional.cross_entropy(scores,gt,reduction='none'),atol=1e-10,rtol=1e-10)
                np.savez_compressed(path,positions=positions,**values)
            for k in values:arrays[cohort][k][positions]=values[k]
            frames.append(index)
            if index==0 or (index+1)%128==0:
                print('exact CE extraction',index+1,'/4096','seconds',round(time.monotonic()-started,1),flush=True)
                write_json(RUN/'status.json',dict(status='running',stage='exact_ce_extraction',frames=index+1))
    if not limit:
        for cohort,v in arrays.items():np.savez_compressed(RUN/f'ce_{cohort}.npz',**v)
        write_json(RUN/'ce_manifest.json',dict(status='complete',frames=frames,rows={c:len(v['lse']) for c,v in arrays.items()},
            exact_full18_CE_formula_checked=True,checkpoint_sha256=p['checkpoint_sha256'],
            source_sha256=file_sha256(__file__),elapsed_seconds=time.monotonic()-started))


def sorted_statistics(margin,code,fixed,native=None):
    order=np.lexsort((margin,code));counts=np.bincount(code,minlength=612)
    return dict(margin=margin[order],offsets=np.r_[0,np.cumsum(counts)],fixed=fixed,
                native=native if native is not None else np.zeros((2,18,18),np.int64))


def exact_counts(stat,thresholds,events=False):
    thresholds=np.atleast_2d(thresholds);n=len(thresholds)
    selected=np.zeros((n,2,18,17),np.float64 if stat['margin'].dtype==np.float64 else np.int64)
    total=np.zeros((2,18),np.int64);native=np.zeros((2,18,17),np.int64)
    for code in range(612):
        r,g,c=code//306,(code//17)%18,code%17
        start,end=stat['offsets'][code:code+2];a=stat['margin'][start:end];total[r,g]+=len(a)
        selected[:,r,g,c]=len(a)-np.searchsorted(a,thresholds[:,c],side='left')
        if events:native[r,g,c]=len(a)-np.searchsorted(a,0.,side='left')
    cc=np.broadcast_to(stat['fixed'],(n,2,18,18)).astype(selected.dtype).copy()
    cc[:,:,:,:17]+=selected;cc[:,:,:,17]+=total[None]-selected.sum(-1)
    if not events:return cc
    delta=selected-native[None];up=np.maximum(delta,0);down=np.maximum(-delta,0)
    good=np.eye(18,17,dtype=bool);empty=np.zeros_like(good);empty[17]=True;wrong=~good&~empty
    ev=np.stack([up[:,:,good].sum(-1),up[:,:,wrong].sum(-1),up[:,:,empty].sum(-1),
                 down[:,:,good].sum(-1),down[:,:,wrong].sum(-1),down[:,:,empty].sum(-1)],-1)
    return cc,ev


def build_tuning():
    initialize();dest=RUN/'tuning';dest.mkdir(exist_ok=True)
    if (dest/'manifest.json').exists():return
    parts=[];codes=[];fixed=np.zeros((2,18,18),np.int64);native=fixed.copy();frames=[];started=time.monotonic()
    for meta in read(CACHE/'train_cache.json')['frames']:
        if not meta['full_candidates']:continue
        with np.load(ROOT/meta['path']) as z:
            parts.append(z['X'][:,0].copy())
            code=z['region'].astype(np.int16)*306+z['gt'].astype(np.int16)*17+z['fg']
            codes.append(code);fixed+=z['fixed_confusion'];native+=z['native_confusion']
        frames.append(meta['index'])
        if len(frames)%128==0:print('tuning census',len(frames),'/994',flush=True)
    stat=sorted_statistics(np.concatenate(parts),np.concatenate(codes),fixed,native)
    np.save(dest/'margin_sorted.npy',stat['margin']);np.savez_compressed(dest/'counts.npz',offsets=stat['offsets'],fixed=fixed,native=native)
    assert np.array_equal(exact_counts(stat,np.zeros(17))[0],native)
    write_json(dest/'manifest.json',dict(status='complete',frames=frames,rows=len(stat['margin']),
        elapsed_seconds=time.monotonic()-started,source_sha256=file_sha256(__file__)))


def tuning_statistics():
    with np.load(RUN/'tuning/counts.npz') as z:stat={k:z[k] for k in z.files}
    stat['margin']=np.load(RUN/'tuning/margin_sorted.npy',mmap_mode='r')
    return stat


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['initialize','extract','tuning']);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    if args.stage=='initialize':initialize();prepare_maps()
    elif args.stage=='extract':extract_ce(args.limit)
    else:build_tuning()
