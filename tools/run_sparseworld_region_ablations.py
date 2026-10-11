#!/usr/bin/env python3
"""A1/A3/A4 frozen interventions and caches for matched output-head adaptation."""
import argparse, copy, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader, Subset
from dataset.utils import custom_collate_fn_temporal
from tools.diagnose_teacher_alignment import loader_for, tree_to
from tools.experiment_future_response_distillation import build_student, paths_for
from tools.experiment_future2current import baseline, _future_window
from tools.sparseworld_region_utils import ROOT,HORIZONS,write_json,sha256
from tools.sparseworld_region_ablation_common import OUT,CENTERS,GROUPS,windows,window,masks,evaluate,records,patch_mask,shifted_mask

@torch.no_grad()
def perceive(model, images, metas, ego, scale=True):
    state=dict(imgs=images,metas=metas,points=None)
    state.update(model.extract_img_feat(**state)); state.update(model.lifter(**state))
    if scale: state['query_points'] = state['query_points'] * ((torch.tanh(model.points_scale(ego))+1)*.35+.8)
    state.update(model.encoder(**state))
    current=state['query_stamps']==0
    final=state['representation'][-1]
    p=model._world(final['query_points'][:,current]).flatten(1,2)
    l=final['opus_logits'][:,current].flatten(1,2)
    cache=dict(features=final['query_features'][:,current].cpu(),
               centers=model._world(state['representation'][-2]['query_points'][:,current]).mean(2).cpu())
    return p,l,cache,state

@torch.no_grad()
def render(model,p,l,mode='native'):
    raster=copy.copy(model.head.rasterizer)
    if mode=='no_padding': raster.padding=False
    if mode=='threshold_low': raster.threshold=torch.full((17,),.2)
    if mode=='threshold_high': raster.threshold=torch.full((17,),.5)
    # group_size=1 disables both group compactness and center-distance filters.
    group=1 if mode=='no_group_filter' else 48
    return raster(p.clone(),l.clone(),group).reshape(200,200,16).cpu().numpy().astype(np.uint8)

def thin(p,l,n,seed):
    q=p.shape[1]//48
    ids=np.sort(np.random.default_rng(seed).choice(q,min(n,q),replace=False))
    ix=torch.as_tensor(ids,device=p.device)
    return p.reshape(1,q,48,3)[:,ix].flatten(1,2),l.reshape(1,q,48,17)[:,ix].flatten(1,2)

def spatial_equal(f,p,seed=20261008):
    def bins(points):
        c=points.reshape(-1,48,3).mean(1).cpu().numpy()
        xy=np.floor((c[:,:2]+40)/4).astype(int)
        return xy[:,0]*1000+xy[:,1]
    a,b=bins(f[0]),bins(p[0]); ia=[];ib=[]
    rng=np.random.default_rng(seed)
    for key in np.intersect1d(a,b):
        x=np.flatnonzero(a==key);y=np.flatnonzero(b==key);n=min(len(x),len(y))
        ia.extend(rng.choice(x,n,False));ib.extend(rng.choice(y,n,False))
    def select(pair,ids):
        ix=torch.as_tensor(sorted(ids),device=pair[0].device)
        return tuple(t.reshape(1,-1,48,t.shape[-1])[:,ix].flatten(1,2) for t in pair)
    return select(f,ia),select(p,ib),len(ia)

def swap_logits(dst,src):
    from scipy.spatial import cKDTree
    distances,ids=cKDTree(src[0][0].cpu().numpy()).query(dst[0][0].cpu().numpy(),workers=1)
    keep=distances<=1.2
    out=dst[1].clone()
    out[:,torch.as_tensor(keep,device='cuda')]=src[1][:,torch.as_tensor(ids[keep],device='cuda')]
    return (dst[0],out),float(keep.mean())

@torch.no_grad()
def forecast_images(model,images,metas,ego):
    _,_,_,state=perceive(model,images,metas,ego)
    enc=state['representation'][-1]
    source=dict(encoded=dict(query_features=enc['query_features'],query_points=enc['query_points']),
                stamps=state['query_stamps'],ego=ego)
    pred=baseline(model,source)
    return pred['points'][5],pred['logits'][5]

def roi_mask(group):
    ids=np.flatnonzero(group)
    if len(ids)==0:return None,None
    blocks=np.floor((CENTERS[ids,:2]+40)/2).astype(int)
    codes=blocks[:,0]*40+blocks[:,1]
    counts=np.bincount(codes,minlength=1600)
    best=int(counts.argmax())
    roi=np.zeros(len(CENTERS),bool);roi[ids[codes==best]]=True
    xy=np.array([best//40,best%40])*2-39
    return roi,xy

@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py')
    manifest=json.loads(Path('out/future2current/feature_identity_2048x512/manifest.json').read_text())
    split=args.split
    loader,sel=loader_for(cfg,split,512 if split=='val' else 2048,args.workers,cfg.identity_sample_seed)
    assert sel==manifest[split]
    recs=records() if split=='val' else [dict(position=i,index=idx,scene=sc,token=str(idx)) for i,(idx,sc) in enumerate(zip(sel['indices'][:args.train_samples],sel['scenes']))]
    if args.limit:recs=recs[:args.limit]
    destroot=OUT/('val' if split=='val' else 'train_cache');destroot.mkdir(parents=True,exist_ok=True)
    (OUT/'val_cache').mkdir(exist_ok=True)
    write_json(OUT/(split+'_selection.json'),recs)
    model=build_student(cfg,Config.fromfile(cfg.student_model_config))
    assert sha256(cfg.student_checkpoint)==cfg.student_checkpoint_sha256
    missing=[r for r in recs if not (destroot/(r['token']+'.'+('npz' if split=='val' else 'pth'))).exists()]
    sources=paths_for(Path(manifest['source_cache']),split,manifest[split])
    batches=DataLoader(Subset(loader.dataset.dataset,[r['index'] for r in missing]),batch_size=1,
                       num_workers=args.workers,collate_fn=custom_collate_fn_temporal,pin_memory=True)
    for n,(rec,batch) in enumerate(zip(missing,batches)):
        start=time.monotonic();source=torch.load(sources[rec['position']],map_location='cpu')
        data=tree_to(batch,'cuda');imgs=data.pop('img');ego=source['ego'].cuda()
        if split=='val': assert data['sample_idx'][0]==rec['token']
        gt=source['metas']['future_occ_labels'][0,[1,3,5]].numpy()
        if split=='val':
            pair=dict(np.load(ROOT/'pairs'/(rec['token']+'.npz')))
            attrs=dict(np.load(ROOT/'attributes'/(rec['token']+'.npz')))
            assert np.array_equal(gt,pair['GT'])
            froll=baseline(model,source)
        hist={};counts={};cache={};diagnostics={};interventions={}
        def add(name,j,pred,groups):
            if name not in hist:
                hist[name]=np.zeros((3,18,18),np.int64);counts[name]=np.zeros((3,len(GROUPS),3),np.int64)
            hist[name][j],counts[name][j]=evaluate(gt[j],pred,groups)
        for j,h in enumerate(HORIZONS):
            groups=masks(rec,j,pair,attrs) if split=='val' else None
            output={}
            for arm,rels in windows(h).items():
                selected,metas=window(data,imgs,rels,h)
                if arm=='P':
                    orig,origmeta=_future_window(data,imgs,h)
                    assert torch.equal(orig,selected) and torch.equal(origmeta['projection_mat'],metas['projection_mat'])
                p,l,c,_=perceive(model,selected,metas,ego)
                output[arm]=(p,l);cache[f'{arm}_{h}']=c
                if split=='val':
                    pred=render(model,p,l)
                    if arm=='P':assert np.array_equal(pred,pair['P'][j]),'P replay failed'
                    add(arm,j,pred,groups)
            if split=='train':continue
            f=(froll['points'][h-1],froll['logits'][h-1]);p=output['P']
            assert np.array_equal(render(model,*f),pair['F'][j]),'F replay failed'
            add('F',j,pair['F'][j],groups)
            hp=tuple(torch.cat([a,b],1) for a,b in zip(output['H'],p))
            add('HP_full',j,render(model,*hp),groups)
            # Duplicate inference/output is the equal-cost H+H / P+P control.
            for arm in ('H','P'):
                twice=tuple(torch.cat([x,x],1) for x in output[arm])
                assert np.array_equal(render(model,*twice),render(model,*output[arm]))
            halves=[thin(*output[a],360,20261008) for a in ('H','P')]
            add('HP_equal_points',j,render(model,*tuple(torch.cat(t,1) for t in zip(*halves))),groups)
            add('F_equal_count',j,render(model,*thin(*f,720,20261008)),groups)
            fe,pe,q=spatial_equal(f,p)
            diagnostics[f'equal_spatial_queries_{h}']=q
            add('F_equal_spatial',j,render(model,*fe),groups);add('P_equal_spatial',j,render(model,*pe),groups)
            for mode in ('no_padding','no_group_filter','threshold_low','threshold_high'):
                for a,pr in [('F',f),('P',p)]:add(a+'_'+mode,j,render(model,*pr,mode),groups)
            for a,dst,src in [('F_geometry_P_semantics',f,p),('P_geometry_F_semantics',p,f)]:
                swapped,fraction=swap_logits(dst,src);diagnostics[f'{a}_match_fraction_{h}']=fraction
                add(a,j,render(model,*swapped),groups)
            selected,metas=window(data,imgs,windows(h)['P'],h)
            pp,ll,_,_=perceive(model,selected,metas,ego,scale=False)
            add('P_unit_scale',j,render(model,pp,ll),groups)
            # Raw and native rendered coverage, separated from semantic correctness.
            for a,pr in [('F',f),('P',p)]:
                ix=np.floor((pr[0][0].cpu().numpy()-[-40,-40,-1])/.4).astype(int)
                inside=((ix>=0)&(ix<[200,200,16])).all(1);ix=ix[inside]
                coverage=np.zeros((200,200,16),bool);coverage[ix[:,0],ix[:,1],ix[:,2]]=True
                diagnostics[f'{a}_coverage_{h}']=[[int(g.sum()),int((coverage.reshape(-1)&g).sum())] for g in groups]
            if h!=6:continue
            ih,mh=window(data,imgs,windows(h)['H'],0)
            ip,mp=window(data,imgs,windows(h)['P'],h)
            fcheck=forecast_images(model,ih,mh,ego)
            assert np.array_equal(render(model,*fcheck),pair['F'][j]),'image F replay failed'
            for roi_name,roi_group in [('future_only_static',groups[2]),('current_near_static',groups[3]&groups[1])]:
                roi,xy=roi_mask(roi_group)
                if roi is None:
                    interventions[roi_name]={'available':False};continue
                # Surrounding static GT provides a geometric context region; no correctness-based selection.
                context=(np.linalg.norm(CENTERS[:,:2]-xy,axis=1)<=8)&groups[1]
                projected={
                    'F':patch_mask(CENTERS[context],pair['F_projection'][j],pair['F_wh'][j],ih.shape[-2],ih.shape[-1]),
                    'P':patch_mask(CENTERS[roi],pair['P_projection'][j],pair['P_wh'][j],ip.shape[-2],ip.shape[-1])}
                detail=dict(available=True,roi_voxels=int(roi.sum()),xy=xy.tolist(),effects={})
                for a,im,me in [('F',ih,mh),('P',ip,mp)]:
                    mask=projected[a];detail[a+'_mask_pixels']=int(mask.sum());detail[a+'_mask_fraction']=float(mask.mean())
                    base=pair[a][j].reshape(-1)
                    detail['effects'][a+'_baseline']=[int((base[roi]==gt[j].reshape(-1)[roi]).sum()),int(roi.sum())]
                    for mode,ma in [('targeted',mask),('random1',shifted_mask(mask,20261008)),('random2',shifted_mask(mask,20261009))]:
                        altered=im.masked_fill(torch.as_tensor(ma,device='cuda')[None,:,None],0.)
                        if a=='F': pp,ll=forecast_images(model,altered,me,ego)
                        else:pp,ll,_,_=perceive(model,altered,me,ego)
                        pred=render(model,pp,ll)
                        add(f'{a}_{roi_name}_{mode}',j,pred,groups)
                        detail['effects'][a+'_'+mode]=[int((pred.reshape(-1)[roi]==gt[j].reshape(-1)[roi]).sum()),int(roi.sum())]
                interventions[roi_name]=detail
        if split=='train':
            cache['GT']=torch.as_tensor(gt.astype(np.uint8));cache['scene']=rec['scene'];cache['index']=rec['index']
            torch.save(cache,destroot/(rec['token']+'.pth'))
        else:
            torch.save(cache,OUT/'val_cache'/(rec['token']+'.pth'))
            names=sorted(hist)
            np.savez_compressed(destroot/(rec['token']+'.npz'),names=np.array(names),hist=np.stack([hist[k] for k in names]),
                                counts=np.stack([counts[k] for k in names]))
            write_json(destroot/(rec['token']+'.json'),dict(diagnostics=diagnostics,interventions=interventions,
                seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),scene=rec['scene'],token=rec['token']))
        print(f'[{split}] {n+1}/{len(missing)} {rec["token"]} {time.monotonic()-start:.1f}s',flush=True)
        del cache,data,imgs,source,output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','train'],default='val');p.add_argument('--limit',type=int)
    p.add_argument('--train-samples',type=int,default=256);p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
