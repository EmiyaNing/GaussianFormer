"""Full validation: actual final-GS core support, semantic support and query load."""
from pathlib import Path
import argparse,fcntl,json,os,sys,time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from tools.semantic_neighborhood_experiment import *
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.semantic_field_support import extension
Q,K=1200,64
# Field order is frozen in output JSON, with explicit denominators in the report.
G_FIELDS=['gaussians','center_camera_fg','center_camera_empty','center_unknown','major_scale_saturated',
          'major_scale_sum','opacity_sum','semantic_amplitude_sum']
QUERY_FIELDS=['queries','zero_core','core_le5','core_le10','core_le20','nearest_top20_GT',
              'core_top20_union_GT','query_core_GT_pairs','zero_nearest']
HIGH_FIELDS=['GT','core','same_core','detected','semantic_correct']
PAIR_FIELDS=['geom_added','geom_lost','same_added','same_lost','GT']

def top20(values):return np.argsort(values,kind='stable')[-Q//5:]
def packed(ids,device):
    a=np.zeros((Q+63)//64,np.uint64)
    for q in ids:a[int(q)//64]|=np.uint64(1)<<np.uint64(int(q)%64)
    return torch.from_numpy(a.view(np.int64)).to(device)
def reference_path(arm,index):return REFERENCE_RUN/'references'/arm/f'{index:05d}.npz'
def atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
    path.with_suffix('.tmp').replace(path)

@torch.inference_mode()
def frame(net,out,batch,arm,index,validate=False):
    from loss.opus_set_loss import _nearest_indices
    _,v=layout(net.head,out['gaussian']);means,scales,inverse,alpha,sem,centers,radii=v
    assert len(means)==Q*K
    labels=batch['occ_label'].flatten().int().contiguous();camera=batch['occ_cam_mask'].flatten().bool()
    valid=camera&(labels>=0)&(labels<18);fg=valid&(labels<17);ids=torch.where(fg)[0]
    xyz=out['sampled_xyz'][0].contiguous();pred=out['pred_occ'][-1][0].argmax(0).flatten().int()
    cm=torch.bincount((labels[valid]*18+pred[valid]).long(),minlength=324).reshape(18,18).cpu().numpy()
    classes=sem.argmax(-1).int().contiguous();initial=None
    if arm!='initial':
        initial=dict(np.load(reference_path('initial',index)))
        np.testing.assert_array_equal(ids.cpu().numpy(),initial['ids'])
    frozen=classes if initial is None else torch.from_numpy(initial['classes'].astype(np.int32)).cuda()
    lookup=torch.full_like(labels,-1);lookup[ids]=torch.arange(len(ids),device='cuda',dtype=torch.int32)
    agg=net.head.aggregator
    bits,counts,coverage=extension().support(means,inverse,alpha,centers,radii,xyz,lookup,
        labels,classes,frozen,len(ids),Q,K,agg.H,agg.W,agg.D)
    if validate:
        from tools.query_final_support import extension as old_extension
        oldbits,oldcoverage,distinct=old_extension().support(means,inverse,alpha,centers,radii,xyz,
            lookup,len(ids),Q,K,agg.H,agg.W,agg.D)
        assert torch.equal(oldbits[0],bits) and torch.equal(oldcoverage[0],coverage)
        assert torch.equal(counts[:,0]>0,distinct[0]>0) and torch.equal(counts[:,2]>0,distinct[1]>0)
        assert torch.equal(counts[:,7]>0,distinct[2]>0)
        # Exhaustive GS count at 16 GT centers; independent vector formula, no spatial hash.
        for t in np.random.default_rng(928).choice(len(ids),min(16,len(ids)),replace=False):
            x=xyz[ids[t]];center=((x-agg.pc_min[0])/agg.grid_size).long()
            box=((center>=centers-radii[:,None])&(center<=centers+radii[:,None])).all(1)
            d=means-x
            power=-.5*(inverse[:,:3]*d.square()).sum(1)-(inverse[:,3:]*d[:,[0,1,0]]*d[:,[1,2,2]]).sum(1)
            positive=box&(alpha*power.exp()>0);distance=(-2*power).clamp_min(0).sqrt()
            same=classes==labels[ids[t]];fr=frozen==labels[ids[t]]
            expected=[int((positive&(distance<=r)).sum()) for r in (1,2,3)]
            expected +=[int((positive&(distance<=r)&same).sum()) for r in (1,2,3)]
            expected +=[int((positive&(distance<=1)&fr).sum()),int(positive.sum())]
            np.testing.assert_array_equal(counts[t].cpu().numpy(),expected)
    hit=counts>0
    load=torch.bincount(_nearest_indices(xyz[ids],means,1024)//K,minlength=Q).cpu().numpy()
    cov=coverage.cpu().numpy();high_initial=(np.isin((_nearest_indices(xyz[ids],means,1024)//K).cpu().numpy(),top20(load))
        if initial is None else initial['high'])
    h=torch.from_numpy(high_initial).cuda();fgpred=pred[ids];fglabel=labels[ids]
    high=np.array([int(h.sum()),int((h&hit[:,0]).sum()),int((h&hit[:,3]).sum()),
                   int((h&(fgpred<17)).sum()),int((h&(fgpred==fglabel)).sum())],np.int64)
    cc=torch.zeros((17,9),device='cuda',dtype=torch.int64)
    cc[:,0]=torch.bincount(fglabel.long(),minlength=17)
    for j in range(8):cc[:,j+1]=torch.bincount(fglabel[hit[:,j]].long(),minlength=17)
    top_union=(bits&packed(top20(cov),bits.device)).ne(0).any(1)
    queries=np.array([Q,int((cov==0).sum()),int((cov<=5).sum()),int((cov<=10).sum()),int((cov<=20).sum()),
        int(load[top20(load)].sum()),int(top_union.sum()),int(cov.sum()),int((load==0).sum())],np.int64)
    coord=((means-agg.pc_min[0])/agg.grid_size).long()
    shape=torch.tensor([agg.H,agg.W,agg.D],device='cuda');inside=((coord>=0)&(coord<shape)).all(1)
    safe=torch.minimum(coord.clamp_min(0),shape-1);flat=(safe[:,0]*agg.W+safe[:,1])*agg.D+safe[:,2]
    known=inside&valid[flat];camerafg=known&(labels[flat]<17);cameraempty=known&(labels[flat]==17)
    major=scales.max(-1).values
    gs=np.array([len(means),int(camerafg.sum()),int(cameraempty.sum()),int((~known).sum()),
        int((major>=.3999).sum()),float(major.sum()),float(alpha.sum()),float(sem.sum())])
    hits_np=hit.cpu().numpy();pairs={}
    for refname in ('initial','baseline'):
        if arm==refname or (refname=='baseline' and arm=='initial'):continue
        ref=initial if refname=='initial' else dict(np.load(reference_path('baseline',index)))
        a=ref['hit'];b=hits_np
        pairs[refname]=np.array([int((b[:,0]&~a[:,0]).sum()),int((a[:,0]&~b[:,0]).sum()),
            int((b[:,3]&~a[:,3]).sum()),int((a[:,3]&~b[:,3]).sum()),len(ids)],np.int64)
    return dict(confusion=cm,class_coverage=cc.cpu().numpy(),queries=queries,high=high,gaussians=gs,
        load=load,query_core=cov,**{'paired_'+k:v for k,v in pairs.items()})


def evaluate(arm,smoke=False):
    if not smoke:assert_protocol()
    assert arm in ('baseline',*ARMS)
    checkpoint=BASELINE if arm=='baseline' else RUN/'train'/arm/'epoch_1.pth'
    checkpoint_sha=sha(checkpoint);folder=RUN/('smoke_eval' if smoke else 'eval')/arm
    if (folder/'complete.json').exists() and not smoke:
        assert json.loads((folder/'complete.json').read_text())['checkpoint_sha256']==checkpoint_sha;return
    c,net=build(ARMS[0],checkpoint)
    from loss.semantic_neighborhood_mean_loss import SemanticNeighborhoodMeanLoss
    extra=SemanticNeighborhoodMeanLoss().cuda()
    from dataset.utils import custom_collate_fn_temporal
    dataset=EvaluationFrames(c,'val');assert len(dataset)==6019
    scenes={}
    for i,(scene,_) in enumerate(dataset.keyframes):scenes.setdefault(scene,[]).append(i)
    assert len(scenes)==150
    pending=[]
    for scene,ids in scenes.items():
        saved=folder/'scenes'/f'{scene}.json'
        if saved.exists() and not smoke:
            meta=json.loads(saved.read_text());assert meta['indices']==ids and meta['checkpoint_sha256']==checkpoint_sha
            assert saved.with_suffix('.npz').exists()
        else:pending.extend(ids)
    if smoke:pending=pending[:2]
    loader=DataLoader(Subset(dataset,pending),batch_size=1,num_workers=3,shuffle=False,
        collate_fn=custom_collate_fn_temporal,pin_memory=True,persistent_workers=True,prefetch_factor=2)
    cache=FeatureCache(net,dataset);start=time.monotonic();active=None;done=0
    try:
        for index,batch in zip(pending,loader):
            scene,frameindex=dataset.keyframes[index]
            if active is None:active=scene;arrays={};indices=[];tokens=[]
            assert active==scene
            batch=to_cuda(batch)
            with torch.inference_mode():
                out=forward(net,batch)
                row=frame(net,out,batch,arm,index,validate=(done==0))
                pieces,diagnostics=extra.components(loss_input(c,out,batch))
                n=int(diagnostics['gs_total']);j=int(diagnostics['GT_foreground'])
                row['aux_geometry']=np.array([n,j,int(diagnostics['gs_local_same3']),
                    int(diagnostics['local_same3_pairs']),float(pieces['mean3'])*n,
                    int(diagnostics['uncovered_GT']),float(pieces['uncovered1'])*j,
                    int(round(float(diagnostics['same_core_GT_fraction'])*j)),
                    int(round(float(diagnostics['idle_core_GS_fraction'])*n)),
                    int(round(float(diagnostics['opacity_zero_fraction'])*n))],np.float64)
            for k,a in row.items():arrays.setdefault(k,[]).append(a)
            indices.append(index);tokens.append(dataset.base.scene_infos[scene][frameindex]['token']);done+=1
            if index==scenes[scene][-1] or (smoke and done==len(pending)):
                path=folder/'scenes'/f'{scene}.npz';atomic_npz(path,**{k:np.stack(v) for k,v in arrays.items()})
                write(path.with_suffix('.json'),dict(status='complete',indices=indices,tokens=tokens,
                    checkpoint_sha256=checkpoint_sha,scene=scene,frames=len(indices)))
                progress=dict(status='running',arm=arm,scenes=len(list((folder/'scenes').glob('*.json'))),
                    expected_scenes=150,processed=done,frames_per_second=done/(time.monotonic()-start))
                write(folder/'progress.json',progress);print(json.dumps(progress),flush=True);active=None
            del out,batch,row
        if smoke:
            write(folder/'smoke_passed.json',dict(status='passed',frames=done,
                old_bitsets_exact=True,exhaustive_count_GT_positions=16));return
        paths=list((folder/'scenes').glob('*.json'));allids=[]
        for path in paths:allids+=json.loads(path.read_text())['indices']
        assert len(paths)==150 and sorted(allids)==list(range(6019))
        result=dict(status='complete',frames=6019,scenes=150,checkpoint_sha256=checkpoint_sha,
            query_fields=QUERY_FIELDS,gaussian_fields=G_FIELDS,high_fields=HIGH_FIELDS,pair_fields=PAIR_FIELDS,
            aux_fields=['GS','GT','GS_with_same3','same3_pairs','sum_GS_mean_distance_m',
                'uncovered_GT','sum_uncovered_center_distance_m','same_core_GT','idle_core_GS','opacity_zero_GS'],
            class_coverage_columns=['GT','geo1','geo2','geo3','same1','same2','same3','initial_class_geo1','native'])
        write(folder/'complete.json',result);write(folder/'progress.json',result)
    finally:cache.close();net.flush_memory_phase()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('arm',choices=('baseline',*ARMS));p.add_argument('--smoke',action='store_true')
    a=p.parse_args();os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);evaluate(a.arm,a.smoke)
