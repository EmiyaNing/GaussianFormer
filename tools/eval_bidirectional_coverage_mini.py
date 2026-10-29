"""Full validation for both final-stage bidirectional geometry objectives."""
from pathlib import Path
import argparse,fcntl,json,os,sys,time
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from tools.bidirectional_coverage_experiment import *
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache
from tools.probe_gaussian_scale_supervision import layout
from tools.semantic_field_support import extension
from tools.eval_semantic_neighborhood_mean import (
    frame, atomic_npz, QUERY_FIELDS, G_FIELDS, HIGH_FIELDS, PAIR_FIELDS, reference_path)

def evaluate(arm,smoke=False):
    if not smoke:assert_protocol()
    assert arm in ARMS
    checkpoint=BASELINE if smoke else RUN/'train'/arm/'epoch_1.pth'
    checkpoint_sha=sha(checkpoint);folder=RUN/('smoke_eval' if smoke else 'eval')/arm
    if (folder/'complete.json').exists() and not smoke:
        assert json.loads((folder/'complete.json').read_text())['checkpoint_sha256']==checkpoint_sha;return
    c,net=build(arm,checkpoint)
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
                _,values=layout(net.head,out['gaussian'])
                classes=values[4].argmax(-1).cpu().numpy().astype(np.int64)
                row['GS_class_count']=np.bincount(classes,minlength=17)
                for reference in ('initial','baseline'):
                    with np.load(reference_path(reference,index)) as saved:
                        original=saved['classes'].astype(np.int64)
                    row[reference+'_GS_class_count']=np.bincount(original,minlength=17)
                    row[reference+'_to_current_classes']=np.bincount(original*17+classes,minlength=289).reshape(17,17)
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
            class_coverage_columns=['GT','geo1','geo2','geo3','same1','same2','same3','initial_class_geo1','native'])
        write(folder/'complete.json',result);write(folder/'progress.json',result)
    finally:cache.close();net.flush_memory_phase()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('arm',choices=ARMS);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();os.chdir(ROOT);RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);evaluate(a.arm,a.smoke)
