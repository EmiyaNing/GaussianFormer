#!/usr/bin/env python3
"""Export authentic same-weight future-perception raw logits on GT-fast voxels."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from tools.run_sparseworld_fast_motion_distillation import loaders,image_loader,window_for,reuse_targets,cache_path
from tools.sparseworld_fast_motion_common import build,HORIZONS
from tools.sparseworld_raw_logit_kd import raw_voxel_logits
from tools.run_sparseworld_region_ablations import perceive,render
from tools.experiment_future_response_distillation import paths_for
from tools.diagnose_teacher_alignment import tree_to
from tools.sparseworld_region_utils import write_json,sha256

CONFIG='config/opus/sparseworld_fast_motion_raw_logit_kl.py'

def main(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile(args.config)
    if args.smoke:cfg.fast_samples_train=2;cfg.fast_samples_val=2;cfg.fast_workers=0
    root=Path(cfg.fast_root+('_smoke' if args.smoke else ''));root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path('out/future2current/feature_identity_2048x512/manifest.json').read_text())
    for split,n in [('train',cfg.fast_samples_train),('val',cfg.fast_samples_val)]:
        manifest[split]['indices']=manifest[split]['indices'][:n];manifest[split]['scenes']=manifest[split]['scenes'][:n]
    assert sha256(cfg.student_checkpoint)==manifest['checkpoint_sha256']==cfg.student_checkpoint_sha256
    reuse_targets(cfg,manifest,root)
    protocol=dict(train=manifest['train'],val=manifest['val'],checkpoint_sha256=cfg.student_checkpoint_sha256,
                  teacher='FP32 frozen strict SparseWorld perception; source ego; five-frame windows ending at steps 2/4/6',
                  raw_readout='geometric group-distance/range filters; per-class max of original logits; no activations, score cutoff, compactness or closing',
                  horizons=HORIZONS,source_cache=manifest['source_cache'])
    metadata=root/'raw_cache_protocol.json'
    if metadata.exists():assert json.loads(metadata.read_text())==json.loads(json.dumps(protocol))
    write_json(metadata,protocol)
    instance=build(cfg,Config.fromfile(cfg.student_model_config)).eval().requires_grad_(False)
    probes=json.loads(Path('out/sparseworld_fast_motion_distillation/20261009_repaired_2048x512/preflight_validation.json').read_text())['records']
    probe_ids=set(r['index'] for r in probes);replays=[];stats={}
    for split in ('train','val'):
        images,selected=loaders(cfg,split);assert selected==manifest[split]
        sources=paths_for(Path(manifest['source_cache']),split,selected)
        directory=root/'teacher_logits'/split;directory.mkdir(parents=True,exist_ok=True)
        missing=[i for i,ix in enumerate(selected['indices']) if not (directory/f'{ix:06d}.npz').exists()]
        started=time.monotonic()
        for i,batch in zip(missing,image_loader(images.dataset,cfg.fast_workers,missing)):
            ix=selected['indices'][i];source=torch.load(sources[i],map_location='cpu')
            with np.load(cache_path(root,split,ix)) as d:targets={k:d[k] for k in ('index','scene','teacher','fast_0','fast_1','fast_2')}
            assert int(targets['index'])==source['index']==ix and str(targets['scene'])==source['scene']==selected['scenes'][i]
            values=dict(index=ix,scene=source['scene']);data=tree_to(batch,'cuda');imgs=data.pop('img');ego=source['ego'].cuda()
            with torch.no_grad():
                for j,h in enumerate(HORIZONS):
                    ids=targets[f'fast_{j}'];values[f'fast_{j}']=ids
                    if not len(ids):z=np.empty((0,17),np.float32);valid=np.empty(0,bool)
                    else:
                        im,me=window_for(data,imgs,h);p,l,_,_=perceive(instance,im,me,ego)
                        volume=raw_voxel_logits(instance,p,l)
                        z=volume[torch.as_tensor(ids,device='cuda',dtype=torch.long)].cpu().numpy()
                        valid=np.isfinite(z).all(-1)
                        if ix in probe_ids or (split=='val' and i in (0,len(sources)-1)) or args.smoke:
                            assert np.array_equal(render(instance,p,l),targets['teacher'][j]),'Frozen teacher native occupancy differs from original cache'
                            replays.append(dict(split=split,index=ix,horizon=h))
                        del im,me,p,l,volume
                    values[f'teacher_raw_{j}']=z;values[f'teacher_raw_valid_{j}']=valid
            path=directory/f'{ix:06d}.npz';tmp=path.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**values)
            tmp.replace(path)
            del data,imgs,batch,source
            if (i+1)%32==0 or i==missing[-1]:print(f'[raw teacher {split}] {i+1}/{len(sources)} {time.monotonic()-started:.1f}s',flush=True)
        # Audit all cache rows, including resumed entries; no missing values are
        # turned into zero logits or a fabricated free distribution.
        totals=dict(fast_voxels=0,supported_voxels=0,negative_elements=0,finite_elements=0);chunks=[]
        for ix,scene in zip(selected['indices'],selected['scenes']):
            with np.load(directory/f'{ix:06d}.npz') as d:
                assert int(d['index'])==ix and str(d['scene'])==scene
                for j in range(3):
                    z=d[f'teacher_raw_{j}'];v=d[f'teacher_raw_valid_{j}']
                    assert z.shape==(len(d[f'fast_{j}']),17) and np.array_equal(v,np.isfinite(z).all(-1))
                    assert np.isneginf(z[~v]).all()
                    totals['fast_voxels']+=len(z);totals['supported_voxels']+=int(v.sum())
                    totals['negative_elements']+=int((z[v]<0).sum());totals['finite_elements']+=z[v].size
                    if v.any():chunks.append(z[v].reshape(-1))
        flat=np.concatenate(chunks);totals['logit_quantiles']=np.quantile(flat,[0,.01,.5,.99,1]).tolist();stats[split]=totals
        del images
    write_json(root/'raw_cache_audit.json',dict(status='complete',teacher_native_replays=replays,statistics=stats,torch_version=torch.__version__))
    print('raw teacher cache complete',stats,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CONFIG);p.add_argument('--smoke',action='store_true');main(p.parse_args())
