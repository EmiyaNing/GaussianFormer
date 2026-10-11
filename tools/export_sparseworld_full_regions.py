#!/usr/bin/env python3
"""All eligible nuScenes keyframes: paired native F/P grids, exact pilot replay."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mmengine import Config
from torch.utils.data import DataLoader,Subset
from dataset.utils import custom_collate_fn_temporal
from tools.diagnose_teacher_alignment import tree_to,seed_all
from tools.experiment_future_response_distillation import build_student
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.sparseworld_region_utils import HORIZONS,sha256,write_json
from tools.sparseworld_full_regions_common import ROOT,make_dataset,record_for,future_window,WindowFeatureCache,encode_window

OLD=Path('out/sparseworld_region_comparison/20261008_native512')

def predict(model,cache,data,imgs,rec):
    metas=dict(data,projection_mat=data['projection_mat'][:,:30],image_wh=data['image_wh'][:,:30])
    ego=model.plan_head(data['temporal_ego_states'].float().reshape(1,-1)).unsqueeze(1)
    state=encode_window(model,imgs[:,:30],metas,ego,cache.get(imgs[:,:30],rec['frame_tokens'][:5]))
    out=forecast_with_position_gradient(model,state['representation'][-1],state['query_stamps'],ego,position_gradient=False)
    shape=(3,200,200,16)
    result=dict(F=np.full(shape,17,np.uint8),P=np.full(shape,17,np.uint8),GT=np.full(shape,255,np.uint8),
                camera=np.zeros(shape,bool),lidar=np.zeros(shape,bool),
                F_projection=np.zeros((3,30,4,4),np.float32),P_projection=np.zeros((3,30,4,4),np.float32),
                F_wh=np.zeros((3,30,2),np.float32),P_wh=np.zeros((3,30,2),np.float32),
                current_GT=data['occ_label'][0].cpu().numpy().astype(np.uint8).reshape(200,200,16),
                current_camera=data['occ_cam_mask'][0].cpu().numpy().reshape(200,200,16),
                current_lidar=data['occ_lidar_mask'][0].cpu().numpy().reshape(200,200,16),
                available=np.asarray([t['available'] for t in rec['targets']]))
    for j,h in enumerate(HORIZONS):
        if not rec['targets'][j]['available']:continue
        result['F'][j]=model.head.rasterizer(out['points'][h-1].float(),out['logits'][h-1].float(),48).view(200,200,16).cpu().numpy()
        result['GT'][j]=data['future_occ_labels'][0,h-1].cpu().numpy().astype(np.uint8)
        result['camera'][j]=data['future_occ_cam_masks'][0,h-1].cpu().numpy()
        result['lidar'][j]=data['future_occ_lidar_masks'][0,h-1].cpu().numpy()
        selected,pm,frames=future_window(data,imgs,h)
        tokens=[rec['frame_tokens'][f] for f in frames]
        pstate=encode_window(model,selected,pm,ego,cache.get(selected,tokens))
        pstate.update(model.head(**pstate))
        result['P'][j]=pstate['final_occ_grid'][0].cpu().numpy().astype(np.uint8)
        transform=data['future_ego_to_current'][:,h-1]
        result['F_projection'][j]=(metas['projection_mat']@transform[:,None])[0].cpu().numpy()
        result['P_projection'][j]=pm['projection_mat'][0].cpu().numpy()
        result['F_wh'][j]=metas['image_wh'][0].cpu().numpy();result['P_wh'][j]=pm['image_wh'][0].cpu().numpy()
        del pstate
    return result


def check_old(rec,result):
    path=OLD/'pairs'/(rec['token']+'.npz')
    if not path.exists():return False
    with np.load(path) as d:
        for key in ('F','P','GT','camera','lidar'):
            if not np.array_equal(result[key],d[key]):
                raise AssertionError(f'Exact old replay failed: {rec["token"]} {key} differing={np.count_nonzero(result[key]!=d[key])}')
        for key in ('F_projection','P_projection','F_wh','P_wh'):
            np.testing.assert_allclose(result[key],d[key],rtol=0,atol=0)
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--no-feature-cache',action='store_true')
    parser.add_argument('--resume-compatible-code',action='store_true',help='Record metadata-only implementation update; scientific protocol must match')
    args=parser.parse_args();root=args.root/('smoke' if args.pilot else '')
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py')
    datasets={};specs={};records=[];splits={}
    import mmengine
    for split in ('val','train'):
        ds,spec=make_dataset(cfg,split);datasets[split]=ds;specs[split]=spec
        raw=mmengine.load(spec.imageset)['metadata'];nraw=len(raw);del raw
        local=[record_for(ds,i,split) for i in range(len(ds))]
        splits[split]=dict(raw_keyframes=nraw,sources=len(ds),scenes=len({r['scene'] for r in local}),
            per_horizon=[sum(r['targets'][j]['available'] for r in local) for j in range(3)],
            excluded_per_horizon=[nraw-sum(r['targets'][j]['available'] for r in local) for j in range(3)],
            exclusion_definition='4 real historical keyframes, current ego state, same-scene future GT/RGB at requested horizon; no random subsampling or official extra boundary filter')
        records.extend(local)
    assert not {r['scene'] for r in records if r['split']=='train'}&{r['scene'] for r in records if r['split']=='val'}
    old=json.loads((OLD/'manifest.json').read_text())['samples'];oldtokens={r['token'] for r in old}
    assert oldtokens<={r['token'] for r in records if r['split']=='val'}
    if args.pilot:
        chosen=sorted(old,key=lambda r:r['token'])[:12];tokens={r['token'] for r in chosen}
        for split in ('val','train'):
            for n in (2,4,6):
                candidate=next(r for r in records if r['split']==split and sum(t['available'] for t in r['targets'])==n//2)
                tokens.add(candidate['token'])
        records=[r for r in records if r['token'] in tokens]
    root.mkdir(parents=True,exist_ok=True);(root/'pairs').mkdir(exist_ok=True)
    protocol=dict(version=1,scope='full eligible nuScenes train and val, per-horizon availability',
        checkpoint=cfg.student_checkpoint,checkpoint_sha256=sha256(cfg.student_checkpoint),
        strict_config=cfg.student_model_config,strict_config_sha256=sha256(cfg.student_model_config),
        input_config='config/opus/future2current_feature_identity.py',dataset_specs=specs,
        horizons=list(HORIZONS),grid_shape=[200,200,16],grid_size=.4,pc_range=[-40,-40,-1,40,40,5.4],
        precision='FP32; TF32 disabled; frozen eval; batch one',mask='none',free_label=17,
        teacher_ego='same current plan-head ego; true five-frame window ending at target',
        historical_frames=5,feature_cache=not args.no_feature_cache,
        missing_targets='GT=255, available=false; excluded from all metrics; padded RGB never used by a teacher',
        pilot=bool(args.pilot),splits=splits,
        code_sha256={p:sha256(p) for p in ['tools/export_sparseworld_full_regions.py','tools/sparseworld_full_regions_common.py']})
    assert protocol['checkpoint_sha256']==cfg.student_checkpoint_sha256
    protocol=json.loads(json.dumps(protocol))  # JSON turns config tuples into lists.
    if (root/'protocol.json').exists():
        previous=json.loads((root/'protocol.json').read_text())
        if previous!=protocol:
            assert {k:v for k,v in previous.items() if k!='code_sha256'}=={k:v for k,v in protocol.items() if k!='code_sha256'},'Scientific protocol changed; use a new output root'
            assert args.resume_compatible_code,'Implementation changed; explicit compatibility record required'
            assert previous['code_sha256']['tools/sparseworld_full_regions_common.py']==protocol['code_sha256']['tools/sparseworld_full_regions_common.py'],'Numerical helpers changed'
            import ast
            archived=root/'code_at_launch'/'export_sparseworld_full_regions.py'
            assert archived.exists() and sha256(archived)==previous['code_sha256']['tools/export_sparseworld_full_regions.py']
            oldtree=ast.parse(archived.read_text());newtree=ast.parse(Path(__file__).read_text())
            for name in ('predict','check_old'):
                oldfn=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                newfn=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
                assert ast.dump(oldfn,include_attributes=False)==ast.dump(newfn,include_attributes=False),'Prediction/replay implementation changed'
            write_json(root/'implementation_resume.json',dict(status='passed',original=previous['code_sha256'],resumed=protocol['code_sha256'],
                reason='Canonical tuple/list comparison for resume; scientific protocol and prediction/replay AST unchanged; original protocol hash retained'))
    else:write_json(root/'protocol.json',protocol)
    write_json(root/'manifest.json',dict(samples=records,splits=splits,old_validation_tokens=sorted(oldtokens),
                                       paired_targets=sum(t['available'] for r in records for t in r['targets'])))
    write_json(root/'scene_split.json',dict(discovery=sorted({r['scene'] for r in records if r['split']=='train'}),
        holdout=sorted({r['scene'] for r in records if r['split']=='val'}),
        meaning='Full training scenes for explanatory fitting; all validation scenes for confirmation. Training accuracy is in-sample; validation scenes were used by earlier pilots.'))
    print('[full coverage]',json.dumps(splits),flush=True)
    if args.prepare_only:return
    torch.set_num_threads(4);seed_all(20261010)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=build_student(cfg,Config.fromfile(cfg.student_model_config));cache=WindowFeatureCache(model,enabled=not args.no_feature_cache)
    digest=sha256(root/'protocol.json');done=0;replays=[];started=time.monotonic()
    timing=root/'timing.jsonl'
    write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records),completed=0))
    with torch.no_grad():
        for split in ('val','train'):
            local=[r for r in records if r['split']==split]
            pending=[r for r in local if not (root/'pairs'/(r['token']+'.npz')).exists()]
            loader=DataLoader(Subset(datasets[split],[r['index'] for r in pending]),batch_size=1,num_workers=args.workers,
                collate_fn=custom_collate_fn_temporal,pin_memory=True,prefetch_factor=2 if args.workers else None)
            done+=len(local)-len(pending)
            for rec,batch in zip(pending,loader):
                t=time.monotonic();assert batch['sample_idx'][0]==rec['token']
                data=tree_to(batch,'cuda');imgs=data.pop('img');result=predict(model,cache,data,imgs,rec)
                replay=check_old(rec,result)
                if replay:replays.append(rec['token'])
                dest=root/'pairs'/(rec['token']+'.npz');tmp=dest.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**result,protocol_sha256=np.asarray(digest),token=np.asarray(rec['token']),split=np.asarray(split))
                tmp.replace(dest);done+=1
                with timing.open('a') as f:f.write(json.dumps(dict(token=rec['token'],split=split,seconds=time.monotonic()-t,old_replay=replay))+'\n')
                if done%64==0 or done==len(records):
                    write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records),completed=done,elapsed_seconds=time.monotonic()-started,cache_hits=cache.hits,cache_misses=cache.misses))
                    print(f'[full export] {done}/{len(records)} {split} elapsed={time.monotonic()-started:.1f}s backbone hits/misses={cache.hits}/{cache.misses}',flush=True)
                del data,imgs,result,batch
            write_json(root/(split+'_export_complete.json'),dict(status='complete',samples=len(local)))
    for rec in records:
        with np.load(root/'pairs'/(rec['token']+'.npz')) as d:
            assert str(d['protocol_sha256'])==digest and str(d['token'])==rec['token']
            assert np.array_equal(d['available'],[t['available'] for t in rec['targets']])
            if check_old(rec,d) and rec['token'] not in replays:replays.append(rec['token'])
    expected=len(oldtokens & {r['token'] for r in records});assert len(replays)==expected
    write_json(root/'replay_audit.json',dict(status='complete',exact_old_samples=expected,tokens=sorted(replays),
        checked=['F','P','GT','camera','lidar','F_projection','P_projection','F_wh','P_wh']))
    write_json(root/'status.json',dict(status='complete',samples=len(records),elapsed_seconds=time.monotonic()-started,
        old_exact_replays=expected,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30))

if __name__=='__main__':main()
