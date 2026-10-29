"""Full train/val boundary-Gaussian census; deterministic and scene-resumable.

python3.10 tools/stat_boundary_gaussians.py --action pilot
python3.10 tools/stat_boundary_gaussians.py --action run --workers 3
python3.10 tools/stat_boundary_gaussians.py --action report
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools.boundary_gaussian_stats import SCOPES,COHORTS,OUTCOMES,extension,gaussian_counts,rates
from tools.boundary_gaussian_stats.data import EvaluationFrames,FeatureCache

RUN=ROOT/'work_dirs/boundary_gaussian_full'
CONFIG=ROOT/'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py'
CHECKPOINT=ROOT/'ckpts/opus_gaussian_8f_tiny.pth'


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): digest.update(block)
    return digest.hexdigest()


def initialize():
    cfg=Config.fromfile(CONFIG)
    files=[CONFIG,ROOT/'tools/stat_boundary_gaussians.py',
        *sorted((ROOT/'tools/boundary_gaussian_stats').glob('*.py')),
        ROOT/'tools/boundary_gaussian_stats/kernels.cu',
        ROOT/'model/head/gaussian_opus_head.py',
        ROOT/'model/head/opus_v1_gaussian_trainable_bias_head.py',
        ROOT/'model/head/localagg/local_aggregate/__init__.py',
        ROOT/'model/head/localagg/src/forward.cu',
        ROOT/'model/head/localagg/src/auxiliary.h',
        ROOT/'dataset/opus_temporal_history.py',ROOT/'dataset/transform_3d.py',ROOT/'dataset/dataset.py',
        ROOT/'model/segmentor/opus_temporal_segmentor.py']
    hashes={str(f.relative_to(ROOT)):sha(f) for f in files}
    availability=read(RUN/'availability.json')
    assert all(sum(availability[s]['missing_counts'].values())==0 for s in ('train','val'))
    protocol=dict(config=str(CONFIG.relative_to(ROOT)),
        requested_config='config/occ3d_opus_gaussian/opusv1_gaussian_softplus_learned_bias_tiny_8f.py',
        config_resolution='Requested directory absent; same-basename full-dataset 8-frame tiny config; strict state_dict load required',
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=sha(CHECKPOINT),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256={s:sha(cfg[s+'_dataset_config'].imageset) for s in ('train','val')},
        expected={s:{k:availability[s][k] for k in ('frames','scenes')} for s in ('train','val')},
        seed=928, eval_mode=True,training_augmentation=False,
        support='Native clipped integer cube with ceil(max(scale)*3/.4) radius and positive FP32 opacity*exp(power); sensitivity: power>=-4.5 within that cube',
        gaussian_denominator='Every decoder-predicted final-stage Gaussian, every keyframe; exclude appended fixed empty Gaussian',
        scopes=list(SCOPES),cohorts=list(COHORTS),outcomes=list(OUTCOMES),
        boundary_definitions=['any >=2 of 18 GT classes (primary; includes empty)',
                              '>=2 of 17 foreground GT classes','both foreground and empty'],
        invalid_gt='Exclude labels outside 0..17',
        attribution='Primary: deduplicated union of boundary-G support, evaluate unchanged final model prediction; secondary: largest individual contribution to final foreground-winning logit, ties min Gaussian index',
        owner_target_class='argmax of 17 foreground rendered logits, including on predicted-empty voxels; no GT class used for choosing owner',
        aggregation='Sum counts over keyframes, then divide; per-scene, train, val, all. Scene macro ratios reported separately',
        cache='Original six-image transform and backbone/FPN chunks; reuse deterministic historical features; test parity against original 48-image inference',
        source_sha256=hashes)
    path=RUN/'protocol.json'
    if path.exists():
        old=read(path)
        assert old==protocol,'Protocol/source changed. Preserve prior results and use a fresh experiment directory.'
    else:
        write(path,protocol)
        cfg.dump(RUN/'resolved_config.py')
        import shutil
        for f in files:
            dest=RUN/'sources'/f.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    return cfg,protocol


def build(cfg):
    from visualize import load_model
    from mmengine.runner import set_random_seed
    set_random_seed(928);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=cfg.get('cudnn_benchmark',False)
    network=load_model(cfg,CHECKPOINT,'cuda:0')
    extension()
    return network


def to_cuda(batch):
    return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def forward(network,batch):
    return network(imgs=batch['img'],metas={k:v for k,v in batch.items() if k!='img'})


def diagnostic(network,out,batch,subset=None):
    head=network.head
    means,alpha,sem,scales,inverse=(v[0,:-1] for v in head.prepare_gaussian_args(out['gaussian']))
    inverse=inverse.flatten(1)[:,[0,4,8,1,5,2]]
    if subset is not None:
        means,alpha,sem,scales,inverse=[t[subset] for t in (means,alpha,sem,scales,inverse)]
    agg=head.aggregator
    labels=batch['occ_label'].flatten().to(torch.int32).contiguous()
    xyz=out['sampled_xyz'][0].contiguous()
    torch.testing.assert_close(xyz,batch['occ_xyz'].reshape(-1,3),rtol=0,atol=0)
    # The kernel uses voxel-grid flat ids. Guard the native xyz ordering explicitly.
    cells=((xyz-agg.pc_min)/agg.grid_size).to(torch.int64)
    flat=(cells[:,0]*agg.W+cells[:,1])*agg.D+cells[:,2]
    if not torch.equal(flat,torch.arange(len(labels),device=labels.device)):
        raise ValueError('Unexpected voxel ordering')
    winner=out['pred_occ'][-1][0,:17].argmax(0).to(torch.int32).contiguous()
    values=[means.contiguous(),scales.contiguous(),inverse.contiguous(),alpha.flatten().contiguous(),
            sem[:,:17].contiguous(),xyz,labels,batch['occ_cam_mask'].flatten().bool().contiguous(),
            winner,out['final_occ'].flatten().to(torch.int32).contiguous()]
    params=[agg.H,agg.W,agg.D,agg.grid_size,float(agg.scale_multiplier),*agg.pc_min[0].cpu().tolist()]
    result=extension().diagnose(*values,*params)
    return result,values,params


def pilot():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from tools.boundary_gaussian_stats.reference import reference
    cfg=Config.fromfile(CONFIG);network=build(cfg);audits=[]
    for split in ('val','train'):
        options=deepcopy(cfg.val_dataset_config);options.imageset=cfg[split+'_dataset_config'].imageset
        options.return_keys=list(options.return_keys)+['history_frame_indices']
        native=OPENOCC_DATASET.build(options)
        fast=EvaluationFrames(cfg,split)
        cache=FeatureCache(network,fast)
        for index in (0,7):
            a=custom_collate_fn_temporal([native[index]])
            b=custom_collate_fn_temporal([fast[index]])
            for key in a:
                if key=='img':
                    torch.testing.assert_close(a[key][:,:6],b[key],rtol=0,atol=0)
                else:
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            a=to_cuda(a);b=to_cuda(b)
            with torch.inference_mode():
                network.extract_img_feat=cache.original
                t=time.perf_counter();original=forward(network,a);torch.cuda.synchronize()
                native_seconds=time.perf_counter()-t
                network.extract_img_feat=cache.extract
                t=time.perf_counter();out=forward(network,b);torch.cuda.synchronize()
                cached_seconds=time.perf_counter()-t
                torch.testing.assert_close(out['pred_occ'][-1],original['pred_occ'][-1],rtol=0,atol=0)
                for key in out['gaussian']._fields:
                    torch.testing.assert_close(getattr(out['gaussian'],key),getattr(original['gaussian'],key),rtol=0,atol=0)
                assert torch.equal(out['final_occ'],original['final_occ'])
                t=time.perf_counter();result,_,_=diagnostic(network,out,b);torch.cuda.synchronize()
                stats_seconds=time.perf_counter()-t
                gcounts=gaussian_counts(result[0]);counts=result[4].cpu().numpy()
                np.testing.assert_array_equal(counts[:,0],counts[:,4]+counts[:,7]+counts[:,8])
                assert (counts[:,1]>=counts[:,4]).all()
                # Independent native-support/ownership audit on a fixed Gaussian subset.
                result_small,inputs,params=diagnostic(network,out,b,torch.arange(0,64,device='cuda'))
                expected=reference(*[v.cpu().numpy() for v in inputs],tuple(params[:3]),params[3],params[4],params[5:])
                for j in (0,1,2,4):
                    np.testing.assert_array_equal(result_small[j].cpu().numpy(),expected[j])
                # Cross-check the sparse influence expression against the actual native renderer.
                # Full Gaussian output statistics are untouched; this is only a probe.
                from tools.center_support_common import kernel_weights
                args=tuple(v[:,:-1].contiguous() for v in network.head.prepare_gaussian_args(out['gaussian']))
                means,alpha,sem,scales,inv=args
                probe=torch.arange(0,640000,20000,device='cuda')
                ref=torch.zeros((len(probe),17),device='cuda')
                for start in range(0,means.shape[1],4096):
                    weight=kernel_weights(out['sampled_xyz'][0,probe],means[0,start:start+4096],
                        alpha[0,start:start+4096],scales[0,start:start+4096],inv[0,start:start+4096],
                        network.head.aggregator.pc_min[0],.4,3.)
                    ref+=weight @ sem[0,start:start+4096,:17]
                torch.testing.assert_close(ref,out['pred_occ'][-1][0,:17,probe].T,atol=3e-5,rtol=3e-5)
            row=dict(split=split,index=index,scene=native.keyframes[index][0],
                history=b['history_frame_indices'][0].cpu().tolist(),
                image_and_projection_bitwise_equal=True,gaussians_and_logits_bitwise_equal=True,
                diagnostic_numpy_reference_equal=True,native_logit_max_probe_error=float((ref-out['pred_occ'][-1][0,:17,probe].T).abs().max()),
                native_forward_seconds=native_seconds,cached_forward_seconds=cached_seconds,
                diagnostic_seconds=stats_seconds,gaussian_counts=gcounts,error_counts=counts.tolist())
            audits.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('gaussian_counts','error_counts')}),flush=True)
            del a,b,out,original,result,result_small,inputs,expected,args,ref
        cache.close();del fast,native,cache
    initialize()
    write(RUN/'pilot.json',dict(status='passed',frames=4,audits=audits,empty_bias=float(network.head.empty_bias),checkpoint_sha256=sha(CHECKPOINT)))


def run(workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader,Subset
    cfg,protocol=initialize()
    p=read(RUN/'pilot.json');assert p['status']=='passed' and p['checkpoint_sha256']==protocol['checkpoint_sha256']
    network=build(cfg)
    total_start=time.perf_counter();completed_frames=0
    for split in ('val','train'):
        dataset=EvaluationFrames(cfg,split)
        expected_scenes={}
        for i,(scene,frame) in enumerate(dataset.keyframes):expected_scenes.setdefault(scene,[]).append(i)
        indices=[]
        for scene,ids in expected_scenes.items():
            path=RUN/'scenes'/split/(scene+'.json')
            if path.exists():
                old=read(path)
                assert old['indices']==ids and old['checkpoint_sha256']==protocol['checkpoint_sha256']
            else:indices.extend(ids)
        loader=DataLoader(Subset(dataset,indices),batch_size=1,shuffle=False,num_workers=workers,
            collate_fn=custom_collate_fn_temporal,pin_memory=True,
            **(dict(prefetch_factor=2,persistent_workers=True) if workers else {}))
        cache=FeatureCache(network,dataset)
        active=None;rows=[];counts=None;gcounts=None
        for index,batch in zip(indices,loader):
            begin=time.perf_counter();scene,frame=dataset.keyframes[index]
            if scene!=active:
                assert active is None,'Prior scene was not committed'
                active=scene;rows=[];counts=np.zeros((4,9,5),np.int64);gcounts=None
            batch=to_cuda(batch)
            with torch.inference_mode():
                output=forward(network,batch)
                if not torch.isfinite(output['pred_occ'][-1]).all():raise ValueError('Nonfinite prediction')
                result,_,_=diagnostic(network,output,batch)
                frame_g=gaussian_counts(result[0]);frame_c=result[4].cpu().numpy()
                assert len(output['gaussian'].means[0])==76800
            np.testing.assert_array_equal(frame_c[:,0],frame_c[:,4]+frame_c[:,7]+frame_c[:,8])
            assert (frame_c[:,1]>=frame_c[:,4]).all()
            counts+=frame_c
            if gcounts is None:gcounts=deepcopy(frame_g)
            else:
                for s in range(4):
                    for k,v in frame_g[s].items():
                        if isinstance(v,list):gcounts[s][k]=(np.array(gcounts[s][k])+v).tolist()
                        else:gcounts[s][k]+=v
            info=dataset.base.scene_infos[scene][frame]
            rows.append(dict(index=index,scene_frame=int(frame),sample_token=info['token'],
                gaussian_counts=frame_g,error_counts=frame_c.tolist(),seconds=time.perf_counter()-begin))
            completed_frames+=1
            if len(rows)==len(expected_scenes[scene]):
                scene_name=Path(info['occ_path']).parts[-3]
                record=dict(status='complete',split=split,scene_token=scene,scene_name=scene_name,
                    indices=expected_scenes[scene],frames=len(rows),gaussian_counts=gcounts,
                    error_counts=counts.tolist(),records=rows,checkpoint_sha256=protocol['checkpoint_sha256'])
                write(RUN/'scenes'/split/(scene+'.json'),record)
                elapsed=time.perf_counter()-total_start
                write(RUN/'progress.json',dict(status='running',split=split,last_scene=scene_name,
                    frames_this_process=completed_frames,seconds_this_process=elapsed,
                    frames_per_second=completed_frames/elapsed,updated_at=datetime.now().astimezone().isoformat()))
                print(f'{split} {scene_name}: {len(rows)} frames; process={completed_frames}, '
                      f'{completed_frames/elapsed:.2f} frames/s; native boundary={gcounts[0]["boundary_any"]/gcounts[0]["predicted"]:.4%}',flush=True)
                active=None
            del output,batch,result
        assert active is None
        cache.close();del cache,dataset,loader
    write(RUN/'execution_complete.json',dict(status='complete',checkpoint_sha256=protocol['checkpoint_sha256'],
        elapsed_seconds=time.perf_counter()-total_start,frames_this_process=completed_frames))
    report()


def report():
    from tools.boundary_gaussian_stats.report import generate
    generate(RUN,ROOT/'analysis/boundary_gaussian_full_results.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['pilot','run','report'],required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='pilot':pilot()
    elif args.action=='run':run(args.workers)
    else:report()
