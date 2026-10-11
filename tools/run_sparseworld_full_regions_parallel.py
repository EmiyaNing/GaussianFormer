#!/usr/bin/env python3
"""Disjoint scene workers reuse the unchanged numerical exporter on one GPU."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sparseworld_region_utils import sha256,write_json
from tools.sparseworld_full_regions_common import ROOT


def worker(args,manifest):
    import cv2,zipfile,numpy as np,torch
    from mmengine import Config
    from torch.utils.data import DataLoader,Subset
    from dataset.utils import custom_collate_fn_temporal
    from tools.diagnose_teacher_alignment import tree_to,seed_all
    from tools.experiment_future_response_distillation import build_student
    from tools.sparseworld_full_regions_common import make_dataset,WindowFeatureCache
    import tools.export_sparseworld_full_regions as exporter
    from tools.run_sparseworld_full_regions_fastio import collect_lineage,save_lineage,fast_zip_init
    root=args.root;scenes=sorted({r['scene'] for r in manifest['samples']});assigned=set(scenes[args.rank::args.jobs])
    records=[r for r in manifest['samples'] if r['scene'] in assigned];dest=root/f'shard_{args.rank}_status.json'
    write_json(dest,dict(status='initializing',pid=os.getpid(),total=len(records)))
    # Separate CPU scheduling only; frozen batch-one numerical functions unchanged.
    if args.jobs==2 and len(os.sched_getaffinity(0))>=16:os.sched_setaffinity(0,set(range(6+5*args.rank,11+5*args.rank)))
    torch.set_num_threads(4);cv2.setNumThreads(1);zipfile.ZipFile.__init__=fast_zip_init
    seed_all(20261010);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py');protocol=json.loads((root/'protocol.json').read_text())
    assert sha256(cfg.student_checkpoint)==protocol['checkpoint_sha256']
    assert sha256(cfg.student_model_config)==protocol['strict_config_sha256']
    assert sha256('tools/sparseworld_full_regions_common.py')==protocol['code_sha256']['tools/sparseworld_full_regions_common.py']
    datasets={split:make_dataset(cfg,split)[0] for split in ('val','train')}
    if args.optimized_images:
        from tools.sparseworld_full_image_io import optimize_image_loaders,install_decoded_image_cache
        audit=json.loads((root/'image_input_parity.json').read_text());assert audit['status']=='passed'
        assert audit['code_sha256']['tools/sparseworld_full_image_io.py']==sha256('tools/sparseworld_full_image_io.py')
        for ds in datasets.values():optimize_image_loaders(ds)
        install_decoded_image_cache()
    model=build_student(cfg,Config.fromfile(cfg.student_model_config));cache=WindowFeatureCache(model)
    forecast=exporter.forecast_with_position_gradient;raw={}
    def capture(instance,encoded,stamps,ego,**kwargs):
        out=forecast(instance,encoded,stamps,ego,**kwargs);raw.update(collect_lineage(instance,encoded,stamps,out));return out
    exporter.forecast_with_position_gradient=capture
    digest=sha256(root/'protocol.json');started=time.monotonic();done=0
    with torch.no_grad():
        for split in ('val','train'):
            local=[r for r in records if r['split']==split]
            pending=[r for r in local if not (root/'pairs'/(r['token']+'.npz')).exists()]
            done+=len(local)-len(pending)
            loader=DataLoader(Subset(datasets[split],[r['index'] for r in pending]),batch_size=1,num_workers=args.workers,
                collate_fn=custom_collate_fn_temporal,pin_memory=True,prefetch_factor=2 if args.workers else None)
            for rec,batch in zip(pending,loader):
                tick=time.monotonic();assert batch['sample_idx'][0]==rec['token'];raw.clear()
                data=tree_to(batch,'cuda');imgs=data.pop('img');result=exporter.predict(model,cache,data,imgs,rec)
                replay=exporter.check_old(rec,result);save_lineage(root,rec,raw)
                pp=root/'pairs'/(rec['token']+'.npz');tmp=pp.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**result,protocol_sha256=np.asarray(digest),token=np.asarray(rec['token']),split=np.asarray(split));tmp.replace(pp)
                with (root/f'shard_{args.rank}_timing.jsonl').open('a') as f:f.write(json.dumps(dict(token=rec['token'],split=split,seconds=time.monotonic()-tick,old_replay=replay))+'\n')
                done+=1
                if done%64==0 or done==len(records):
                    write_json(dest,dict(status='running',pid=os.getpid(),completed=done,total=len(records),elapsed_seconds=time.monotonic()-started))
                    print(f'[shard {args.rank}] {done}/{len(records)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                del data,imgs,result,batch;raw.clear()
    write_json(dest,dict(status='complete',pid=os.getpid(),completed=done,total=len(records),elapsed_seconds=time.monotonic()-started,
                         peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--rank',type=int,default=0);p.add_argument('--worker',action='store_true')
    p.add_argument('--optimized-images',action='store_true',help='Lossless uint8 decode cache and early-cast elimination; requires exact input parity audit')
    args=p.parse_args();root=args.root;manifest=json.loads((root/'manifest.json').read_text())
    if args.worker:return worker(args,manifest)
    previous=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    if previous.get('status')=='running':
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('The serial exporter is still running; stop it before launching disjoint scene workers')
    protocol=json.loads((root/'protocol.json').read_text());write_json(root/'parallel_execution.json',dict(jobs=args.jobs,
        allocation='Sorted full scene names, rank-strided assignment; each scene and output token has exactly one writer',
        previous_status=previous,code_sha256=sha256(__file__),numerical_exporter_sha256=sha256('tools/export_sparseworld_full_regions.py'),
        data_loader_workers_per_process=args.workers,optimized_images=args.optimized_images,
        inference='Unchanged exporter.predict/check_old, frozen FP32 batch one; unchanged read-only query collector; lossless ZIP level 1'))
    processes=[];logs=[];started=time.monotonic()
    for rank in range(args.jobs):
        log=(root/f'shard_{rank}.log').open('a');logs.append(log)
        command=[sys.executable,__file__,'--root',str(root),'--jobs',str(args.jobs),'--workers',str(args.workers),'--rank',str(rank),'--worker']
        if args.optimized_images:command.append('--optimized-images')
        processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
    while True:
        failures=[(i,p.returncode) for i,p in enumerate(processes) if p.poll() not in (None,0)]
        if failures:
            for process in processes:
                if process.poll() is None:process.terminate()
            write_json(root/'status.json',dict(status='failed',shard_failures=failures))
            raise RuntimeError(f'Shard failures: {failures}; inspect shard logs')
        done=sum(1 for r in manifest['samples'] if (root/'pairs'/(r['token']+'.npz')).exists())
        write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(manifest['samples']),completed=done,
                   elapsed_seconds=time.monotonic()-started,execution='two disjoint scene processes' if args.jobs==2 else 'scene processes'))
        print(f'[parallel export] {done}/{len(manifest["samples"])} elapsed={time.monotonic()-started:.1f}s',flush=True)
        if all(process.poll()==0 for process in processes):break
        time.sleep(50)
    for log in logs:log.close()
    import numpy as np
    from tools.export_sparseworld_full_regions import check_old
    digest=sha256(root/'protocol.json');replays=[]
    for rec in manifest['samples']:
        with np.load(root/'pairs'/(rec['token']+'.npz')) as pair:
            assert str(pair['token'])==rec['token'] and str(pair['protocol_sha256'])==digest
            assert np.array_equal(pair['available'],[t['available'] for t in rec['targets']])
            if check_old(rec,pair):replays.append(rec['token'])
        assert (root/'query_lineage'/(rec['token']+'.npz')).exists()
    expected=len(set(manifest['old_validation_tokens'])&{r['token'] for r in manifest['samples']})
    assert len(replays)==expected
    write_json(root/'replay_audit.json',dict(status='complete',exact_old_samples=expected,tokens=sorted(replays),
        checked=['F','P','GT','camera','lidar','F_projection','P_projection','F_wh','P_wh']))
    for split in ('val','train'):
        write_json(root/(split+'_export_complete.json'),dict(status='complete',samples=sum(r['split']==split for r in manifest['samples'])))
    write_json(root/'status.json',dict(status='complete',samples=len(manifest['samples']),elapsed_seconds=time.monotonic()-started,old_exact_replays=expected,
        peak_cuda_gib_per_worker=[json.loads((root/f'shard_{i}_status.json').read_text())['peak_cuda_gib'] for i in range(args.jobs)]))
    print('[parallel export] complete; exact previous-case replay passed',flush=True)


if __name__=='__main__':main()
