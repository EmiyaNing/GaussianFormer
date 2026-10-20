"""Official RayIoU and complete GPU-resident inference timing for frozen policies."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.empty_decision_optimization import (RUN, CONFIG, CHECKPOINT, BAND, PRIMARY_BUDGET,
    read, write_json, file_sha256, seed_all, build_loader, build_network, to_cuda,
    candidate_features, support_field)
from tools.fit_empty_decision_optimization import Predictor
from tools.eval_empty_decision_optimization import score_for
from tools.eval_center_support import import_file, official_pose_infos, RayCounts


def official_setup():
    import mmengine
    official = Path('/home/emiya/data_disk/OPUS')
    sources = ['loaders/ray_metrics.py','loaders/ego_pose_dataset.py','lib/dvr/dvr.cpp','lib/dvr/dvr.cu']
    hashes = {p:file_sha256(official/p) for p in sources}
    os.environ.setdefault('TORCH_EXTENSIONS_DIR',str(ROOT/'work_dirs/center_support_contribution1/torch_extensions'))
    os.environ.setdefault('MAX_JOBS','2')
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST','8.9')
    previous = os.getcwd()
    try:
        os.chdir(official)
        ray = import_file('empty_decision_official_ray',official/'loaders/ray_metrics.py')
        poses = import_file('empty_decision_official_poses',official/'loaders/ego_pose_dataset.py')
    finally:
        os.chdir(previous)
    full_path = ROOT/'data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl'
    full = mmengine.load(full_path)
    pose_dataset = poses.EgoPoseDataset(official_pose_infos(full['infos'],full['metadata']))
    pose_index = {info['token']:i for i,info in enumerate(pose_dataset.data_infos)}
    key = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()+str(full_path.stat().st_mtime_ns).encode()).hexdigest()[:16]
    cache = ROOT/'work_dirs/center_support_contribution1/ray_gt_cache'/key
    cache.mkdir(parents=True,exist_ok=True)
    cfg = __import__('mmengine').Config.fromfile(CONFIG)
    mini = mmengine.load(cfg.val_dataset_config.imageset)
    commit = subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    return ray,pose_dataset,pose_index,cache,mini,dict(root=str(official),commit=commit,sha256=hashes)


def ray_evaluate(limit=0):
    seed_all(928)
    validation = read(RUN/'validation/summary.json')
    frames = validation['frames'][:limit] if limit else validation['frames']
    ids = validation['primary_predictions']
    ray,poses,pose_index,cache,mini,provenance = official_setup()
    directions = torch.from_numpy(ray.generate_lidar_rays())
    counts = {k:RayCounts() for k in ids}
    first_pred = {k:[] for k in ids}
    first_gt = []
    parity = {k:False for k in ids}
    dest = RUN/('ray_smoke' if limit else 'ray')
    dest.mkdir(exist_ok=True)
    started = time.monotonic()
    with torch.no_grad():
        for n,frame in enumerate(frames):
            stem = ROOT/frame['path']
            meta = read(stem.with_suffix('.json'))
            token = mini['infos'][meta['scene']][meta['scene_frame']]['token']
            _,origins = poses[pose_index[token]]
            origins = origins[None]
            result_path = dest/f'frame_{frame["index"]:04d}.json'
            if result_path.exists() and n >= 8:
                saved = read(result_path)
                assert saved['selection_sha256'] == validation['selection_sha256']
                assert saved['official_source_sha256'] == provenance['sha256']
                for k,value in saved['counts'].items():
                    counts[k].gt += value['gt']
                    counts[k].pred += value['pred']
                    counts[k].tp += value['tp']
                    parity[k] = parity[k] or saved['official_parity'].get(k,False)
                continue
            with np.load(stem.with_suffix('.npz')) as z:
                gt = z['gt'].reshape(200,200,16)
                assert bool((gt < 18).all())
                gt_sha = hashlib.sha256(gt.astype(np.int64).tobytes()).hexdigest()
                cached = cache/(token+'.npz')
                if cached.exists():
                    with np.load(cached) as c:
                        gt_ray = c['rays']
                        assert np.array_equal(c['origins'],origins.numpy())
                        if 'gt_sha256' in c:
                            assert str(c['gt_sha256']) == gt_sha
                else:
                    gt_ray = ray.process_one_sample(gt.astype(np.int64),directions,origins)
                    np.savez_compressed(cached,rays=gt_ray,origins=origins.numpy(),gt_sha256=gt_sha)
                valid_rays = gt_ray[:,0].astype(np.int32) != 17
                trimmed_gt = gt_ray[valid_rays]
                if n < 8:
                    first_gt.append(trimmed_gt)
                saved = dict(selection_sha256=validation['selection_sha256'],official_source_sha256=provenance['sha256'],
                    counts={},official_parity={})
                for k in ids:
                    prediction = z[k].reshape(200,200,16).astype(np.int64)
                    pred_ray = ray.process_one_sample(prediction,directions,origins)[valid_rays]
                    current = RayCounts()
                    current.add(pred_ray,trimmed_gt)
                    counts[k].add(pred_ray,trimmed_gt)
                    saved['counts'][k] = dict(gt=current.gt.tolist(),pred=current.pred.tolist(),tp=current.tp.tolist())
                    if n < 8:
                        first_pred[k].append(pred_ray)
                        expected = np.asarray(ray.calc_metrics(first_pred[k],first_gt))
                        assert np.allclose(expected,counts[k].iou(),rtol=0,atol=0,equal_nan=True)
                        parity[k] = True
                    saved['official_parity'][k] = parity[k]
            write_json(result_path,saved)
            if n == 0 or (n+1)%32 == 0:
                print('official RayIoU',n+1,'/',len(frames),'models',len(ids),'elapsed',time.monotonic()-started,flush=True)
                write_json(RUN/'status.json',dict(status='running',stage='official_ray',frames=n+1,expected=len(frames)))
    results = {}
    for k,c in counts.items():
        iou = c.iou()
        results[k] = dict(RayIoU=float(np.nanmean(iou)*100),
            **{f'RayIoU@{r}':float(np.nanmean(iou[i])*100) for i,r in enumerate([1,2,4])},
            counts=dict(gt=c.gt.tolist(),predicted=c.pred.tolist(),true_positive=c.tp.tolist()),
            per_class=[[float(x*100) if np.isfinite(x) else None for x in row] for row in iou])
    if not limit:
        old = read(ROOT/'work_dirs/center_support_contribution1/train/a0_original_seed928/evaluation_epoch5.json')
        for key in ['gt','predicted','true_positive']:
            assert np.array_equal(results['B0']['counts'][key],old['ray_counts'][key]),key
    result = dict(status='complete',samples=len(frames),models=results,official_source=provenance,
        official_streaming_parity=parity,baseline_previous_counts_exact_parity=not bool(limit),
        elapsed_seconds=time.monotonic()-started,
        protocol='official OPUS rays; complete validation-scene ego origins; no camera mask; GT non-free rays; 1/2/4m',
        source_sha256=file_sha256(__file__))
    write_json(dest/'summary.json',result)
    return result


def benchmark(frames=8,repeats=5):
    seed_all(928)
    selection = read(RUN/'operating_points.json')
    validation = read(RUN/'validation/summary.json')
    points = {p['id']:p for p in selection['operating_points'] if p['id'] in validation['primary_predictions']}
    # P1_q05 has the same architecture as P1; no duplicate timing is necessary.
    points = {k:v for k,v in points.items() if v['family'] != 'P1_q05'}
    manifest = read(RUN/'tuning/summary.json')['models']
    predictors = {name:Predictor(ROOT/v['path']) for name,v in manifest.items() if name.endswith('seed928')}
    net = build_network()
    _,loader,_ = build_loader('val',range(frames))
    timings = {k:[] for k in points}
    peaks = {k:[] for k in points}
    rng = np.random.default_rng(928)

    def forward(batch,point):
        out = net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k != 'img'})
        if point['family'] == 'B0':
            return out['final_occ'][0]
        logits = out['pred_occ'][-1][0]
        maximum,fg = logits[:17].max(0)
        margin = maximum-logits[17]
        ids = torch.where(margin.abs() <= BAND)[0]
        if point['source'] == 'margin':
            score = margin[ids]
        else:
            predictor = predictors[point['source']]
            if predictor.record['support']:
                support,_ = support_field(net.head,out,check=False)
            else:
                support = torch.zeros_like(margin)
            x = candidate_features(out,support,ids)
            score = score_for(point['source'],predictor,x,fg[ids],{})
        threshold = torch.as_tensor(point['threshold'],device=score.device,dtype=score.dtype)
        if threshold.ndim:
            threshold = threshold[fg[ids]]
        prediction = out['final_occ'][0].clone()
        prediction[ids] = torch.where(score >= threshold,fg[ids],17)
        return prediction

    with torch.inference_mode():
        for frame,batch in enumerate(loader):
            batch = to_cuda(batch)
            keys = list(points)
            rng.shuffle(keys)
            for key in keys:
                point = points[key]
                for _ in range(2):
                    forward(batch,point)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for _ in range(repeats):
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    prediction = forward(batch,point)
                    end.record()
                    end.synchronize()
                    elapsed.append(start.elapsed_time(end))
                timings[key].append(elapsed)
                peaks[key].append(torch.cuda.max_memory_allocated()/2**20)
            print('timing input',frame+1,'/',frames,flush=True)
    result = dict(status='complete',frames=frames,repeats=repeats,warmups_per_frame_method=2,
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,batch_size=1,
        input='preloaded GPU-resident images/metas; data loading and H2D excluded for every method',
        support='one extra native aggregation; not fused; included in all support-using methods',
        models={},source_sha256=file_sha256(__file__))
    for key,values in timings.items():
        a = np.asarray(values)
        result['models'][key] = dict(mean_latency_ms=float(a.mean()),median_latency_ms=float(np.median(a)),
            FPS=float(1000/a.mean()),p10_latency_ms=float(np.quantile(a,.1)),p90_latency_ms=float(np.quantile(a,.9)),
            peak_allocated_MiB=max(peaks[key]),per_frame_repeat_ms=a.tolist())
    write_json(RUN/'benchmark.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['ray','benchmark'])
    parser.add_argument('--limit',type=int,default=0)
    args = parser.parse_args()
    if args.stage == 'ray':
        ray_evaluate(args.limit)
    else:
        benchmark()


if __name__ == '__main__':
    main()
