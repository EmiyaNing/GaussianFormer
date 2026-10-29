"""Full frozen small-checkpoint census, reusing the audited tiny diagnostics.

python3.10 tools/diagnose_small_gaussians.py --action pilot
python3.10 tools/diagnose_small_gaussians.py --action run --workers 3
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from mmengine import Config
from tools import stat_query_budget as query
from tools.stat_boundary_gaussians import diagnostic, forward, to_cuda, read, write, sha
from tools.boundary_gaussian_stats import gaussian_counts, SCOPES, COHORTS
from tools.boundary_gaussian_stats.data import FeatureCache
from tools.stat_gaussian_support_regions import gaussian_types

RUN = ROOT / 'work_dirs/small_gaussian_diagnosis_full'
CONFIG = ROOT / 'config/full_occ3d_oups_guassian/opusv1_gaussian_softplus_learned_bias_small_8f.py'
CHECKPOINT = ROOT / 'ckpts/opus_gaussian_8f_small.pth'
Q, CHILDREN, COARSE = 1200, 64, 32
query.Q, query.CHILDREN, query.COARSE_CHILDREN = Q, CHILDREN, COARSE
GROUPS = (
    'all', 'center_visible_foreground', 'center_visible_empty', 'center_other',
    'center_visible_empty_reaches_foreground_native',
    'center_visible_empty_pure_empty_native', 'pure_empty_camera_native',
    'pure_empty_all_native', 'pure_empty_camera_3sigma',
    'center_visible_empty_pure_empty_3sigma', 'pure_empty_all_3sigma',
    'pure_empty_camera_native_fp_owner',
    'center_visible_empty_pure_empty_native_fp_owner',
    'pure_foreground_camera_3sigma', 'mixed_foreground_camera_3sigma',
)
ATTRS = ('semantic_max', 'semantic_sum', 'raw_logit_max', 'sigmoid_score_max',
         'opacity', 'effective_amplitude', 'scale_mean', 'scale_max', 'native_fp_owner_voxels')
FLAGS = ('score_gt_05', 'score_gt_09', 'score_gt_095', 'opacity_gt_09',
         'score_and_opacity_gt_09', 'scale_any_ge_039', 'scale_all_ge_039')


def initialize():
    cfg = Config.fromfile(CONFIG)
    availability = read(ROOT / 'work_dirs/boundary_gaussian_full/availability.json')
    base = read(ROOT / 'work_dirs/boundary_gaussian_full/protocol.json')
    for source, digest in base['source_sha256'].items():
        assert sha(ROOT / source) == digest, source
    annotations = {s: sha(cfg[s + '_dataset_config'].imageset) for s in ('train', 'val')}
    assert annotations == base['annotation_sha256']
    files = [Path(__file__), CONFIG, ROOT / 'tools/stat_query_budget.py',
             ROOT / 'tools/query_budget_stats/kernels.cu',
             ROOT / 'tools/stat_gaussian_support_regions.py',
             ROOT / 'tools/probe_gaussian_scale_supervision.py']
    protocol = dict(version='small_gaussian_diagnosis_v1', config=str(CONFIG.relative_to(ROOT)),
        checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha(CHECKPOINT),
        expected={s: {k: availability[s][k] for k in ('frames', 'scenes')} for s in ('train', 'val')},
        queries=Q, children=CHILDREN, coarse_children=COARSE,
        fields=query.FIELDS, stratum_fields=query.STRAT_FIELDS,
        scopes=SCOPES, cohorts=COHORTS, groups=GROUPS, attributes=ATTRS, flags=FLAGS,
        annotation_sha256=annotations, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        inherited_source_sha256=base['source_sha256'], seed=928,
        scope='All 28130 train and 6019 val frames; frozen eval; camera mask primary; no subsampling',
        high_score='max sigmoid(raw set-classification logits)>0.9; diagnostic score, not calibrated probability; also report 0.5/0.95',
        support='Native production candidate cube and positive FP32 opacity*kernel; 3sigma additionally power>=-4.5',
        assignment='Nearest actual stage-5 Gaussian center; parent index=flat_index//32',
        attribution='Highest individual contribution to maximum foreground class; descriptive, not deletion effect')
    protocol = json.loads(json.dumps(protocol))
    if (RUN / 'protocol.json').exists():
        assert read(RUN / 'protocol.json') == protocol, 'Protocol changed; use a new output directory'
    else:
        write(RUN / 'protocol.json', protocol)
        cfg.dump(RUN / 'resolved_config.py')
        for p in files:
            dst = RUN / 'sources' / p.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
    return cfg, protocol


def build(cfg):
    from visualize import load_model
    from mmengine.runner import set_random_seed
    set_random_seed(928)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    return load_model(cfg, CHECKPOINT, 'cuda:0')


def extra_statistics(network, out, batch, result, values):
    means, scales, inverse, alpha, semantic, xyz, labels, camera, winner, pred = values
    ng = len(means)
    raw = out['opus_pred_logits'][-1][0]
    torch.testing.assert_close(torch.nn.functional.softplus(raw), semantic, rtol=0, atol=0)
    agg = network.head.aggregator
    cells = ((means - agg.pc_min[0]) / agg.grid_size).long()
    in_grid = ((cells >= 0) & (cells < cells.new_tensor([agg.H, agg.W, agg.D]))).all(1)
    safe_cells = torch.minimum(torch.maximum(cells, torch.zeros_like(cells)), cells.new_tensor([agg.H-1, agg.W-1, agg.D-1]))
    ids = (safe_cells[:, 0] * agg.W + safe_cells[:, 1]) * agg.D + safe_cells[:, 2]
    cy = labels[ids]
    visible = in_grid & camera[ids] & (cy >= 0) & (cy < 18)
    center_fg, center_empty = visible & (cy < 17), visible & (cy == 17)
    bits = result[0]
    pure_native, pure_core = bits[1] == (1 << 17), bits[3] == (1 << 17)
    has_fg = (bits[1] & ((1 << 17)-1)) > 0
    valid = camera & (labels >= 0) & (labels < 18)
    fp = valid & (labels == 17) & (pred != 17)
    packed = result[3][0]
    owned = fp & (packed > 0)
    owner = (0xffffffff - (packed[owned] & 0xffffffff)).long()
    owner_counts = torch.bincount(owner, minlength=ng)
    types_core = gaussian_types(bits[3])
    masks = (
        torch.ones(ng, device=means.device, dtype=torch.bool), center_fg, center_empty, ~visible,
        center_empty & has_fg, center_empty & pure_native, pure_native,
        bits[0] == (1 << 17), pure_core, center_empty & pure_core, bits[2] == (1 << 17),
        pure_native & (owner_counts > 0), center_empty & pure_native & (owner_counts > 0),
        types_core == 2, types_core == 0,
    )
    score = raw.sigmoid().amax(1)
    attrs = torch.stack((semantic.amax(1), semantic.sum(1), raw.amax(1), score, alpha,
                         alpha * semantic.sum(1), scales.mean(1), scales.amax(1), owner_counts), 1).double()
    flags = torch.stack((score > .5, score > .9, score > .95, alpha > .9,
                         (score > .9) & (alpha > .9), scales.amax(1) >= .39,
                         scales.amin(1) >= .39), 1).long()
    mask_matrix = torch.stack(masks).double()
    numbers = mask_matrix.sum(1).long()
    sums = mask_matrix @ attrs
    flag_counts = mask_matrix @ flags.double()
    query_boundaries = []
    for scope in (1, 3):
        data = bits[scope]
        fg = data & ((1 << 17)-1)
        many_fg = (fg > 0) & ((fg & (fg-1)) != 0)
        many = (data > 0) & ((data & (data-1)) != 0)
        interface = (fg > 0) & ((data & (1 << 17)) != 0)
        query_boundaries.append(torch.stack((many, many_fg, interface), 1).reshape(Q, CHILDREN, 3).sum(1))
    qb = torch.stack(query_boundaries, 1).short().cpu().numpy()
    return dict(group_counts=numbers.cpu().numpy(), attribute_sums=sums.cpu().numpy(),
                flag_counts=flag_counts.long().cpu().numpy(), boundary_queries=qb,
                fp_owners=int(owner_counts.sum()), unowned_fp=int((fp & (packed == 0)).sum()))


def pilot():
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from tools.boundary_gaussian_stats.reference import reference
    cfg, protocol = initialize()
    network = build(cfg)
    audits = []
    for split in ('val', 'train'):
        dataset = query.QueryFrames(cfg, split)
        opts = deepcopy(cfg.val_dataset_config)
        opts.imageset = cfg[split + '_dataset_config'].imageset
        opts.return_keys = list(opts.return_keys) + ['history_frame_indices']
        native = OPENOCC_DATASET.build(opts)
        cache = FeatureCache(network, dataset)
        b = to_cuda(custom_collate_fn_temporal([dataset[0]]))
        a = to_cuda(custom_collate_fn_temporal([native[0]]))
        with torch.inference_mode():
            network.extract_img_feat = cache.original
            original = forward(network, a)
            network.extract_img_feat = cache.extract
            output = forward(network, b)
            torch.testing.assert_close(original['pred_occ'][-1], output['pred_occ'][-1], rtol=0, atol=0)
            for name in output['gaussian']._fields:
                torch.testing.assert_close(getattr(output['gaussian'], name), getattr(original['gaussian'], name), rtol=0, atol=0)
            row = query.frame_statistics(network, output, b, audit_reference=True)
            result, values, params = diagnostic(network, output, b)
            special = extra_statistics(network, output, b, result, values)
            subset = torch.tensor([0, 63, 64, 38399, 38400, 76799], device='cuda')
            small, inputs, sp = diagnostic(network, output, b, subset)
            expected = reference(*[v.cpu().numpy() for v in inputs], tuple(sp[:3]), sp[3], sp[4], sp[5:])
            for j in (0, 1, 2, 4):
                np.testing.assert_array_equal(small[j].cpu().numpy(), expected[j])
            from tools.stat_gaussian_voxel_coverage import measure
            for qi in (600, 1199):
                lo, hi = qi*CHILDREN, (qi+1)*CHILDREN
                sub, _ = measure(network, output, b, torch.arange(lo, hi, device='cuda'))
                assert int((sub[0][:,1]>0).sum()) == int(row['metrics'][qi,query.FIELDS.index('core_coverage_all_fg')])
            np.testing.assert_array_equal(result[4][1,0].cpu().numpy(), outcome_counts(row['confusion']))
            assert special['fp_owners'] == row['audit']['false_positive'] and special['unowned_fp'] == 0
            gc = gaussian_counts(result[0])
            audits.append(dict(split=split, index=0, query_shape=list(row['metrics'].shape),
                cached_native_predictions_exact=True, numpy_boundary_reference=True,
                core_query_ids_above_600_verified=True, confusion_boundary_exact=True,
                visible_empty_centers=int(special['group_counts'][2]),
                visible_empty_high_score_high_opacity=int(special['flag_counts'][2,4]),
                boundary_camera_3sigma=gc[3]['boundary_any']/76800))
        cache.close()
        del a, b, original, output, row, result, values, special, dataset, native, cache
    write(RUN/'pilot.json', dict(status='passed', checkpoint_sha256=protocol['checkpoint_sha256'], audits=audits))
    print(json.dumps(audits), flush=True)


def outcome_counts(confusion):
    a = np.asarray(confusion)
    correct = int(np.trace(a[:17,:17]))
    wrong = int(a[:17,:17].sum())-correct
    return np.array([correct, wrong, a[:17,17].sum(), a[17,:17].sum(), a[17,17]], dtype=np.int64)


def run(workers):
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Subset
    cfg, protocol = initialize()
    assert read(RUN/'pilot.json')['status'] == 'passed'
    network = build(cfg)
    start, processed, completed, total_done = time.perf_counter(), 0, 0, 0
    for split in ('val','train'):
        dataset = query.QueryFrames(cfg, split)
        by_scene = {}
        for i,(scene,_) in enumerate(dataset.keyframes):
            by_scene.setdefault(scene, []).append(i)
        assert len(dataset) == protocol['expected'][split]['frames']
        assert len(by_scene) == protocol['expected'][split]['scenes']
        pending = []
        for scene,ids in by_scene.items():
            path = RUN/'scenes'/split/(scene+'.json')
            if path.exists():
                prior = read(path)
                assert prior['checkpoint_sha256'] == protocol['checkpoint_sha256'] and prior['indices'] == ids
                assert path.with_suffix('.npz').exists()
                completed += 1
                total_done += len(ids)
            else:
                pending.extend(ids)
        loader = DataLoader(Subset(dataset,pending),batch_size=1,shuffle=False,num_workers=workers,
            collate_fn=custom_collate_fn_temporal,pin_memory=True,
            **(dict(prefetch_factor=2,persistent_workers=True) if workers else {}))
        cache = FeatureCache(network,dataset)
        active = None
        for index,batch in zip(pending,loader):
            scene,frame = dataset.keyframes[index]
            if active is None:
                active = scene
                records,metrics,classes,bq = [],[],[],[]
                strata = np.zeros((11,17,4,4,5), np.int64)
                gc = None
                errors = np.zeros((4,9,5),np.int64)
                cm = np.zeros((18,18),np.int64)
                group_counts = np.zeros(len(GROUPS),np.int64)
                attrs = np.zeros((len(GROUPS),len(ATTRS)),np.float64)
                flags = np.zeros((len(GROUPS),len(FLAGS)),np.int64)
            assert active == scene
            batch = to_cuda(batch)
            with torch.inference_mode():
                output = forward(network,batch)
                assert torch.isfinite(output['pred_occ'][-1]).all()
                row = query.frame_statistics(network,output,batch)
                result,values,params = diagnostic(network,output,batch)
                extra = extra_statistics(network,output,batch,result,values)
                counts = result[4].cpu().numpy()
                np.testing.assert_array_equal(counts[1,0],outcome_counts(row['confusion']))
                assert row['audit']['unattributed_fp'] == extra['unowned_fp'] == 0
                assert row['audit']['false_positive'] == extra['fp_owners']
                np.testing.assert_array_equal(counts[:,0],counts[:,4]+counts[:,7]+counts[:,8])
                fg = row['metrics'][:,query.FIELDS.index('gt_foreground')]
                parts = sum(row['metrics'][:,query.FIELDS.index(k)] for k in ('gt_own_core','gt_neighbor_only_core','gt_no_core'))
                np.testing.assert_array_equal(fg,parts)
            frame_g = gaussian_counts(result[0])
            if gc is None:
                gc = deepcopy(frame_g)
            else:
                for s in range(4):
                    for key,value in frame_g[s].items():
                        gc[s][key] = (np.asarray(gc[s][key])+value).tolist() if isinstance(value,list) else gc[s][key]+value
            info = dataset.base.scene_infos[scene][frame]
            records.append(dict(index=index,scene_frame=int(frame),sample_token=info['token'],**row['audit']))
            metrics.append(row['metrics']);classes.append(row['class_counts']);strata += row['strata']
            bq.append(extra['boundary_queries']);errors += counts;cm += row['confusion']
            group_counts += extra['group_counts'];attrs += extra['attribute_sums'];flags += extra['flag_counts']
            processed += 1;total_done += 1
            if len(records) == len(by_scene[scene]):
                path = RUN/'scenes'/split/(scene+'.json')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.with_suffix('.npz.tmp').open('wb') as stream:
                    np.savez_compressed(stream,metrics=np.stack(metrics),class_counts=np.stack(classes),
                        strata=strata,boundary_queries=np.stack(bq),group_counts=group_counts,
                        attribute_sums=attrs,flag_counts=flags,confusion=cm,boundary_errors=errors)
                path.with_suffix('.npz.tmp').replace(path.with_suffix('.npz'))
                scene_name = Path(info['occ_path']).parts[-3]
                write(path,dict(status='complete',split=split,scene_token=scene,scene_name=scene_name,
                    indices=by_scene[scene],frames=len(records),records=records,
                    gaussian_counts=gc,checkpoint_sha256=protocol['checkpoint_sha256']))
                completed += 1
                elapsed = time.perf_counter()-start
                fps = processed/elapsed
                write(RUN/'progress.json',dict(status='running',completed_scenes=completed,total_scenes=850,
                    completed_frames=total_done,total_frames=34149,split=split,last_scene=scene_name,
                    frames_per_second=fps,remaining_minutes=(34149-total_done)/max(fps,1e-6)/60,
                    seconds=elapsed,updated_at=datetime.now().astimezone().isoformat()))
                if completed%10 == 0:
                    print(f'{completed}/850 scenes, {total_done}/34149 frames, {fps:.2f} frames/s',flush=True)
                active = None
            del output,batch,row,result,values,extra
        assert active is None
        cache.close()
        del loader,dataset,cache
        write(RUN/(split+'_complete.json'),dict(status='complete',**protocol['expected'][split]))
    elapsed = time.perf_counter()-start
    write(RUN/'execution_complete.json',dict(status='complete',frames=total_done,scenes=completed,
        seconds=elapsed,checkpoint_sha256=protocol['checkpoint_sha256']))
    write(RUN/'progress.json',dict(status='complete',completed_scenes=completed,total_scenes=850,
        completed_frames=total_done,total_frames=34149,seconds=elapsed))
    print('Full small-checkpoint diagnosis complete.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['pilot','run'],required=True)
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    pilot() if args.action == 'pilot' else run(args.workers)
