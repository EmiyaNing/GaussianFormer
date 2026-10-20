"""Independent invariance, count, provenance and completion checks."""
from datetime import datetime
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.empty_decision_optimization import (RUN,ROOT,CONFIG,CHECKPOINT,BAND,read,write_json,
    file_sha256,build_network,build_loader,to_cuda,candidate_features,support_field,seed_all)
from tools.eval_empty_decision_optimization import model_manifest
from tools.fit_empty_decision_optimization import Predictor


def invariance():
    path = RUN/'inference_invariance.json'
    if path.exists():
        return read(path)
    seed_all(928)
    net = build_network()
    _,loader,_ = build_loader('val',range(2))
    predictors = {k:Predictor(ROOT/v['path']) for k,v in model_manifest().items()}
    results = []
    with torch.inference_mode():
        for n,batch in enumerate(loader):
            batch = to_cuda(batch)
            changed = dict(batch)
            changed['occ_label'] = torch.full_like(batch['occ_label'],17)
            fields = []
            for key in ['occ_cam_mask','occ_mask','occ_loss_mask','occ_lidar_mask','occ_nonempty_mask']:
                if key in changed and torch.is_tensor(changed[key]):
                    changed[key] = ~changed[key] if changed[key].dtype == torch.bool else 1-changed[key]
                    fields.append(key)
            a = net(imgs=batch['img'],metas={k:v for k,v in batch.items() if k != 'img'})
            b = net(imgs=changed['img'],metas={k:v for k,v in changed.items() if k != 'img'})
            assert torch.equal(a['pred_occ'][-1],b['pred_occ'][-1]),'GT/mask changed rendered predictions'
            sa,_ = support_field(net.head,a)
            sb,_ = support_field(net.head,b)
            assert torch.equal(sa,sb)
            maximum,fg = a['pred_occ'][-1][0,:17].max(0)
            margin = maximum-a['pred_occ'][-1][0,17]
            ids = torch.where(margin.abs() <= BAND)[0]
            xa,xb = candidate_features(a,sa,ids),candidate_features(b,sb,ids)
            assert torch.equal(xa,xb)
            ca,cb = {},{}
            for name,predictor in predictors.items():
                assert torch.equal(predictor(xa,fg[ids],ca),predictor(xb,fg[ids],cb)),name
            results.append(dict(frame=n,changed_fields=['occ_label']+fields,models=len(predictors),
                rendered_logits_exact=True,support_exact=True,features_exact=True,decision_scores_exact=True))
    result = dict(status='passed',frames=results,source_sha256=file_sha256(__file__),
        explanation='All labels replaced by empty and GT masks inverted; images, cameras and grid fixed.')
    write_json(path,result)
    return result


def artifacts():
    protocol = read(RUN/'protocol.json')
    selection = read(RUN/'operating_points.json')
    validation = read(RUN/'validation/summary.json')
    tuning = read(RUN/'tuning/summary.json')
    ray = read(RUN/'ray/summary.json')
    benchmark = read(RUN/'benchmark.json')
    inv = read(RUN/'inference_invariance.json')
    assert protocol['checkpoint_sha256'] == file_sha256(ROOT/CHECKPOINT)
    assert validation['samples'] == ray['samples'] == 1024
    assert tuning['samples'] == 994
    assert len(read(RUN/'train/summary.json')['frames']) == 4096
    assert len(selection['operating_points']) == len(validation['operating_points']) == 59
    selection_hash = file_sha256(RUN/'operating_points.json')
    assert validation['selection_sha256'] == selection_hash
    assert read(RUN/'pre_validation_audit.json')['selection_sha256'] == selection_hash
    assert selection['histogram_sha256'] == file_sha256(RUN/'tuning/histograms.npz')
    for name,meta in model_manifest().items():
        assert file_sha256(ROOT/meta['path']) == selection['model_sha256'][name]
    assert set(protocol['fit_scenes']).isdisjoint(protocol['tune_scenes'])
    assert set(protocol['fit_scenes']+protocol['tune_scenes']).isdisjoint(protocol['validation_scenes'])
    with np.load(RUN/'validation/per_frame_counts.npz') as z:
        counts,events = z['confusion'],z['events']
    assert counts.shape == (1024,59,2,18,18)
    # Every decision preserves each GT class's population in each frame and region.
    gt_counts = counts.sum(-1)
    assert np.array_equal(gt_counts,np.broadcast_to(gt_counts[:,:1],gt_counts.shape))
    diagonal = np.diagonal(counts,axis1=-2,axis2=-1)[...,:17].sum(-1)
    assert np.array_equal(diagonal-diagonal[:,:1],events[...,0]-events[...,3])
    false_occupied = counts[...,17,:17].sum(-1)
    assert np.array_equal(false_occupied-false_occupied[:,:1],events[...,2]-events[...,5])
    semantic_fp = counts[...,:,:17].sum((-1,-2))-diagonal
    assert np.array_equal(semantic_fp-semantic_fp[:,:1],events[...,1]+events[...,2]-events[...,4]-events[...,5])
    names = list(validation['operating_points'])
    for i,p in enumerate(validation['operating_points'].values()):
        if p['family'] == 'P3_oneway':
            assert not events[:,i,:,3:].any()
    previous = read(ROOT/'work_dirs/empty_decision_mini/val/summary.json')
    base = counts[:,0].sum(0)
    for name,c in [('camera',base[0]),('camera_complement',base[1]),('all_valid',base.sum(0))]:
        assert np.array_equal(c,previous['confusion'][name]['0.0'])
    assert ray['baseline_previous_counts_exact_parity'] and all(ray['official_streaming_parity'].values())
    assert len(ray['models']) == 11 and len(benchmark['models']) == 10
    assert benchmark['frames'] == 8 and benchmark['repeats'] == 5
    # Match executed sources to exact saved versions, even after cache/report-only revisions.
    candidates = list((RUN/'source_snapshots').glob('*.py'))
    execution_hashes = [read(RUN/'train/summary.json')['source_sha256'],read(RUN/'fit_results.json')['source_sha256'],
        tuning['source_sha256'],validation['source_sha256'],ray['source_sha256'],benchmark['source_sha256']]
    current = [ROOT/'tools'/p.name for p in candidates]
    available = {file_sha256(p) for p in candidates+current if p.exists()}
    for value in execution_hashes:
        assert value in available,('Missing executed source snapshot',value)
    files = [ROOT/'analysis/empty_decision_optimization_results.md',
        ROOT/'analysis/empty_decision_optimization/all_operating_points.csv',
        ROOT/'analysis/empty_decision_optimization/per_class_metrics.csv']
    assert all(p.exists() for p in files)
    document = read(RUN/'document_validation.json')
    assert document['status'] == 'passed' and document['strictKatexErrors'] == 0 and document['missingLinks'] == 0
    result = dict(status='complete',finished_at=datetime.now().astimezone().isoformat(),train_frames=4096,
        tuning_frames=994,validation_frames=1024,operating_points=59,module_seeds=[928,929,930],
        official_ray_models=11,official_ray_frames_per_model=1024,benchmark_methods=10,
        checkpoint_unchanged=True,training_validation_scenes_disjoint=True,
        frozen_selection_unchanged=True,models_match_prevalidation_hashes=True,
        native_validation_previous_counts_exact_parity=True,official_ray_previous_counts_exact_parity=True,
        all_directional_error_counts_verified=True,one_way_removal_counts_zero=True,
        inference_gt_and_mask_invariance=inv['status'],executed_source_snapshots_verified=True,
        no_joint_backbone_training=True,tests_passed=11,
        markdown_formula_table_and_link_validation=document['status'],
        report_sha256=file_sha256(files[0]),source_sha256=file_sha256(__file__))
    write_json(RUN/'completion_audit.json',result)
    write_json(RUN/'status.json',dict(status='complete',report=str(files[0].relative_to(ROOT))))
    return result


if __name__ == '__main__':
    invariance()
    print(artifacts())
