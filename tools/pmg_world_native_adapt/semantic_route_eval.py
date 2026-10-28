"""Evaluate one fixed child-Gaussian semantic routing hypothesis, without training."""

import argparse
from contextlib import contextmanager, redirect_stdout
import gc
import json
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from model.segmentor.pmg_query_update_gate_world import PMGQueryUpdateGateWorld
from model.segmentor.pmg_semantic_gaussian_route_world import FORECAST_MODULES
from model.segmentor.sparse_opus_gausisan_world import SparseOpusGaussianWorld
from tools.pmg_world.evaluate import evaluate
from tools.pmg_world.regions import histogram
from tools.pmg_world_repair.train import Tee
from tools.sparse_opus_gausisan_world import runtime


@contextmanager
def unmodified_g1(model):
    assert '_forecast' not in model.__dict__ and '_render_occupancy' not in model.__dict__
    model._forecast = MethodType(PMGQueryUpdateGateWorld._forecast,model)
    model._render_occupancy = MethodType(SparseOpusGaussianWorld._render_occupancy,model)
    try:
        yield
    finally:
        del model._forecast
        del model._render_occupancy


def all_grids(result):
    return [result['final_occ_grid'],*result['future_occ_grids']]


@torch.no_grad()
def engineering_check(model,dataset,cfg,output):
    rng=runtime.capture_rng()
    data=next(iter(runtime.loader(dataset,cfg.val_loader,[0],cfg.seed)))
    images,metas=runtime.move_batch(data,'cuda')
    with unmodified_g1(model):
        original=model(imgs=images,metas=metas)
    original_grids=[value.clone() for value in all_grids(original)]
    del original
    model.route_mode='candidate'
    candidate=model(imgs=images,metas=metas)
    assert all(torch.equal(a,b) for a,b in zip(original_grids,all_grids(candidate)))
    del candidate,original_grids
    model.route_mode='baseline'
    baseline=model(imgs=images,metas=metas)
    targets=[metas['occ_label'],*metas['future_occ_labels'].unbind(1)]
    counts=np.stack([histogram(p.cpu().numpy(),t.cpu().numpy()) for p,t in zip(all_grids(baseline),targets)])
    with np.load('analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz') as old:
        assert np.array_equal(counts,old['global_confusion'][0])
    baseline_current=baseline['final_occ_grid'].clone()
    del baseline,targets
    model.route_mode='semantic'
    model.audit_route_fields=True
    result=model(imgs=images,metas=metas)
    assert torch.equal(result['final_occ_grid'],baseline_current)
    assert len(model.last_route_audit)==6
    assert all(all(row.values()) for row in model.last_route_audit)
    masks=[dict(seconds=(i+1)*.5,children=mask.numel(),selected=int(mask.sum())) for i,mask in enumerate(model._route_masks)]
    assert all(0<row['selected']<row['children'] for row in masks)
    audit=dict(status='passed',preflight_training_updates=0,validation_index=0,
        candidate_only_seven_grids_match_original_G1_exact=True,
        baseline_only_seven_horizon_counts_match_T_A0_exact=True,
        semantic_route_current_grid_exact=True,retained_reference_all_fields_exact=True,
        selected_candidate_all_fields_exact=True,route_masks=masks,
        slot_scope='Same query and child allocation; does not assert physical object correspondence',
        shared_query_features_note='Result query contexts remain native; Gaussian attributes selected after separate decoding')
    runtime.atomic_json(output/'engineering_check.json',audit)
    print('SEMANTIC_ROUTE_PREFLIGHT '+json.dumps(audit),flush=True)
    model.audit_route_fields=False
    del result,images,metas,data,baseline_current
    model.memory.clear()
    model.memory_queue.clear()
    gc.collect()
    torch.cuda.empty_cache()
    runtime.restore_rng(rng)


def run(args):
    cfg=Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output,report=Path(args.work_dir),Path(cfg.report_directory)
    output.mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    if (output/'completed.json').exists():
        raise FileExistsError('The fixed single semantic-routing probe is already complete')
    reference=torch.load(cfg.adaptation_checkpoint,map_location='cpu',mmap=True,weights_only=False)
    candidate=torch.load(cfg.candidate_checkpoint,map_location='cpu',mmap=True,weights_only=False)
    assert torch.equal(reference['query_stamps'],candidate['query_stamps'])
    assert reference['epoch_index']==candidate['epoch_index']
    prefixes=set(FORECAST_MODULES)
    shared=[name for name in reference['state_dict'] if name.split('.',1)[0] not in prefixes]
    changed_shared=[name for name in shared if not torch.equal(reference['state_dict'][name],candidate['state_dict'][name])]
    assert not changed_shared,changed_shared
    model=build_segmentor(cfg.model)
    loaded=model.load_state_dict(candidate['state_dict'],strict=False)
    bank_keys={'reference_forecast_modules.'+name for name in model.reference_forecast_modules.state_dict()}
    assert set(loaded.missing_keys)==bank_keys and not loaded.unexpected_keys
    for name in FORECAST_MODULES:
        tensors={key[len(name)+1:]:value for key,value in reference['state_dict'].items() if key.startswith(name+'.')}
        model.reference_forecast_modules[name].load_state_dict(tensors,strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.cuda().eval()
    counts=model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(candidate['epoch_index']))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all=candidate['query_stamps'].cuda()
    model.pts_bbox_head.reset_mask()
    files=[args.config,__file__,'model/segmentor/pmg_semantic_gaussian_route_world.py',
        'model/segmentor/pmg_query_update_gate_world.py','model/segmentor/sparseworld_strict.py',
        'model/segmentor/sparse_opus_full_horizon_baseline.py','model/segmentor/sparse_opus_gausisan_world.py',
        'model/encoder/opus_v1_gaussian_encoder.py','model/encoder/opus_v1_gaussian_softplus_encoder.py',
        'model/head/gaussian_opus_head.py','model/head/sparse_opus_gausisan_world_head.py','tools/pmg_world/evaluate.py']
    identity=dict(reference_checkpoint=cfg.adaptation_checkpoint,candidate_checkpoint=cfg.candidate_checkpoint,
        checkpoint_sha256={str(p):runtime.digest(p) for p in (cfg.adaptation_checkpoint,cfg.candidate_checkpoint)},
        source_sha256={str(p):runtime.digest(p) for p in files},manifests=runtime.audit_manifests(cfg))
    runtime.atomic_json(output/'identity.json',identity)
    protocol=dict(arm=cfg.pmg_experiment.arm,additional_training_updates=0,validation_samples=1024,
        reference='T-A0 epoch10',candidate='G1 final512',route_class_ids=list(model.route_class_ids),
        route_class_names=list(cfg.semantic_route_experiment.class_names),
        route_source='Reference per-child raw semantic argmax; no GT or candidate-derived mask',
        route_level='All five decoded Gaussian fields per child',one_shared_perception=True,
        independent_forecast_paths=True,selected_outputs_feedback=False,gate_amplitude_tuning=False,
        shared_state_exact=True,query_assignment_exact=True,
        reference_forecast_extra_parameters=sum(p.numel() for p in model.reference_forecast_modules.parameters()),
        trainable_parameters=0,original_G1_gate_retained=True,
        interpretation='Post-hoc fixed-weight semantic output routing; not a retrained conditional gate or a motion oracle',
        acceptance='Original joint thresholds versus T-A0; also compare full G1, N-native and N-half',
        next_training='No training in this diagnostic; judge learnable architecture after this result')
    runtime.atomic_json(output/'protocol.json',protocol)
    cfg.dump(str(output/'resolved_config.py'))
    print('PROTOCOL '+json.dumps(protocol),flush=True)
    dataset=OPENOCC_DATASET.build(cfg.val_dataset_config)
    assert len(dataset)==1024
    engineering_check(model,dataset,cfg,output)
    if args.check_only:
        return
    model.reset_route_statistics(enabled=True)
    runtime.atomic_json(output/'progress.json',dict(status='evaluating',additional_training_updates=0,samples=1024))
    started=time.monotonic()
    metrics=evaluate(model,dataset,cfg,report,1)
    runtime.atomic_json(report/'route_statistics.json',model.route_statistics())
    state=model.state_dict()
    changed_candidate=[name for name,value in candidate['state_dict'].items() if not torch.equal(value,state[name].detach().cpu())]
    changed_reference=[name for name,value in reference['state_dict'].items() if name.split('.',1)[0] in prefixes
                       and not torch.equal(value,state['reference_forecast_modules.'+name].detach().cpu())]
    audit=dict(candidate_state_changed_keys=changed_candidate,reference_state_changed_keys=changed_reference,
        all_weights_and_buffers_exact=not changed_candidate and not changed_reference,
        query_assignment_exact=torch.equal(reference['query_stamps'],model.pts_bbox_head.ind_stamps_all.cpu()))
    runtime.atomic_json(output/'state_invariants.json',audit)
    assert audit['all_weights_and_buffers_exact'] and audit['query_assignment_exact']
    runtime.atomic_json(output/'completed.json',dict(status='completed',additional_training_updates=0,
        validation_samples=1024,evaluation_seconds=metrics['evaluation_seconds'],
        evaluation_and_final_audit_seconds=time.monotonic()-started,selection_mean_miou=metrics['selection_mean_miou']))
    runtime.atomic_json(output/'progress.json',dict(status='completed',additional_training_updates=0,samples=1024))
    print('SEMANTIC_ROUTE_COMPLETED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    output=Path(args.work_dir)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'evaluation.log').open('a',buffering=1) as log,redirect_stdout(Tee(sys.stdout,log)):
        try:
            run(args)
        except Exception as error:
            runtime.atomic_json(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
            traceback.print_exc(file=sys.stdout)
            raise


if __name__=='__main__':
    main()
