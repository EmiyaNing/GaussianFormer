"""One fixed-budget joint-state adapter probe with the native core frozen."""

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.pmg_world.evaluate import evaluate
from tools.pmg_world.regions import histogram
from tools.pmg_world_native_adapt.train import adaptation_mode
from tools.pmg_world_native_adapt.gate_train import native_rollout
from tools.pmg_world_repair.train import Tee, loss_control
from tools.sparse_opus_gausisan_world import runtime

PREFIX = 'native_state_adapter'


def is_trainable(name):
    return name.startswith(PREFIX + '.')


def engineering_check(model, objective, train, validation, cfg, order, output):
    """Check identity, actual output coupling, bounds, and trainable gradients."""
    rng = runtime.capture_rng()
    initial = {k: v.detach().clone() for k, v in model.native_state_adapter.state_dict().items()}
    fields = ('forecast_points_list', 'forecast_features_list', 'forecast_semantics_list')
    model.eval()
    data = next(iter(runtime.loader(validation, cfg.val_loader, [0], cfg.seed)))
    images, metas = runtime.move_batch(data, 'cuda')
    with torch.no_grad():
        with native_rollout(model):
            native = model(imgs=images, metas=metas)
        model.memory.clear()
        model.memory_queue.clear()
        candidate = model(imgs=images, metas=metas)
        assert all(torch.equal(a, b) for key in fields for a, b in zip(native[key], candidate[key]))
        grids = [candidate['final_occ_grid'], *candidate['future_occ_grids']]
        targets = [metas['occ_label'], *metas['future_occ_labels'].unbind(1)]
        counts = np.stack([histogram(p.cpu().numpy(), t.cpu().numpy()) for p, t in zip(grids, targets)])
        with np.load('analysis/pmg_world_stage3_20261023/diagnosis_1024/samples/0000.npz') as old:
            assert np.array_equal(counts, old['global_confusion'][0])
        model.native_state_adapter.up.bias.fill_(.1)
        perturbed = model(imgs=images, metas=metas)
        assert torch.equal(candidate['final_occ_grid'], perturbed['final_occ_grid'])
        changed_fields = {key: any(not torch.equal(a, b) for a, b in zip(candidate[key], perturbed[key])) for key in fields}
        assert all(changed_fields.values()), changed_fields
        model.force_zero_adapter = True
        restored = model(imgs=images, metas=metas)
        assert all(torch.equal(a, b) for key in fields for a, b in zip(native[key], restored[key]))
        model.force_zero_adapter = False
        model.native_state_adapter.load_state_dict(initial)
    del native, candidate, perturbed, restored, images, metas, data, grids, targets
    adaptation_mode(model)
    data = next(iter(runtime.loader(train, cfg.train_loader, order[:1], cfg.seed)))
    images, metas = runtime.move_batch(data, 'cuda')
    forward_rng = runtime.capture_rng()
    with native_rollout(model), torch.no_grad():
        result = model(imgs=images, metas=metas)
        native_loss, native_details = objective(dict(result, metas=metas, update_stamp_statistics=False))
        native_value = float(native_loss)
        native_terms = {key: float(value) for key, value in native_details.items()}
        native_states = {key: [v.detach().clone() for v in result[key]] for key in fields}
    del result, native_loss
    runtime.restore_rng(forward_rng)
    adaptation_mode(model)
    result = model(imgs=images, metas=metas)
    assert set(result['gaussian_logits_by_horizon']) == set(range(7))
    assert all(torch.equal(a, b) for key in fields for a, b in zip(native_states[key], result[key]))
    loss, details = objective(dict(result, metas=metas, update_stamp_statistics=False))
    active = {key: float(details[key]) - value for key, value in native_terms.items()
              if key.startswith(('fu', 'loss_traj')) or (key.startswith('gaussian.h') and key != 'gaussian.h0.loss_voxel')}
    assert max(abs(v) for v in active.values()) < 1e-5, active
    loss.backward()
    gradients = {name: float(p.grad.double().norm()) if p.grad is not None else None
                 for name, p in model.native_state_adapter.named_parameters()}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.native_state_adapter.parameters())
    assert gradients['up.weight'] > 0 and gradients['up.bias'] > 0
    assert gradients['down.weight'] == 0 and gradients['down.bias'] == 0
    assert all(p.grad is None for name, p in model.named_parameters() if not is_trainable(name))
    model.zero_grad(set_to_none=True)
    assert all(torch.equal(value, model.native_state_adapter.state_dict()[key]) for key, value in initial.items())
    audit = dict(status='passed', preflight_optimizer_updates=0, validation_sample=0, training_sample=order[0],
        native_future_points_features_logits_exact=True, initial_seven_horizon_counts_exact=True,
        nonzero_adapter_current_exact=True, nonzero_adapter_changes=changed_fields,
        forced_zero_restores_native=True, gradient_norms=gradients,
        zero_down_gradient_expected='Zero final projection initially blocks gradient to the first projection only',
        frozen_gradients_absent=True, total_loss_difference=float(loss.detach()) - native_value,
        maximum_active_loss_term_error=max(abs(v) for v in active.values()), focal_control=loss_control())
    runtime.atomic_json(output / 'engineering_check.json', audit)
    print('ADAPTER_PREFLIGHT ' + json.dumps(audit), flush=True)
    del result, loss, details, images, metas, data, native_states, initial
    gc.collect()
    torch.cuda.empty_cache()
    runtime.restore_rng(rng)
    adaptation_mode(model)


def run(args):
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output, report = Path(args.work_dir), Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / 'completed.json').exists() or (output / 'latest.pth').exists():
        raise FileExistsError('This probe must be run once, uninterrupted; existing artifacts are preserved')
    files = [args.config, __file__, 'model/segmentor/pmg_native_state_adapter_world.py',
        'model/segmentor/sparse_opus_full_horizon_baseline.py', 'model/segmentor/sparse_opus_gausisan_world.py',
        'model/segmentor/sparseworld_strict.py', 'loss/pmg_repair_distance_weight.py',
        'loss/pmg_time_supervision_loss.py', 'loss/sparseworld_strict_loss.py',
        'tools/pmg_world_native_adapt/train.py', 'tools/pmg_world_native_adapt/gate_train.py',
        'tools/pmg_world_repair/train.py', 'tools/pmg_world/evaluate.py']
    identity = dict(checkpoint_sha256=runtime.digest(cfg.adaptation_checkpoint),
                    source_sha256={str(p): runtime.digest(p) for p in files}, manifests=runtime.audit_manifests(cfg))
    path = output / 'identity.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('Source identity changed since preflight')
    runtime.atomic_json(path, identity)
    model = build_segmentor(cfg.model)
    source = torch.load(cfg.adaptation_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    missing = model.load_state_dict(source['state_dict'], strict=False)
    expected_new = {PREFIX + '.' + name for name in model.native_state_adapter.state_dict()}
    assert set(missing.missing_keys) == expected_new and not missing.unexpected_keys
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(source['epoch_index']))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = source['query_stamps'].cuda()
    model.pts_bbox_head.reset_mask()
    del source, counts
    for name, p in model.named_parameters():
        p.requires_grad_(is_trainable(name))
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters) == 24896
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train, validation = OPENOCC_DATASET.build(cfg.train_dataset_config), OPENOCC_DATASET.build(cfg.val_dataset_config)
    accumulation, planned = int(cfg.gradient_accumulation), int(cfg.adaptation_updates)
    order = np.random.default_rng(20261023).permutation(len(train)).tolist()[:planned * accumulation]
    ref = json.loads(Path('out/pmg_world_native_adapt_20261024/N_native/protocol.json').read_text())
    assert order == ref['training_sample_indices'] and len(order) == 2048
    assert planned == 512 and accumulation == 4 and len(validation) == 1024
    cfg.dump(str(output / 'resolved_config.py'))
    protocol = dict(arm='G2-state-adapter', training_sample_indices=order, sample_order_seed=20261023,
        planned_updates=planned, training_samples=2048, validation_samples=1024, trainable_modules=[PREFIX],
        trainable_parameters=24896, new_parameters=24896, frozen_native_core=True,
        frozen_perception=True, frozen_plan_head=True, frozen_attribute_decoder=True,
        source_checkpoint=cfg.adaptation_checkpoint, source_sha256=identity['checkpoint_sha256'],
        query_assignment_fixed=True, native_dropout_disabled=True, microbatch_size=1, accumulation=4,
        future_focal_distance_weight_detached=True, distillation=False, GT_region_training_masks=False,
        final_checkpoint_only=True, parameter_interpolation=False, parameter_projection=False,
        adapter_rank=32, residual_relative_norm_bound=.1, actual_training_seed=cfg.seed,
        interpretation='Composite architecture probe: vector residual plus frozen core and a fixed local norm bound; not an isolated gate-capacity ablation',
        bound_scope='Each correction versus the native increment at the current adapted state; not a multi-step or output guarantee')
    runtime.atomic_json(output / 'protocol.json', protocol)
    print('PROTOCOL ' + json.dumps({k:v for k,v in protocol.items() if k != 'training_sample_indices'}), flush=True)
    engineering_check(model, objective, train, validation, cfg, order, output)
    if args.check_only:
        return
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=.01)
    adaptation_mode(model)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started, losses, updates = time.monotonic(), [], 0

    def save():
        runtime.atomic_save(output / 'latest.pth', dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            epoch_index=model.curr_epoch, updates=updates, query_stamps=model.pts_bbox_head.ind_stamps_all,
            rng=runtime.capture_rng()))

    for microbatch, data in enumerate(runtime.loader(train, cfg.train_loader, order, cfg.seed)):
        images, metas = runtime.move_batch(data, 'cuda')
        result = model(imgs=images, metas=metas)
        loss, details = objective(dict(result, metas=metas, update_stamp_statistics=False))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        (loss / accumulation).backward()
        losses.append(float(loss.detach()))
        latest_losses = {key:float(value) for key,value in details.items()}
        del loss, result, details, images, metas, data
        if (microbatch + 1) % accumulation:
            continue
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True))
        factor = (1/3 + 2/3*updates/32) if updates < 32 else .001+.999*.5*(1+np.cos(np.pi*(updates-32)/(planned-32)))
        for group in optimizer.param_groups:
            group['lr'] = 2e-4 * float(factor)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if updates % 16 == 0:
            progress = dict(status='training', updates=updates, planned_updates=planned,
                samples_consumed=updates*accumulation, mean_loss=float(np.mean(losses[-64:])),
                gradient_norm=norm, lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-started,
                latest_losses=latest_losses, adapter_last_microbatch=model.last_adapter_statistics)
            runtime.atomic_json(output / 'progress.json', progress)
            print('STATE_ADAPTER ' + json.dumps(progress), flush=True)
        if updates % 128 == 0:
            save()
    assert updates == planned
    save()
    training_seconds = time.monotonic()-started
    peak = torch.cuda.max_memory_allocated()/2**20
    source = torch.load(cfg.adaptation_checkpoint, map_location='cpu', mmap=True, weights_only=False)
    state = model.state_dict()
    changed = [key for key, value in source['state_dict'].items() if not torch.equal(value, state[key].detach().cpu())]
    query_exact = torch.equal(source['query_stamps'], model.pts_bbox_head.ind_stamps_all.cpu())
    new_keys = sorted(set(state)-set(source['state_dict']))
    runtime.atomic_json(output/'state_invariants.json', dict(frozen_state_changed_keys=changed,
        frozen_state_exact=not changed, query_assignment_exact=query_exact, new_state_keys=new_keys))
    assert not changed and query_exact and set(new_keys)==expected_new
    del source, state, optimizer, objective
    gc.collect()
    torch.cuda.empty_cache()
    runtime.atomic_json(output/'progress.json',dict(status='evaluating',updates=updates,training_seconds=training_seconds))
    # Collect complete-validation diagnostics without changing model outputs.
    recorded = {}
    original_record = model._record_adapter
    def record_adapter(step, ratio):
        original_record(step, ratio)
        item = model.last_adapter_statistics[-1]
        assert item['ratio_max'] <= .100001, item
        count = item['query_evaluations']
        row = recorded.setdefault(step, dict(query_evaluations=0, ratio_sum=0., ratio_max=0.,
            delta_norm_sum=0., increment_norm_sum=0., cosine_count=0, cosine_sum=0.))
        row['query_evaluations'] += count
        row['ratio_sum'] += item['ratio_mean'] * count
        row['ratio_max'] = max(row['ratio_max'], item['ratio_max'])
        row['delta_norm_sum'] += item['delta_norm_mean'] * count
        row['increment_norm_sum'] += item['increment_norm_mean'] * count
        row['cosine_count'] += item['cosine_valid_count']
        row['cosine_sum'] += item['cosine_mean'] * item['cosine_valid_count']
    model._record_adapter = record_adapter
    try:
        metrics=evaluate(model,validation,cfg,report,1)
    finally:
        del model._record_adapter
    rows=[]
    for step, row in sorted(recorded.items()):
        count=row['query_evaluations']
        rows.append(dict(seconds=step*.5, query_evaluations=count,
            ratio_mean=row['ratio_sum']/count, ratio_max=row['ratio_max'],
            delta_norm_mean=row['delta_norm_sum']/count,
            increment_norm_mean=row['increment_norm_sum']/count,
            cosine_valid_count=row['cosine_count'],
            cosine_mean=row['cosine_sum']/max(row['cosine_count'],1)))
    runtime.atomic_json(report/'adapter_statistics.json',dict(rows=rows,
        scope='All validation query slots; not physical objects, GT motion groups, or global stability guarantees'))
    runtime.atomic_json(report/'final_512updates_metrics.json',dict(metrics,actual_training_updates=updates,
        training_samples_consumed=2048,budget_note='512 updates, not a full epoch'))
    runtime.atomic_json(output/'completed.json',dict(status='completed',updates=updates,
        training_seconds=training_seconds,evaluation_seconds=metrics['evaluation_seconds'],
        total_training_and_evaluation_seconds=time.monotonic()-started,peak_training_cuda_mib=peak,
        selection_mean_miou=metrics['selection_mean_miou']))
    runtime.atomic_json(output/'progress.json',dict(status='completed',updates=updates))
    print('STATE_ADAPTER_COMPLETED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    directory=Path(args.work_dir)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'train.log').open('a',buffering=1) as log:
        original=sys.stdout
        sys.stdout=Tee(original,log)
        try:
            run(args)
        except Exception as error:
            runtime.atomic_json(directory/'failure.json',dict(type=type(error).__name__,message=str(error)))
            traceback.print_exc(file=sys.stdout)
            raise
        finally:
            sys.stdout=original


if __name__=='__main__':
    main()
