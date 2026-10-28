"""Disposable parity, throughput and multi-scene stability validation.

python3 -m tools.sparse_opus_gausisan_world.validate_performance
Reads an existing checkpoint; all optimizer updates are in memory only.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

import torch
from mmengine import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from model.ops.sparse_localagg import CompactLocalAggregator, _C
from train import find_nonfinite_gradients


def memory():
    stats = torch.cuda.memory_stats()
    return dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                reserved_mib=torch.cuda.memory_reserved() / 2**20,
                peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                inactive_split_mib=stats.get('inactive_split_bytes.all.current', 0) / 2**20,
                allocation_retries=stats.get('num_alloc_retries', 0))


def compare(reference, candidate):
    assert reference['gradients'].keys() == candidate['gradients'].keys()
    differences = sum(float((value - candidate['gradients'][name]).double().square().sum())
                      for name, value in reference['gradients'].items())
    norm = sum(float(value.double().square().sum()) for value in reference['gradients'].values())
    result = dict(loss_absolute_error=abs(reference['loss'] - candidate['loss']),
                  logits_max_error=max(float((value - candidate['logits'][horizon]).abs().max())
                                       for horizon, value in reference['logits'].items()),
                  occupancy_mismatches=[int((a != b).sum()) for a, b in zip(reference['grids'], candidate['grids'])],
                  gradient_relative_l2=(differences / max(norm, 1e-30))**.5,
                  empty_scalar_gradients=[float(x['gradients']['gaussian_head.empty_scalar'])
                                          for x in (reference, candidate)],
                  empty_bias_gradients=[float(x['gradients']['gaussian_head.empty_bias'])
                                        for x in (reference, candidate)])
    assert result['loss_absolute_error'] < 2e-4, result
    assert result['logits_max_error'] < 2e-5, result
    assert max(result['occupancy_mismatches']) <= 5, result
    # End-to-end CUDA sampling/backbone backward is not bitwise deterministic.
    # A repeated legacy baseline is also recorded for comparison.
    assert result['gradient_relative_l2'] < .008, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/sparse_opus_gaussian_world/sparse_opus_gaussian_world_nuscenes_occ3d_15e.py')
    parser.add_argument('--checkpoint', default='out/sparse_opus_gaussian_world_nuscenes_occ3d_15e/epoch_9.pth')
    parser.add_argument('--output', default='analysis/sparse_opus_gaussian_world_performance/optimized_validation.json')
    parser.add_argument('--stability-steps', type=int, default=100)
    args = parser.parse_args()
    assert torch.cuda.is_available() and _C is not None
    assert args.stability_steps >= 2 and args.stability_steps % 2 == 0
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    os.environ['amp'] = os.environ['eval'] = 'false'
    set_random_seed(17)
    cfg = Config.fromfile(args.config)
    saved = torch.load(args.checkpoint, map_location='cpu')
    model = build_segmentor(cfg.model)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cuda()
    optimizer = build_optim_wrapper(model, cfg.optimizer)
    optimizer.load_state_dict(saved['optimizer'])
    original_lrs = [group['lr'] for group in optimizer.param_groups]
    del saved
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    original_statistics = model.pts_bbox_head.num_stamps_all.clone()
    from local_aggregate import LocalAggregator
    legacy_operator = LocalAggregator(**cfg.model.gaussian_head.cuda_kwargs).cuda()
    compact_operator = model.gaussian_head.aggregator
    assert isinstance(compact_operator, CompactLocalAggregator)
    # Both use the same checkpoint-defined grid origin.
    legacy_operator.pc_min.copy_(compact_operator.pc_min)
    train_data = OPENOCC_DATASET.build(cfg.train_dataset_config)
    sample_pairs = [(0, 1000), (100, 5000), (2500, 7500), (4000, 10000),
                    (6000, 12000), (8000, 14000), (11000, 16000), (13000, 19000)]
    batches = [custom_collate_fn_temporal([train_data[a], train_data[b]]) for a, b in sample_pairs]
    scene_names = sorted({str(train_data.keyframes[index][0])
                          for pair in sample_pairs for index in pair})
    records = dict(config=args.config, checkpoint=args.checkpoint,
                   extension=str(_C.__file__), torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                   batch_size=2, gradient_accumulation=2, sample_pairs=sample_pairs,
                   scene_names=scene_names, checkpoint_written=False, parity=[], benchmarks=[], stability=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(records, indent=2) + '\n')

    def configure(optimized, epoch, seed):
        set_random_seed(seed)
        model.pts_bbox_head.num_stamps_all.copy_(original_statistics)
        model.gaussian_head.aggregator = compact_operator if optimized else legacy_operator
        model.gaussian_head.analytic_empty = optimized
        model.gaussian_head.compact_aggregation = optimized
        model.render_training_grids = not optimized
        model.train()
        model.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

    def step(cpu_batch, update=False, snapshot=False, all_grids=False):
        data = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in cpu_batch.items()}
        images = data.pop('img')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = model(imgs=images, metas=data, return_all_occ=all_grids)
        inputs = dict(metas=data, global_iter=0)
        inputs.update({k: result[v] for k, v in cfg.loss_input_convertion.items()})
        loss, details = objective(inputs)
        assert torch.isfinite(loss)
        (loss / 2).backward()
        assert not find_nonfinite_gradients(model)
        torch.cuda.synchronize()
        record = dict(seconds=time.perf_counter() - started, loss=float(loss.detach()),
                      horizons=list(result['gaussian_logits_by_horizon']),
                      future_grids=len(result['future_occ_grids']), before_cleanup=memory())
        captured = None
        if snapshot:
            captured = dict(loss=record['loss'],
                            logits={k: v.detach().cpu() for k, v in result['gaussian_logits_by_horizon'].items()},
                            grids=[v.detach().cpu() for v in [result['final_occ_grid'], *result['future_occ_grids']]],
                            gradients={k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None})
        if update:
            started = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            record['seconds'] += time.perf_counter() - started
        del result, inputs, loss, details, images, data
        record['after_cleanup'] = memory()
        return record, captured

    for epoch, batch_index in [(0, 0), (5, 0), (6, 1), (9, 0), (10, 1)]:
        snapshots = []
        for optimized in (False, True):
            configure(optimized, epoch, 1700 + epoch)
            _, snapshot = step(batches[batch_index], snapshot=True, all_grids=True)
            snapshots.append(snapshot)
        parity = compare(*snapshots)
        if epoch == 10:
            configure(False, epoch, 1700 + epoch)
            _, repeated = step(batches[batch_index], snapshot=True, all_grids=True)
            parity['repeated_legacy'] = compare(snapshots[0], repeated)
            del repeated
        records['parity'].append(dict(epoch_index=epoch, samples=sample_pairs[batch_index], **parity))
        del snapshots, snapshot
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        print('PARITY', epoch, parity, flush=True)
        save()

    for group in optimizer.param_groups:
        group['lr'] = 0.
    for epoch in (9, 10):
        for optimized in (False, True):
            configure(optimized, epoch, 2700 + epoch)
            torch.cuda.empty_cache()
            steps = []
            for index in range(6):
                row, _ = step(batches[0], update=index % 2 == 1)
                steps.append(row)
            benchmark = dict(epoch_index=epoch, optimized=optimized, steps=steps,
                             steady_seconds=statistics.mean(row['seconds'] for row in steps[2:]),
                             peak_mib=max(row['before_cleanup']['peak_mib'] for row in steps[2:]))
            records['benchmarks'].append(benchmark)
            print('BENCHMARK', epoch, optimized, benchmark['steady_seconds'], benchmark['peak_mib'], flush=True)
            save()

    for group, lr in zip(optimizer.param_groups, original_lrs):
        group['lr'] = lr
    configure(True, 10, 17)
    first_weight = model.reg_branch[-1].weight.detach().cpu().clone()
    for index in range(args.stability_steps):
        row, _ = step(batches[index % len(batches)], update=index % 2 == 1)
        records['stability'].append(dict(iteration=index, samples=sample_pairs[index % len(batches)], **row))
        if index == args.stability_steps // 2 - 1:
            val_data = OPENOCC_DATASET.build(cfg.val_dataset_config)
            batch = custom_collate_fn_temporal([val_data[0]])
            data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            images = data.pop('img')
            model.eval()
            os.environ['eval'] = 'true'
            with torch.no_grad():
                evaluated = model(imgs=images, metas=data)
            assert len(evaluated['future_occ_grids']) == 6
            records['eval_boundary'] = dict(future_grids=6, online_cache_entries=len(model.memory))
            del val_data, batch, data, images, evaluated
            model.train()
            model.set_epoch(10)
            os.environ['eval'] = 'false'
            assert not model.memory
        if index % 10 == 9:
            print('STABILITY', index + 1, row['after_cleanup'], flush=True)
            save()
    records['optimizer_changed_weights'] = not torch.equal(first_weight, model.reg_branch[-1].weight.detach().cpu())
    assert records['optimizer_changed_weights']
    settled = [row['after_cleanup']['allocated_mib'] for row in records['stability'] if row['iteration'] % 2 == 1]
    records['stability_summary'] = dict(steps=args.stability_steps, optimizer_steps=args.stability_steps // 2,
                                       min_allocated_mib=min(settled), max_allocated_mib=max(settled),
                                       allocated_spread_mib=max(settled) - min(settled),
                                       final=memory())
    assert max(settled) - min(settled) < 32, records['stability_summary']
    assert records['stability'][-1]['after_cleanup']['allocation_retries'] == 0
    records['status'] = 'passed'
    save()
    print('PASSED', records['stability_summary'], flush=True)


if __name__ == '__main__':
    main()
