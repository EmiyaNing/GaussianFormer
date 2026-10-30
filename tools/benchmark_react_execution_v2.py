"""React execution ablations: DAF microbenchmark and complete local steps.

Complete steps include DataLoader wait, H2D, model, all CE/Lovasz outputs,
backward, gradient clipping and AdamW. Use validation scenes for repeatable
input augmentation; no DDP, scheduler, logging or checkpoint I/O is timed.
AdamW has lr=0 so both paths retain the same random model weights.
"""

import argparse
import gc
import json
import logging
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset
from mmengine import Config
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import OPENOCC_DATASET, loader_worker_options
from dataset.utils import custom_collate_fn_temporal, move_batch_to_device, _OCC_GRID_CACHE
from loss import OPENOCC_LOSS
from model.encoder.gaussian_encoder.deformable_module import DeformableFeatureAggregation
from model.encoder.gaussian_encoder.ops import (
    DeformableAggregationFunction as DAF,
    DeformableAggregationSumFunction as DAFSum,
)


def summarize(rows):
    result = {name: statistics.mean(row[name] for row in rows)
              for name in rows[0] if isinstance(rows[0][name], (float, int))}
    for name in ('peak_mib', 'incremental_peak_mib'):
        if name in result:
            result['mean_' + name] = result[name]
            result[name] = max(row[name] for row in rows)
    return result


def daf_benchmark(args):
    torch.manual_seed(928)
    n, k, c, cameras, groups = args.queries, 9, 128, 6, 4
    sizes = [(108, 200), (54, 100), (27, 50), (14, 25)]
    starts = [0]
    for h, w in sizes[:-1]:
        starts.append(starts[-1] + h * w)
    shape = torch.tensor(sizes, device='cuda', dtype=torch.int32)
    starts = torch.tensor(starts, device='cuda', dtype=torch.int32)
    feature = torch.randn(1, cameras, sum(h * w for h, w in sizes), c,
                          device='cuda', requires_grad=True)
    location = torch.rand(1, n * k, cameras, 2, device='cuda')
    location[torch.rand(1, n * k, cameras, device='cuda') > .2] = -1
    location.requires_grad_()
    weights = (torch.rand(1, n * k, cameras, 4, groups, device='cuda') /
               (k * cameras * 4)).requires_grad_()
    probe = torch.randn(1, n, c, device='cuda')
    rows = []
    for training in (False, True):
        for fused in (False, True):
            for index in range(args.warmup + args.steps):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated_before = torch.cuda.memory_allocated()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                wall = time.perf_counter()
                start.record()
                with torch.set_grad_enabled(training):
                    if fused:
                        output = DAFSum.apply(feature, shape, starts, location, weights, k)
                    else:
                        output = DAF.apply(feature, shape, starts, location, weights)
                        output = output.reshape(1, n, k, c).sum(2)
                    if training:
                        gradients = torch.autograd.grad(output, (feature, location, weights), probe)
                        del gradients
                end.record()
                torch.cuda.synchronize()
                if index >= args.warmup:
                    rows.append(dict(training=training, fused=fused,
                                     wall_ms=(time.perf_counter() - wall) * 1000,
                                     stream_ms=start.elapsed_time(end),
                                     peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                                     incremental_peak_mib=(torch.cuda.max_memory_allocated() -
                                                           allocated_before) / 2**20))
                del output
    result = {}
    for training in (False, True):
        group = {('fused' if fused else 'original'): summarize(
            [row for row in rows if row['training'] == training and row['fused'] == fused])
            for fused in (False, True)}
        group['speedup'] = group['original']['wall_ms'] / group['fused']['wall_ms']
        result['forward_backward' if training else 'forward'] = group
    return dict(queries=n, points_per_query=k, channels=c, valid_camera_fraction=.2,
                summary=result, records=rows)


class ModuleTimer:
    """CUDA stream intervals (including idle gaps) plus host call durations."""
    def __init__(self, network):
        self.records = {}
        self.handles = []
        self.phase = 'forward'
        modules = [(name, getattr(network, name)) for name in
                   ('img_backbone', 'img_neck', 'lifter', 'encoder', 'head')]
        modules += [(f'daf_{index}', layer) for index, layer in enumerate(
            layer for layer in network.modules() if isinstance(layer, DeformableFeatureAggregation))]
        for name, module in modules:
            def before(module, inputs, name=name):
                if self.phase != 'forward':
                    return
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                self.records.setdefault(name, []).append([start, None, time.perf_counter(), None])

            def after(module, inputs, output, name=name):
                if self.phase != 'forward':
                    return
                record = self.records[name][-1]
                record[3] = (time.perf_counter() - record[2]) * 1000
                record[1] = torch.cuda.Event(enable_timing=True)
                record[1].record()

            self.handles.extend([module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after)])

    def values(self):
        result = {}
        for name, calls in self.records.items():
            result[name + '_stream_ms'] = sum(start.elapsed_time(end) for start, end, _, _ in calls)
            result[name + '_host_ms'] = sum(host for _, _, _, host in calls)
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def pipeline_benchmark(args):
    configs = {name: Config.fromfile('config/img_point_lite_react/' + filename) for name, filename in (
        ('fast1', 'img_voxel_react_occ3d_fast.py'),
        ('pipeline_only', 'img_voxel_react_occ3d_fast_v2.py'),
        ('fast2', 'img_voxel_react_occ3d_fast_v2.py'),
        ('fast3', 'img_voxel_react_occ3d_fast_v3.py'),
        ('fast4_python', 'img_voxel_react_occ3d_fast_v4.py'),
        ('fast4', 'img_voxel_react_occ3d_fast_v4.py'),
        ('fast4_ffn', 'img_voxel_react_occ3d_fast_v4_mem.py'),
        ('fast4_mem', 'img_voxel_react_occ3d_fast_v4_mem.py'),
        ('fast5_image', 'img_voxel_react_occ3d_fast_v5_image.py'),
        ('fast5_workers2', 'img_voxel_react_occ3d_fast_v5.py'),
        ('fast5', 'img_voxel_react_occ3d_fast_v5.py'),
        ('fast5_fused', 'img_voxel_react_occ3d_fast_v5_fused.py'),
        ('fast5_fp32', 'img_voxel_react_occ3d_fast_v5_fp32.py'),
        ('fast5_mem', 'img_voxel_react_occ3d_fast_v5_mem.py'))}
    configs['fast5_workers2'].train_loader.num_workers = 2
    configs['fast4_python'].model.encoder.deformable_model.fused_weight_softmax = False
    configs['fast4_ffn'].model.encoder.checkpoint_deformable = False
    torch.manual_seed(928)
    tiled_paths = {'fast3', 'fast4_python', 'fast4', 'fast4_ffn', 'fast4_mem',
                   'fast5_image', 'fast5_workers2', 'fast5', 'fast5_fused', 'fast5_fp32', 'fast5_mem'}
    with_tiled = bool(tiled_paths.intersection(args.paths.split(',')))
    network = build_segmentor(configs['fast3' if with_tiled else 'fast2'].model).cuda()
    initial = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    reference_model = build_segmentor(configs['fast1'].model)
    reference_model.load_state_dict(initial, strict=True)
    del reference_model, network
    gc.collect()
    torch.cuda.empty_cache()
    objective = OPENOCC_LOSS.build(configs['fast2'].loss).cuda()
    records = []
    for training in ([True] if args.phase == 'train' else
                     [False] if args.phase == 'eval' else [False, True]):
        for path in args.paths.split(','):
            config = configs[path]
            if args.lidar_history is not None:
                config.val_dataset_config.num_lidar_history = args.lidar_history
            # Build the requested topology (fixed BN is a subclass with the
            # same state keys). Initialization/startup is outside measurements.
            network = build_segmentor(config.model).cuda()
            network.load_state_dict(initial, strict=True)
            network.train(training)
            network.encoder.checkpoint_ffn = config.model.encoder.get('checkpoint_ffn', False)
            network.encoder.checkpoint_deformable = config.model.encoder.get('checkpoint_deformable', False)
            for layer in network.modules():
                if isinstance(layer, DeformableFeatureAggregation):
                    layer.fuse_point_sum = path not in ('fast1', 'pipeline_only')
                    for name in ('batched_projection', 'factor_camera_weights',
                                 'compact_weight_mask', 'fused_weight_softmax',
                                 'fuse_normalized_sampling'):
                        setattr(layer, name, config.model.encoder.deformable_model.get(name, False))
            optimizer = build_optim_wrapper(network, config.optimizer) if training else None
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group['lr'] = 0.
            dataset = OPENOCC_DATASET.build(config.val_dataset_config)
            indices = [int(index) % len(dataset) for index in args.samples.split(',')]
            indices = [indices[index % len(indices)] for index in range(args.warmup + args.steps)]
            loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False,
                                collate_fn=custom_collate_fn_temporal,
                                **loader_worker_options(config.train_loader))
            timer = ModuleTimer(network)
            iterator_start = time.perf_counter()
            iterator = iter(loader)
            iterator_creation_ms = (time.perf_counter() - iterator_start) * 1000
            for index in range(args.warmup + args.steps):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timer.records = {}
                timer.phase = 'forward'
                wall = time.perf_counter()
                batch = next(iterator)
                wait_ms = (time.perf_counter() - wall) * 1000
                h2d_start, h2d_end, forward_end, loss_end, backward_end, optimizer_end = [
                    torch.cuda.Event(enable_timing=True) for _ in range(6)]
                h2d_start.record()
                batch = move_batch_to_device(batch, 'cuda', config.get('non_blocking_transfer', False))
                images = batch.pop('img')
                h2d_end.record()
                torch.manual_seed(928 + index)
                with torch.set_grad_enabled(training):
                    result = network(imgs=images, metas=batch)
                    forward_end.record()
                    if training:
                        inputs = dict(metas=batch, global_iter=index)
                        inputs.update({name: result[key] for name, key in
                                       config.loss_input_convertion.items()})
                        loss, _ = objective(inputs)
                        loss_end.record()
                        timer.phase = 'backward'
                        loss.backward()
                        backward_end.record()
                        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_max_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_end.record()
                    else:
                        optimizer_end.record()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - wall) * 1000
                row = dict(path=path, training=training, sample_index=indices[index],
                           total_ms=elapsed_ms, data_wait_ms=wait_ms,
                           h2d_ms=h2d_start.elapsed_time(h2d_end),
                           forward_ms=h2d_end.elapsed_time(forward_end),
                           peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                           gaussian_count=int(result['gaussians'][-1].means.shape[1]))
                row.update(timer.values())
                if training:
                    row.update(loss_ms=forward_end.elapsed_time(loss_end),
                               backward_ms=loss_end.elapsed_time(backward_end),
                               optimizer_clip_ms=backward_end.elapsed_time(optimizer_end),
                               loss=float(loss.detach()), grad_norm=float(norm))
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError('Non-finite training loss/gradient norm')
                    del loss, inputs
                if index >= args.warmup:
                    records.append(row)
                del result, images, batch
            timer.close()
            print(json.dumps(dict(path=path, training=training,
                                  iterator_creation_ms=iterator_creation_ms,
                                  summary=summarize([r for r in records if r['path'] == path
                                                     and r['training'] == training]))), flush=True)
            # Release native per-worker voxel buffers before the next ablation.
            if hasattr(iterator, '_shutdown_workers'):
                iterator._shutdown_workers()
            del iterator, loader, dataset, optimizer
            network.zero_grad(set_to_none=True)
            network.lifter._lidar_processor = None
            del network
            _OCC_GRID_CACHE.clear()
            gc.collect()
            torch.cuda.empty_cache()
    summary = {}
    for training in sorted({row['training'] for row in records}):
        selected = [row for row in records if row['training'] == training]
        group = {path: summarize([row for row in selected if row['path'] == path])
                 for path in args.paths.split(',')}
        if 'fast1' in group and 'fast2' in group:
            group['total_speedup'] = group['fast1']['total_ms'] / group['fast2']['total_ms']
        if 'fast2' in group and 'fast3' in group:
            group['round_three_speedup'] = group['fast2']['total_ms'] / group['fast3']['total_ms']
        if 'fast3' in group:
            for path in ('fast4_python', 'fast4', 'fast4_ffn', 'fast4_mem'):
                if path in group:
                    group[path + '_speedup'] = group['fast3']['total_ms'] / group[path]['total_ms']
        summary['train_step' if training else 'inference_step'] = group
    return dict(summary=summary, records=records,
                scope=__doc__, dataset_split='val', strict_checkpoint_compatible=True,
                worker_startup_excluded=True, random_weights=True, optimizer_lr=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['daf', 'train', 'eval', 'all'], default='all')
    parser.add_argument('--paths', default='fast1,pipeline_only,fast2')
    parser.add_argument('--samples', default='128,256')
    parser.add_argument('--lidar-history', type=int)
    parser.add_argument('--queries', type=int, default=60000)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error('Use at least one warmup and one measured step')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.backends.cudnn.benchmark = True
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    torch.cuda.set_per_process_memory_fraction(.85)
    report = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cpu_threads=torch.get_num_threads(), arguments=vars(args).copy())
    report['arguments']['out'] = str(args.out)
    if args.phase in ('daf', 'all'):
        report['daf'] = daf_benchmark(args)
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(report['daf']['summary']), flush=True)
    if args.phase != 'daf':
        report['pipeline'] = pipeline_benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
