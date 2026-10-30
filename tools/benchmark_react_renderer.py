"""Freeze real React renderer inputs, audit gradients and benchmark backends."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal, move_batch_to_device
from model.head.react_local_aggregate import PreparedReactAggregator


def capture(args):
    config = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d_fast_v2.py')
    config.val_dataset_config.num_lidar_history = args.lidar_history
    torch.manual_seed(928)
    dataset = OPENOCC_DATASET.build(config.val_dataset_config)
    batch = custom_collate_fn_temporal([dataset[args.sample % len(dataset)]])
    batch = move_batch_to_device(batch, 'cuda')
    images = batch.pop('img')
    network = build_segmentor(config.model).cuda().train()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        network.load_state_dict(state.get('state_dict', state), strict=True)
    calls = []

    def save(module, inputs, kwargs):
        calls.append(dict(inputs=[x.detach().cpu() for x in inputs],
                          points_int=kwargs['points_int'].detach().cpu()))

    handle = network.head.aggregator.register_forward_pre_hook(save, with_kwargs=True)
    with torch.no_grad():
        network(imgs=images, metas=batch)
    handle.remove()
    if len(calls) != 3:
        raise RuntimeError(f'Expected first/parent/delta renderer inputs, got {len(calls)}')
    final = [calls[1]['inputs'][0]] + [torch.cat([a, b], 1)
        for a, b in zip(calls[1]['inputs'][1:], calls[2]['inputs'][1:])]
    calls.append(dict(inputs=final, points_int=calls[1]['points_int']))
    for name, call in zip(('first', 'parent', 'delta', 'final'), calls):
        call['name'] = name
    aggregate = network.head.aggregator
    settings = {key: getattr(aggregate, key) for key in
                ('H', 'W', 'D', 'scale_multiplier', 'grid_size')}
    settings['pc_min'] = aggregate.pc_min.cpu().flatten().tolist()
    pack = dict(settings=settings, calls=calls, sample=args.sample,
                lidar_history=args.lidar_history, checkpoint=args.checkpoint,
                random_weights=not bool(args.checkpoint))
    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, args.inputs)
    return pack


def run(args, pack):
    backends = {}
    for name in args.backends.split(','):
        if name == 'legacy':
            backend = PreparedReactAggregator(**pack['settings'], validate_inputs=False)
        else:
            from model.head.react_tiled_aggregate import TiledReactAggregator
            backend = TiledReactAggregator(**pack['settings'], validate_inputs=False,
                                           tile_size=args.tile_size,
                                           radius_hint=args.radius_hint)
        backends[name] = backend.cuda()
    rows, audit, topology = [], {}, {}
    for call in pack['calls']:
        values = [value.cuda().detach().requires_grad_(index in (1, 2, 3, 5))
                  for index, value in enumerate(call['inputs'])]
        points_int = call['points_int'].cuda()
        variables = [values[index] for index in (1, 2, 3, 5)]
        torch.manual_seed(173)
        probe = torch.randn(values[0].shape[1], 18, device='cuda')
        reference = reference_gradients = None
        for name, backend in backends.items():
            output = backend(*values, points_int=points_int)
            gradients = torch.autograd.grad(output, variables, probe)
            if reference is None:
                reference = output.detach()
                reference_gradients = [g.detach() for g in gradients]
            else:
                difference = (output.detach() - reference).abs()
                relative = [float((a - b).norm() / b.norm().clamp_min(1e-20))
                            for a, b in zip(gradients, reference_gradients)]
                entry = dict(logit_max_abs=float(difference.max()),
                             logit_mean_abs=float(difference.mean()),
                             gradient_relative_l2=relative)
                torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)
                assert max(relative) < .002, entry
                audit[call['name'] + '/' + name] = entry
                del difference
            del output, gradients
        del reference, reference_gradients
        for training in (False, True):
            for repeat in range(args.warmup + args.repeats):
                order = list(backends.items())
                if repeat % 2:
                    order.reverse()
                for name, backend in order:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = torch.cuda.memory_allocated()
                    start = time.perf_counter()
                    profiling = ((args.profile_legacy and name == 'legacy') or
                                 args.profile_backend == name) and (
                                     training or not args.profile_training)
                    if profiling and repeat == args.warmup:
                        torch.cuda.cudart().cudaProfilerStart()
                    with torch.set_grad_enabled(training):
                        output = backend(*values, points_int=points_int)
                        if training:
                            gradients = torch.autograd.grad(output, variables, probe)
                            del gradients
                    torch.cuda.synchronize()
                    if profiling and repeat == args.warmup:
                        torch.cuda.cudart().cudaProfilerStop()
                    milliseconds = (time.perf_counter() - start) * 1000
                    if repeat >= args.warmup:
                        rows.append(dict(call=call['name'], backend=name, training=training,
                                         gaussian_count=values[1].shape[1], ms=milliseconds,
                                         peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                                         incremental_peak_mib=(torch.cuda.max_memory_allocated() - before) / 2**20))
                    del output
        if 'tiled' in backends:
            backend = backends['tiled']
            with torch.no_grad():
                prepared = backend._prepare_inputs(*values, points_int=points_int)
                centers, radii = prepared[3], prepared[6]
                capacity = backend.capacity_for(values[1].shape[1])
                from model.head.react_tiled_aggregate import extension
                offsets, ids = extension().tile_bins(centers, radii, backend.H,
                    backend.W, backend.D, backend.tile_size, capacity)
                limit = centers.new_tensor([backend.H, backend.W, backend.D])
                lo = torch.minimum((centers-radii).clamp_min(0), limit)
                hi = torch.minimum((centers+radii+1).clamp_min(0), limit)
                topology[call['name']] = dict(
                    voxel_pairs=int((hi-lo).prod(-1).sum()),
                    tile_pairs=int(offsets[-1]), capacity=capacity,
                    fallback_bins=int((offsets[1:] > capacity).sum()))
                del prepared, centers, radii, offsets, ids, limit, lo, hi
        print('completed', call['name'], flush=True)
        del values, points_int, variables, probe
        gc.collect()
        torch.cuda.empty_cache()
    summary = {}
    for row in rows:
        key = row['call'] + ('/forward_backward/' if row['training'] else '/forward/') + row['backend']
        selected = [r for r in rows if r['call'] == row['call'] and
                    r['training'] == row['training'] and r['backend'] == row['backend']]
        summary[key] = {name: statistics.mean(r[name] for r in selected)
                        for name in ('ms', 'incremental_peak_mib')}
    return dict(device=torch.cuda.get_device_name(), input_file=str(args.inputs),
                tile_size=args.tile_size, radius_hint=args.radius_hint,
                audit=audit, topology=topology, summary=summary, records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path,
                        default=Path('work_dirs/react_execution_optimization_v3/renderer_inputs.pt'))
    parser.add_argument('--out', type=Path,
                        default=Path('work_dirs/react_execution_optimization_v3/renderer_benchmark.json'))
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--profile-legacy', action='store_true')
    parser.add_argument('--profile-backend', choices=['legacy', 'tiled'])
    parser.add_argument('--profile-training', action='store_true')
    parser.add_argument('--backends', default='legacy,tiled')
    parser.add_argument('--sample', type=int, default=128)
    parser.add_argument('--lidar-history', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--tile-size', type=int, default=4)
    parser.add_argument('--radius-hint', type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    pack = (torch.load(args.inputs, map_location='cpu', weights_only=False)
            if args.inputs.exists() else capture(args))
    if args.capture_only:
        print('captured', [(x['name'], list(x['inputs'][1].shape)) for x in pack['calls']])
        return
    report = run(args, pack)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
