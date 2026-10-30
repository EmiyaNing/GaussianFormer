"""Paired real-sample React execution benchmark and forward/gradient audit.

One set of weights is used for both paths. Full image resolution, all three
supervised outputs, CE/Lovasz and all Gaussians are retained. Timings include
model-side CPU voxelization; dataset loading/H2D and optimizer steps are not
included. The optional history override is applied equally to both paths.
"""

import argparse
import gc
import json
import math
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
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from model.encoder.gaussian_encoder.utils import GaussianPrediction


def set_execution(network, enabled):
    encoder, head = network.encoder, network.head
    encoder.cache_image_features = enabled
    encoder.skip_unused_anchor_embed = enabled
    encoder.terminal_densify = enabled
    head.analytic_covariance = enabled
    head.separate_empty = enabled
    head.incremental_densify = enabled
    head.cache_render_grid = enabled
    head.aggregator.validate_inputs = not enabled


def loss_inputs(config, result, batch):
    values = {'metas': batch}
    values.update({name: result[key] for name, key in config.loss_input_convertion.items()})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--py-config', default=
                        'config/img_point_lite_react/img_voxel_react_occ3d_fast.py')
    parser.add_argument('--samples', default='128,256')
    parser.add_argument('--lidar-history', type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--memory-fraction', type=float, default=.8)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('The paired integration benchmark requires CUDA')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(928)
    config = Config.fromfile(args.py_config)
    if args.lidar_history is not None:
        config.val_dataset_config.num_lidar_history = args.lidar_history
    dataset = OPENOCC_DATASET.build(config.val_dataset_config)
    samples = [dataset[int(index) % len(dataset)] for index in args.samples.split(',')]
    network = build_segmentor(config.model).cuda()
    objective = OPENOCC_LOSS.build(config.loss).cuda()
    initial_state = {name: value.detach().cpu().clone()
                     for name, value in network.state_dict().items()}
    # The optimized path introduces no checkpoint parameters or persistent
    # buffers: strict loading is used throughout this paired experiment.
    baseline_config = Config.fromfile('config/img_point_lite_react/img_voxel_react_occ3d.py')
    baseline_model = build_segmentor(baseline_config.model)
    baseline_model.load_state_dict(initial_state, strict=True)
    del baseline_model

    def to_device(sample):
        batch = custom_collate_fn_temporal([sample])
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch.pop('img'), batch

    audit = {}
    images, batch = to_device(samples[0])
    reference_logits = reference_gradients = None
    reference_anchors = None
    frozen_gaussians = None

    def canonical_voxels(module, inputs, output):
        # spconv hash insertion can permute rows between forwards. A fixed
        # RNG seed would then attach dropout masks/top-k tie choices to
        # different physical anchors. Canonicalize ONLY for the equality
        # audit; remove this hook before timing either production path.
        sparse = output['out_tensor']
        indices = sparse.indices.long()
        z, y, x = sparse.spatial_shape
        keys = ((indices[:, 0] * z + indices[:, 1]) * y + indices[:, 2]) * x + indices[:, 3]
        order = keys.argsort()
        output['out_tensor'] = type(sparse)(
            features=sparse.features[order], indices=sparse.indices[order],
            spatial_shape=sparse.spatial_shape, batch_size=sparse.batch_size)
        return output

    audit_hook = network.lifter.lidar_backbone.register_forward_hook(canonical_voxels)
    for audit_name, enabled in (('baseline', False), ('baseline_repeat', False), ('fast', True)):
        network.load_state_dict(initial_state, strict=True)
        network.train()
        network.zero_grad(set_to_none=True)
        set_execution(network, enabled)
        torch.manual_seed(928)
        render_sizes = []
        embedding_calls = []
        hooks = [network.head.aggregator.register_forward_pre_hook(
            lambda module, inputs: render_sizes.append(inputs[1].shape[1])),
                 network.encoder.anchor_encoder.register_forward_hook(
            lambda *unused: embedding_calls.append(1))]
        result = network(imgs=images, metas=batch)
        for hook in hooks:
            hook.remove()
        value, _ = objective(loss_inputs(config, result, batch))
        value.backward()
        logits = [tensor.detach().cpu() for tensor in result['pred_occ']]
        gradients = {name: parameter.grad.detach().cpu().clone()
                     for name, parameter in network.named_parameters()
                     if parameter.requires_grad and parameter.grad is not None}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        entry = dict(loss=float(value.detach()), render_gaussian_counts=render_sizes,
                     anchor_embedding_calls=len(embedding_calls),
                     gaussian_counts=[g.means.shape[1] for g in result['gaussians']])
        if audit_name == 'baseline':
            reference_logits, reference_gradients = logits, gradients
            reference_anchors = result['anchor_init'].detach().cpu()
            frozen_gaussians = [GaussianPrediction(*(value.detach().cpu() for value in g[:5]))
                                for g in result['gaussians']]
        else:
            entry['initial_anchor_max_abs_error'] = float(
                (reference_anchors - result['anchor_init'].detach().cpu()).abs().max())
            entry['logit_max_abs_error'] = [float((a - b).abs().max())
                                            for a, b in zip(logits, reference_logits)]
            entry['logit_mean_abs_error'] = [float((a - b).abs().mean())
                                             for a, b in zip(logits, reference_logits)]
            print('audit errors', entry, flush=True)
            assert gradients.keys() == reference_gradients.keys()
            relative = {}
            for name, gradient in gradients.items():
                reference = reference_gradients[name]
                relative[name] = float((gradient - reference).norm() /
                                        reference.norm().clamp_min(1e-8))
            entry['max_parameter_gradient_relative_l2'] = max(relative.values())
            numerator = sum(float((gradient - reference_gradients[name]).norm()) ** 2
                            for name, gradient in gradients.items())
            denominator = sum(float(gradient.norm()) ** 2
                              for gradient in reference_gradients.values())
            entry['global_parameter_gradient_relative_l2'] = math.sqrt(numerator / max(denominator, 1e-30))
            entry['largest_gradient_differences'] = sorted(
                relative.items(), key=lambda item: item[1], reverse=True)[:8]
            entry['pointwise_close_to_first_baseline'] = all(
                torch.allclose(a, b, atol=2e-3, rtol=2e-3)
                for a, b in zip(logits, reference_logits))
        audit[audit_name] = entry
        del result, value, logits, gradients
    audit_hook.remove()
    audit['pointwise_network_equivalence_verified'] = (
        audit['baseline_repeat']['pointwise_close_to_first_baseline'] and
        audit['fast']['pointwise_close_to_first_baseline'] and
        audit['fast']['max_parameter_gradient_relative_l2'] < .02)
    # Report baseline-to-baseline variability rather than attributing it to
    # the optimization or silently relaxing an equality tolerance. Component
    # tests independently check deterministic inputs and parameter gradients.
    # Isolate the modified rendering path from the original sparse encoder's
    # repeatability. Use real full-scene Gaussian tensors, with identical
    # parent/child leaves, and check every Gaussian-field gradient.
    def leaf_gaussian(value):
        return GaussianPrediction(*(tensor.cuda().detach().requires_grad_() for tensor in value[:5]))

    first, parent = [leaf_gaussian(g) for g in frozen_gaussians[:2]]
    parent_count = parent.means.shape[1]
    child = leaf_gaussian(GaussianPrediction(*(tensor[:, parent_count:]
                                               for tensor in frozen_gaussians[-1][:5])))
    final = GaussianPrediction(*(torch.cat([a, b], 1) for a, b in zip(parent[:5], child[:5])))
    frozen_states = [dict(gaussian=first), dict(gaussian=parent),
                     dict(gaussian=final, delta_gaussian=child, densify_parent_index=1)]
    variables = first[:5] + parent[:5] + child[:5] + (network.head.empty_scalar,)
    frozen_reference = frozen_gradients = None
    for enabled in (False, True):
        set_execution(network, enabled)
        output = network.head(frozen_states, metas=batch)
        tensors = output['pred_occ']
        probe = sum(tensor.square().mean() for tensor in tensors)
        gradients = [g.detach().cpu() for g in torch.autograd.grad(
            probe, variables, retain_graph=not enabled)]
        logits = [tensor.detach().cpu() for tensor in tensors]
        if not enabled:
            frozen_reference, frozen_gradients = logits, gradients
        else:
            for a, b in zip(logits, frozen_reference):
                torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)
            numerator = sum(float((a - b).norm()) ** 2 for a, b in zip(gradients, frozen_gradients))
            denominator = sum(float(b.norm()) ** 2 for b in frozen_gradients)
            relative = math.sqrt(numerator / max(denominator, 1e-30))
            assert relative < .002, relative
            audit['fixed_real_gaussians'] = dict(
                passed=True, logit_max_abs_error=[float((a-b).abs().max())
                                                  for a, b in zip(logits, frozen_reference)],
                all_gaussian_and_empty_gradients_relative_l2=relative,
                field_order='first/parent/child: means,scales,rotations,opacity,semantics; empty_scalar')
        del output, tensors, probe, gradients, logits
    del (reference_logits, reference_gradients, reference_anchors, frozen_gaussians,
         frozen_states, variables, first, parent, child, final,
         frozen_reference, frozen_gradients, images, batch)
    network.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    records = []
    for training in (False, True):
        network.load_state_dict(initial_state, strict=True)
        network.train(training)
        for sample_index, sample in enumerate(samples):
            images, batch = to_device(sample)

            def run(enabled, timed):
                set_execution(network, enabled)
                network.zero_grad(set_to_none=True)
                torch.manual_seed(1000 + sample_index)
                if timed:
                    torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.set_grad_enabled(training):
                    result = network(imgs=images, metas=batch)
                    if training:
                        value, _ = objective(loss_inputs(config, result, batch))
                        value.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated() / 2**20
                if timed:
                    records.append(dict(mode='train_fwd_loss_bwd' if training else 'eval',
                                        path='fast' if enabled else 'baseline',
                                        sample_index=int(args.samples.split(',')[sample_index]),
                                        seconds=elapsed, peak_allocated_mib=peak))
                del result

            for _ in range(args.warmup):
                run(False, False)
                run(True, False)
            for repeat in range(args.repeats):
                # Alternate order to reduce bias from warm caches and clocks.
                for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
                    run(enabled, True)
            del images, batch
            network.zero_grad(set_to_none=True)
        print('completed', 'training' if training else 'inference', flush=True)

    summary = {}
    for mode in ('eval', 'train_fwd_loss_bwd'):
        group = {}
        for path in ('baseline', 'fast'):
            selected = [row for row in records if row['mode'] == mode and row['path'] == path]
            group[path] = dict(mean_ms=statistics.mean(row['seconds'] for row in selected) * 1000,
                               median_ms=statistics.median(row['seconds'] for row in selected) * 1000,
                               peak_allocated_mib=max(row['peak_allocated_mib'] for row in selected))
        group['speedup'] = group['baseline']['mean_ms'] / group['fast']['mean_ms']
        summary[mode] = group
    report = dict(config=args.py_config, device=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
                  cpu_interop_threads=torch.get_num_interop_threads(),
                  image_shape=list(config.input_shape), lidar_history=
                  config.val_dataset_config.get('num_lidar_history', 0),
                  sample_indices=args.samples, warmup=args.warmup, repeats=args.repeats,
                  scope='model forward including CPU voxelization; training includes CE/Lovasz and backward; no optimizer',
                  random_weights=True, correctness=audit, summary=summary, records=records)
    report['correctness_audit_canonical_voxel_order'] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['correctness'], indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
