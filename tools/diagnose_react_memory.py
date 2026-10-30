"""Read-only React training memory audit with allocation stacks and saved shapes."""

import argparse
from collections import defaultdict
import gc
import json
import logging
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.optim import build_optim_wrapper
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal, move_batch_to_device
from loss import OPENOCC_LOSS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/img_point_lite_react/img_voxel_react_occ3d_fast_v3.py')
    parser.add_argument('--sample', type=int, default=128)
    parser.add_argument('--history', type=int, default=9)
    parser.add_argument('--out', type=Path, default=Path('work_dirs/react_execution_optimization_v3/memory_audit'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    torch.manual_seed(928)
    torch.backends.cudnn.benchmark = True
    cfg = Config.fromfile(args.config)
    cfg.val_dataset_config.num_lidar_history = args.history
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    batch = move_batch_to_device(custom_collate_fn_temporal([dataset[args.sample]]), 'cuda')
    imgs = batch.pop('img')
    network = build_segmentor(cfg.model).cuda().train()
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    optimizer = build_optim_wrapper(network, cfg.optimizer)
    for group in optimizer.param_groups:
        group['lr'] = 0.

    def forward_loss():
        output = network(imgs=imgs, metas=batch)
        values = dict(metas=batch, global_iter=0)
        values.update({name: output[key] for name, key in cfg.loss_input_convertion.items()})
        return output, objective(values)[0]

    for _ in range(2):
        output, loss = forward_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del output, loss
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    phase, stack, marks, saved = ['start'], [], [], {}

    def mark(label):
        marks.append(dict(label=label, allocated_mib=torch.cuda.memory_allocated()/2**20,
                          interval_peak_mib=torch.cuda.max_memory_allocated()/2**20))
        torch.cuda.reset_peak_memory_stats()

    handles = []
    for name, module in [('img_backbone',network.img_backbone), ('img_neck',network.img_neck),
                         ('lifter',network.lifter), ('voxel_backbone',network.lifter.lidar_backbone),
                         ('encoder',network.encoder), ('head',network.head), ('loss',objective)]:
        def pre(module, inputs, name=name):
            mark(phase[0] + '/' + name + '/enter')
            stack.append(name)
        def post(module, inputs, output, name=name):
            mark(phase[0] + '/' + name + '/exit')
            stack.pop()
        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]

    def pack(tensor):
        if tensor.is_cuda:
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            item = saved.setdefault(key, dict(bytes=storage.nbytes(), shapes=[], groups=[]))
            shape = list(tensor.shape)
            if shape not in item['shapes']:
                item['shapes'].append(shape)
            group = phase[0] + '/' + (stack[-1] if stack else 'other')
            if group not in item['groups']:
                item['groups'].append(group)
        return tensor

    persistent = {}
    for name, tensor in list(network.named_parameters()) + list(network.named_buffers()):
        if tensor.is_cuda:
            persistent[tensor.untyped_storage().data_ptr()] = 'model/' + name
    optimizer_bytes = 0
    for state in optimizer.optimizer.state.values():
        for name, tensor in state.items():
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                persistent[tensor.untyped_storage().data_ptr()] = 'optimizer/' + name
                optimizer_bytes += tensor.numel()*tensor.element_size()
    parameter_bytes = sum(p.numel()*p.element_size() for p in network.parameters())

    torch.cuda.memory._record_memory_history(enabled='all', context='all', stacks='python', max_entries=100000)
    initial = torch.cuda.memory._snapshot()
    mark('baseline')
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        phase[0] = 'forward'
        output, loss = forward_loss()
        mark('after_loss')
        phase[0] = 'backward'
        loss.backward()
        mark('after_backward')
        gradient_bytes = sum(p.grad.numel()*p.grad.element_size()
                             for p in network.parameters() if p.grad is not None)
        phase[0] = 'optimizer'
        torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_max_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mark('after_optimizer')
    torch.cuda.synchronize()
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    for handle in handles:
        handle.remove()
    with (args.out/'snapshot.pickle').open('wb') as stream:
        pickle.dump(snapshot, stream)
    (args.out/'saved_storages.json').write_text(json.dumps(saved, indent=2))

    # Reconstruct allocator-live blocks at the high-water mark. Segment cache
    # allocations/frees are deliberately excluded from allocated-byte totals.
    live = {block['address']: dict(size=block['size'], frames=block.get('frames', []))
            for segment in initial['segments'] for block in segment['blocks']
            if block['state'] == 'active_allocated'}
    current = sum(item['size'] for item in live.values())
    peak, at_peak = current, dict(live)
    for entry in snapshot['device_traces'][torch.cuda.current_device()]:
        if entry['action'] == 'alloc':
            live[entry['addr']] = dict(size=entry['size'], frames=entry.get('frames', []))
            current += entry['size']
            if current > peak:
                peak, at_peak = current, dict(live)
        elif entry['action'] == 'free_requested':
            item = live.pop(entry['addr'], None)
            if item:
                current -= item['size']
    groups = defaultdict(int)
    blocks = []
    for address, item in sorted(at_peak.items(), key=lambda pair: pair[1]['size'], reverse=True):
        meta = saved.get(address, {})
        frames = [f for f in item['frames'] if any(
            text in f.get('filename','') for text in ('/model/', '/loss/', '/spconv/', '/mmseg/'))]
        label = persistent.get(address) or (meta.get('groups') or ['temporary/unknown'])[0]
        groups[label] += item['size']
        blocks.append(dict(mib=item['size']/2**20, label=label, shapes=meta.get('shapes', []),
                           frames=frames[-8:]))
    report = dict(config=args.config, sample=args.sample, history=args.history,
                  parameter_mib=parameter_bytes/2**20, optimizer_state_mib=optimizer_bytes/2**20,
                  parameter_gradient_mib=gradient_bytes/2**20,
                  measured_peak_mib=max(row['interval_peak_mib'] for row in marks),
                  reconstructed_peak_mib=peak/2**20, marks=marks,
                  live_at_peak_by_saved_origin_mib={k:v/2**20 for k,v in sorted(
                      groups.items(), key=lambda pair: pair[1], reverse=True)},
                  largest_blocks_at_peak=blocks[:35],
                  note='Use measured_peak_mib and marks for phase accounting. Allocator replay and saved-origin labels are diagnostic; storage addresses can be reused between allocations.')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'largest_blocks_at_peak'}, indent=2))
    print(json.dumps(blocks[:12], indent=2))


if __name__ == '__main__':
    main()
