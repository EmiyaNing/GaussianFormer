"""Matched one-epoch fine-tuning, with atomic optimizer/RNG recovery."""
from pathlib import Path
import gc
import json
import math
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.all_layer_render_experiment import *


def train(arm):
    protocol = assert_protocol()
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    from torch.utils.data import DataLoader, Sampler
    folder = RUN / 'train' / arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        record = json.loads((folder / 'complete.json').read_text())
        assert sha(folder / 'epoch_1.pth') == record['checkpoint_sha256']
        return
    latest = folder / 'latest.pth'
    cfg, net, opt, objective, state = build(arm, latest if latest.exists() else None, True)
    completed = int(state.get('experiment_microbatches', 0))
    updates = int(state.get('experiment_updates', 0))
    bases = state.get('experiment_base_lrs', [g['lr'] for g in opt.param_groups])
    dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
    assert len(dataset) == cfg.expected_train_frames
    assert len({x[0] for x in dataset.keyframes}) == cfg.expected_train_scenes
    order = np.random.default_rng(cfg.seed).permutation(len(dataset)).tolist()
    order_sha = hashlib.sha256(np.asarray(order, np.int64).tobytes()).hexdigest()
    if completed:
        assert state['data_order_sha256'] == order_sha
        assert state['protocol_sha256'] == sha(RUN / 'protocol.json')
    trace_path = folder / 'trace.jsonl'
    if trace_path.exists():
        lines = trace_path.read_text().splitlines()
        kept = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row['microbatch'] <= completed:
                kept.append(line)
        if kept != lines:
            trace_path.rename(folder / f'trace_before_resume_{time.time_ns()}.jsonl')
            trace_path.write_text('\n'.join(kept) + ('\n' if kept else ''))
    baseline_rows = None
    if arm == 'all_layers':
        baseline_rows = [json.loads(line) for line in (RUN / 'train/baseline/trace.jsonl').read_text().splitlines()]
        assert len(baseline_rows) == cfg.expected_train_frames
    class FixedSampler(Sampler):
        def __iter__(self): return iter(order[completed:])
        def __len__(self): return len(order) - completed
    loader = DataLoader(TrainingFrames(dataset, cfg.seed), batch_size=1, sampler=FixedSampler(),
        num_workers=3, collate_fn=custom_collate_fn_temporal, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
        generator=torch.Generator().manual_seed(cfg.seed + 500000))
    if completed:
        torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        restore_numpy_rng(state['rng_numpy'])
        random.setstate(state['rng_random'])
    else:
        seed_all(cfg.seed * 1000000 + 200000)
    del state
    net.train()
    opt.zero_grad()
    total, acc = len(order), cfg.gradient_accumulation
    total_updates = math.ceil(total / acc)
    start, beginning = time.monotonic(), completed
    torch.cuda.reset_peak_memory_stats()
    def checkpoint(count):
        state = dict(state_dict=net.state_dict(), optimizer=opt.state_dict(), epoch=1,
            experiment_microbatches=count, experiment_updates=updates, experiment_base_lrs=bases,
            data_order_sha256=order_sha, protocol_sha256=sha(RUN / 'protocol.json'),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
            rng_numpy=numpy_rng_checkpoint(), rng_random=random.getstate())
        temporary = latest.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(latest)
    try:
        with trace_path.open('a', buffering=1) as trace:
            for i, batch in enumerate(loader, completed):
                if i % acc == 0:
                    warmup = min(1., (updates+1)/cfg.finetune_warmup_updates)
                    cosine = .5 * (1 + math.cos(math.pi * updates / max(total_updates-1, 1)))
                    factor = (.1+.9*warmup) * (cfg.finetune_min_ratio + (1-cfg.finetune_min_ratio)*cosine)
                    for group, base in zip(opt.param_groups, bases):
                        group['lr'] = base * factor
                digest = fingerprint(batch)
                if baseline_rows is not None:
                    reference = baseline_rows[i]
                    assert reference['index'] == order[i] and reference['input_sha256'] == digest
                batch = to_cuda(batch)
                out = forward(net, batch)
                inputs = loss_input(cfg, out, batch)
                inputs['global_iter'] = i
                value, details = objective(inputs)
                if not torch.isfinite(value):
                    raise FloatingPointError(f'Nonfinite loss: {arm}, frame {i+1}')
                (value / min(acc, total-(i//acc)*acc)).backward()
                row = dict(microbatch=i+1, index=order[i], input_sha256=digest, loss=float(value.detach()),
                           losses={k: float(v) for k, v in details.items()})
                if arm == 'all_layers':
                    terms = [row['losses'][f'AllLayerOccupancyLoss/layer_{j+1}'] for j in range(6)]
                    np.testing.assert_allclose(row['losses']['AllLayerOccupancyLoss'], sum(terms), rtol=2e-7)
                if (i+1) % acc == 0 or i+1 == total:
                    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
                    opt.step()
                    opt.zero_grad()
                    updates += 1
                    row.update(optimizer_update=updates, grad_norm=float(norm))
                trace.write(json.dumps(row) + '\n')
                count = i+1
                if count % 16 == 0 or count in (beginning+1, total):
                    torch.cuda.synchronize()
                    rate = (time.monotonic()-start)/(count-beginning)
                    progress = dict(status='running', phase='training', arm=arm, microbatches=count,
                        total=total, optimizer_updates=updates, seconds_per_frame=rate,
                        remaining_minutes=(total-count)*rate/60, loss=row['loss'],
                        lr=opt.param_groups[0]['lr'], peak_cuda_GB=torch.cuda.max_memory_allocated()/1e9)
                    write(folder / 'progress.json', progress)
                    write(RUN / 'status.json', progress)
                    print(json.dumps(progress), flush=True)
                if count % (cfg.save_every_updates*acc) == 0 or count == total:
                    checkpoint(count)
                del batch, out, inputs, value
        final = folder / 'epoch_1.pth'
        if final.exists():
            final.unlink()
        os.link(latest, final)
        assert_protocol()
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        assert len(rows) == total and [r['index'] for r in rows] == order
        assert np.isfinite([r['loss'] for r in rows]).all() and updates == total_updates == 2045
        status = dict(status='complete', arm=arm, frames=total, unique_frames=len(set(order)), scenes=698,
            epochs=1, optimizer_updates=updates, checkpoint=str(final), checkpoint_sha256=sha(final),
            data_order_sha256=order_sha, seconds_this_run=time.monotonic()-start,
            peak_cuda_GB=torch.cuda.max_memory_allocated()/1e9,
            paired_inputs_exact=baseline_rows is not None)
        write(folder / 'complete.json', status)
        write(folder / 'progress.json', status)
    finally:
        release(net)
        del net, opt, objective, loader
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm', choices=ARMS)
    train(p.parse_args().arm)
