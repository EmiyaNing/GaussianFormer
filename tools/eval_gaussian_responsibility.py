"""Full mini validation with equal, training-only calibration for A0..A3."""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.gaussian_responsibility_experiment import RUN, CONFIGS, checkpoint, initialize, read, write, sha
from tools.empty_decision_optimization import build_loader, to_cuda, seed_all
from tools.empty_mechanism_stage1 import network_output
from tools.eval_gf2_probability_decomposition import extract_fields, hist_add, fit_global, tune
from tools.calibrate_empty_mechanism_stage2 import EDGES, NB, fitting_thresholds


RULES = ['native', 'hierarchical', 'global_calibrated', 'class_calibrated']


def build(name):
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model  # noqa: F401
    seed_all(928)
    config = Config.fromfile(CONFIGS[name])
    net = build_segmentor(config.model).cuda().eval().requires_grad_(False)
    state = torch.load(checkpoint(name), map_location='cpu', weights_only=False)
    assert state['epoch'] == 5 and state['global_iter'] == 20480
    net.load_state_dict(state['state_dict'], strict=True)
    return net


def confusion(y, pred):
    return np.bincount(y.astype(np.int64) * 18 + pred.astype(np.int64), minlength=324).reshape(18, 18)


def summarize_counts(counts):
    c = np.asarray(counts, np.float64)
    diag = np.diagonal(c, axis1=-2, axis2=-1)
    union = c.sum(-2) + c.sum(-1) - diag
    iou = np.divide(diag, union, out=np.zeros_like(diag), where=union > 0)
    occupied = c[..., :17, :].sum((-2, -1))
    tp = c[..., :17, :17].sum((-2, -1))
    fp = c[..., 17, :17].sum(-1)
    fn = c[..., :17, 17].sum(-1)
    correct = diag[..., :17].sum(-1)
    safe = lambda a, b: np.divide(a, b, out=np.zeros_like(np.asarray(a)), where=b > 0) * 100
    return dict(mIoU=iou[..., :17].mean(-1) * 100,
        IoU=safe(tp, tp + fp + fn), FNR=safe(fn, occupied),
        FPR=safe(fp, c[..., 17, :].sum(-1)),
        foreground_final_accuracy=safe(correct, occupied),
        foreground_conditional_accuracy=safe(correct, tp), per_class_IoU=iou[..., :17] * 100)


def plain(value):
    return {key: np.asarray(v).tolist() for key, v in value.items()}


def calibrate(name, net, protocol, destination):
    path = destination / 'selection.json'
    if path.exists():
        return read(path)
    histpath = destination / 'calibration_hist.npz'
    if not histpath.exists():
        if name == 'A0':
            existing = np.load(ROOT / 'work_dirs/gf2_probability_decomposition/evaluation/probability/calibration_hist.npz')
            hist = existing['hist'][-1].copy()
            indices = existing['indices']
        else:
            dataset, _, _ = build_loader('train')
            chosen = {}
            for index, (scene, _) in enumerate(dataset.keyframes):
                chosen.setdefault(scene, index)
            assert set(chosen) == set(protocol['fit_scenes']) | set(protocol['tune_scenes'])
            indices = [chosen[s] for s in sorted(chosen)]
            _, loader, _ = build_loader('train', indices)
            hist = np.zeros((2, 18, 17, NB), np.int64)
            fitting = set(protocol['fit_scenes'])
            with torch.inference_mode():
                for n, (index, batch) in enumerate(zip(indices, loader)):
                    batch = to_cuda(batch)
                    output = network_output(net, batch)
                    fields = extract_fields('probability', output, batch)
                    hist_add(hist[0 if dataset.keyframes[index][0] in fitting else 1], fields, 1.)
                    del batch, output, fields
                    if (n + 1) % 128 == 0:
                        print(name, 'calibration', n + 1, '/', len(indices), flush=True)
        np.savez_compressed(histpath, hist=hist, indices=indices,
                            checkpoint_sha256=np.array(sha(checkpoint(name))))
    stored = np.load(histpath)
    assert str(stored['checkpoint_sha256']) == sha(checkpoint(name))
    hist = stored['hist']
    global_cut, global_fit = fit_global(hist[0])
    global_cuts, global_tune, _ = tune(hist[1], np.full(17, global_cut))
    class_initial, boundaries = fitting_thresholds(hist[0])
    class_cuts, class_tune, _ = tune(hist[1], class_initial)
    result = dict(global_threshold=float(global_cuts[0]), global_fit=global_fit,
        global_tune=global_tune, class_thresholds=class_cuts.tolist(),
        class_fit_boundaries=boundaries, class_tune=class_tune,
        selection_data='523 fitting +175 tuning training scenes; validation excluded',
        checkpoint_sha256=sha(checkpoint(name)))
    write(path, result)
    return result


def evaluate(name):
    protocol = initialize()
    dest = RUN / 'evaluation' / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    start = time.monotonic()
    net = None if name == 'A0' else build(name)
    selection = calibrate(name, net, protocol, dest)
    dataset, loader, _ = build_loader('val')
    records, timing = [], []
    rank_correct = rank_total = 0
    for index, batch in enumerate(loader):
        if name == 'A0':
            a = dict(np.load(ROOT / f'work_dirs/gf2_probability_decomposition/evaluation/probability/frames/{index:04d}.npz'))
        else:
            batch = to_cuda(batch)
            with torch.inference_mode():
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); output = network_output(net, batch); end.record()
                end.synchronize()
                if index >= 16:
                    timing.append(begin.elapsed_time(end))
                a = extract_fields('probability', output, batch)
            del output, batch
        y = a['y'].astype(np.int64)
        fg = a['q'].argmax(-1)
        predicates = dict(native=a['native'], hierarchical=np.where(a['o'] >= .5, fg, 17),
            global_calibrated=np.where(a['margin'] >= selection['global_threshold'], fg, 17),
            class_calibrated=np.where(a['margin'] >= np.asarray(selection['class_thresholds'])[fg], fg, 17))
        matrices = {r: confusion(y, pred).tolist() for r, pred in predicates.items()}
        foreground = y < 17
        correct = int((foreground & (fg == y)).sum())
        count = int(foreground.sum())
        rank_correct += correct; rank_total += count
        with np.errstate(divide='ignore', invalid='ignore'):
            conditional_nll = float(-np.log(np.maximum(a['q'][foreground, y[foreground]], 1e-6)).sum())
        scene, scene_frame = dataset.keyframes[index]
        records.append(dict(index=index, scene=scene, scene_frame=int(scene_frame),
            counts=matrices, foreground=count, rank_correct=correct,
            conditional_nll_sum=conditional_nll, total_voxels=len(y)))
        if (index + 1) % 128 == 0:
            print(name, 'validation', index + 1, '/ 1024', flush=True)
    assert len(records) == 1024
    scenes = sorted({r['scene'] for r in records})
    scene_index = {s: i for i, s in enumerate(scenes)}
    scene_counts = np.zeros((len(scenes), len(RULES), 18, 18), np.int64)
    scene_rank = np.zeros((len(scenes), 2), np.int64)
    for row in records:
        i = scene_index[row['scene']]
        for j, rule in enumerate(RULES):
            scene_counts[i, j] += np.array(row['counts'][rule], np.int64)
        scene_rank[i] += [row['rank_correct'], row['foreground']]
    np.savez_compressed(dest / 'scene_counts.npz', counts=scene_counts, rank=scene_rank,
                        scenes=np.array(scenes), rules=np.array(RULES))
    result = dict(model=name, rules={r: plain(summarize_counts(scene_counts[:, j].sum(0))) for j, r in enumerate(RULES)},
        rank_accuracy=100 * rank_correct / rank_total,
        conditional_NLL=sum(r['conditional_nll_sum'] for r in records) / rank_total,
        frames=1024, scenes=len(scenes), records=records,
        inference_mean_ms=float(np.mean(timing)) if timing else None,
        inference_timing='CUDA events around native forward on validation inputs; 16 warm-up frames; A0 cached, no timing',
        elapsed_seconds=time.monotonic() - start)
    write(dest / 'statistics.json', result)
    write(dest / 'complete.json', dict(status='complete', frames=1024, scenes=len(scenes),
        checkpoint_sha256=sha(checkpoint(name)), calibration='training scenes only'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=list(CONFIGS))
    evaluate(parser.parse_args().model)
