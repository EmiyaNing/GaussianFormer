"""Frozen Gaussian empty-decision experiment; all selection uses training scenes."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tools.center_support_common import write_json, kernel_weights
from tools.diagnose_opus_v1_gaussian import file_sha256

RUN = ROOT / 'work_dirs/empty_decision_optimization'
REPORT = ROOT / 'analysis/empty_decision_optimization'
CONFIG = 'config/center_support_contribution1/a0_original.py'
CHECKPOINT = 'work_dirs/center_support_contribution1/train/a0_original_seed928/epoch_5.pth'
FEATURES = ['margin', 'x', 'y', 'z', 'range_xy', 'fg_top_gap', 'fg_entropy', 'log_fg_total', 'log_support']
REGIONS = ['camera', 'camera_complement', 'all_valid']
SEEDS = [928, 929, 930]
BAND = 2.0
SAMPLE_PER_STRATUM = 256
BINS, SCORE_MIN, SCORE_MAX = 1024, -16., 16.
BUDGETS = [0., .005, .02, .05, None]
PRIMARY_BUDGET = .005


def read(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)


def initialize():
    RUN.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    path = RUN / 'protocol.json'
    if path.exists():
        protocol = read(path)
        assert protocol['checkpoint_sha256'] == file_sha256(ROOT / CHECKPOINT)
        return protocol
    from mmengine import Config
    import mmengine
    cfg = Config.fromfile(CONFIG)
    train = mmengine.load(cfg.train_dataset_config.imageset)
    val = mmengine.load(cfg.val_dataset_config.imageset)
    train_scenes = sorted(set(s for s, _ in train['metadata']))
    val_scenes = sorted(set(s for s, _ in val['metadata']))
    assert set(train_scenes).isdisjoint(val_scenes)
    rng = np.random.default_rng(928)
    rng.shuffle(train_scenes)
    cutoff = int(.75 * len(train_scenes))
    protocol = dict(created_at=datetime.now().astimezone().isoformat(),
        config=CONFIG, checkpoint=CHECKPOINT, checkpoint_sha256=file_sha256(ROOT / CHECKPOINT),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        annotation_sha256={s: file_sha256(cfg[s+'_dataset_config'].imageset) for s in ['train', 'val']},
        train_frames=len(train['metadata']), validation_frames=len(val['metadata']),
        fit_scenes=train_scenes[:cutoff], tune_scenes=train_scenes[cutoff:], validation_scenes=val_scenes,
        seeds=SEEDS, band=BAND, features=FEATURES, sample_per_region_margin_sign=SAMPLE_PER_STRATUM,
        fit_row_cap=400000, tune_row_cap=200000, budgets=BUDGETS, primary_budget=PRIMARY_BUDGET,
        candidate_rule='abs(fg_max - already_bias_corrected_empty) <= 2; native ties foreground',
        supervision='all valid train labels, camera masks used for training group losses only',
        inference_inputs='rendered scores, predicted class, xyz/range, actual foreground Gaussian support',
        probability_epochs=160, decision_steps=240, probability_selection='training tuning-scene weighted NLL',
        operating_point_selection='full tuning scenes only; maximize camera mIoU subject to complement FPR budget',
        no_joint_backbone_training=True, evaluation='all 1024 validation frames; native hard predictions',
        score_histogram=dict(bins=BINS, minimum=SCORE_MIN, maximum=SCORE_MAX),
        extraction_support='one extra native aggregation; prototype, not a fused implementation')
    write_json(path, protocol)
    return protocol


def build_loader(split, indices=None):
    from mmengine import Config
    from dataset import get_dataloader
    from dataset.utils import custom_collate_fn_temporal
    cfg = Config.fromfile(CONFIG)
    dcfg = copy.deepcopy(cfg.val_dataset_config)
    if split == 'train':
        dcfg.imageset = cfg.train_dataset_config.imageset
    _, loader = get_dataloader(cfg.train_dataset_config, dcfg, cfg.train_loader,
        dict(cfg.val_loader, batch_size=1, num_workers=4), val_only=True, seed=928)
    dataset = loader.dataset
    indices = list(range(len(dataset))) if indices is None else list(indices)
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=4,
        collate_fn=custom_collate_fn_temporal, pin_memory=True)
    return dataset, loader, indices


def build_network():
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model  # noqa: F401
    network = build_segmentor(Config.fromfile(CONFIG).model).cuda().eval()
    ckpt = torch.load(ROOT / CHECKPOINT, map_location='cpu', weights_only=False)
    network.load_state_dict(ckpt['state_dict'], strict=True)
    return network


def to_cuda(batch):
    return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def support_field(head, output, check=False):
    args = tuple(v[:, :-1] for v in head.prepare_gaussian_args(output['gaussian']))
    means, alpha, sem, scales, inverse = args
    # Keep native binning and covariance arithmetic. Only channel 0 is read.
    moments = torch.zeros_like(sem)
    moments[:, :, 0] = 1.
    xyz = output['sampled_xyz']
    rendered = head.aggregator(xyz.float().contiguous(), means.contiguous(),
        alpha.reshape(1, -1).contiguous(), moments.contiguous(), scales.contiguous(), inverse.contiguous())
    support = rendered[:, 0].clone()
    audit = {}
    if check:
        ids = torch.linspace(0, xyz.shape[1]-1, 32, device=xyz.device).long()
        reference = torch.zeros(len(ids), device=xyz.device)
        for k in range(0, means.shape[1], 2048):
            w = kernel_weights(xyz[0, ids], means[0, k:k+2048], alpha[0, k:k+2048],
                scales[0, k:k+2048], inverse[0, k:k+2048], head.aggregator.pc_min[0],
                head.aggregator.grid_size, head.aggregator.scale_multiplier)
            reference += w.sum(-1)
        audit['max_reference_support_error'] = float((support[ids]-reference).abs().max())
        assert torch.allclose(support[ids], reference, atol=2e-5, rtol=2e-5), audit
    return support, audit


def candidate_features(output, support, ids):
    logits = output['pred_occ'][-1][0]
    front = logits[:17, ids].T
    top = front.topk(2, -1).values
    xyz = output['sampled_xyz'][0, ids]
    total = front.sum(-1).clamp_min(0)
    normalized = front.clamp_min(0) / total[:, None].clamp_min(1e-20)
    ent = -(normalized*normalized.clamp_min(1e-20).log()).sum(-1)
    x = torch.cat([(top[:, 0]-logits[17, ids])[:, None], xyz, xyz[:, :2].norm(dim=-1, keepdim=True),
        (top[:, 0]-top[:, 1])[:, None], ent[:, None], torch.log1p(total)[:, None],
        torch.log1p(support[ids])[:, None]], -1)
    assert x.shape[1] == len(FEATURES) and torch.isfinite(x).all()
    return x


def confusion(gt, pred, region):
    valid = (gt >= 0) & (gt < 18)
    code = region[valid].long()*324+gt[valid].long()*18+pred[valid].long()
    return torch.bincount(code, minlength=648).reshape(2, 18, 18).cpu().numpy()


def extract(limit=0):
    protocol = initialize()
    seed_all(928)
    dataset, loader, indices = build_loader('train', range(limit) if limit else None)
    network = build_network()
    dest = RUN / ('train_smoke' if limit else 'train')
    dest.mkdir(exist_ok=True)
    phash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    frames, started = [], time.monotonic()
    with torch.inference_mode():
        for n, (index, batch) in enumerate(zip(indices, loader)):
            stem = dest / f'frame_{index:04d}'
            scene, sf = dataset.keyframes[index]
            if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
                meta = read(stem.with_suffix('.json'))
                assert meta['protocol_sha256'] == phash
            else:
                batch = to_cuda(batch)
                out = network(imgs=batch['img'], metas={k:v for k,v in batch.items() if k != 'img'})
                logits = out['pred_occ'][-1][0]
                maximum, fg = logits[:17].max(0)
                margin = maximum-logits[17]
                gt = out['sampled_label'][0].long()
                valid = (gt >= 0) & (gt < 18)
                region = (~batch['occ_cam_mask'][0].reshape(-1).bool()).long()
                native = logits.argmax(0)
                assert torch.equal(native, out['final_occ'][0].reshape(-1))
                assert torch.equal(native, torch.where(margin >= 0, fg, 17))
                eligible = valid & (margin.abs() <= BAND)
                support, audit = support_field(network.head, out, check=index < 2)
                rng = np.random.default_rng(928+index)
                fields = {k:[] for k in ['X', 'gt', 'fg', 'region', 'weight', 'voxel_id']}
                populations = []
                for r in [0, 1]:
                    for sign in [0, 1]:
                        ids = torch.where(eligible & (region == r) & ((margin >= 0).long() == sign))[0]
                        population = len(ids)
                        populations.append(population)
                        if not population:
                            continue
                        if population > SAMPLE_PER_STRATUM:
                            ids = ids[torch.as_tensor(rng.choice(population, SAMPLE_PER_STRATUM, replace=False), device=ids.device)]
                        fields['X'].append(candidate_features(out, support, ids).cpu().numpy())
                        fields['gt'].append(gt[ids].cpu().numpy().astype(np.int8))
                        fields['fg'].append(fg[ids].cpu().numpy().astype(np.int8))
                        fields['region'].append(np.full(len(ids), r, np.int8))
                        fields['weight'].append(np.full(len(ids), population/len(ids), np.float64))
                        fields['voxel_id'].append(ids.cpu().numpy().astype(np.int32))
                base = confusion(gt, native, region)
                fixed = confusion(gt[~eligible], native[~eligible], region[~eligible])
                meta = dict(index=index, scene=scene, scene_frame=int(sf), protocol_sha256=phash,
                    native_confusion=base.tolist(), fixed_confusion=fixed.tolist(), candidate_strata_populations=populations,
                    numerical_audit=audit, native_parity=True,
                    learned_bias=float(network.head.empty_bias.detach()))
                np.savez_compressed(stem.with_suffix('.npz'), **{k:np.concatenate(v) for k,v in fields.items()})
                write_json(stem.with_suffix('.json'), meta)
            frames.append(dict(index=index, scene=scene, scene_frame=int(sf), path=str(stem.relative_to(ROOT))))
            if n == 0 or (n+1) % 128 == 0:
                print(f'extract {n+1}/{len(indices)} elapsed={time.monotonic()-started:.1f}s', flush=True)
                write_json(RUN/'status.json', dict(status='running', stage='extract', frames=n+1, expected=len(indices)))
    summary = dict(status='complete', frames=frames, elapsed_seconds=time.monotonic()-started,
        source_sha256=file_sha256(__file__), protocol_sha256=phash)
    write_json(dest/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    initialize()
    with (ROOT/'work_dirs/center_support_contribution1/execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        extract(args.limit)


if __name__ == '__main__':
    main()
