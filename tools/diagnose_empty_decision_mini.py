"""Frozen mini-set census and label-free local evidence for empty decisions."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mmengine import Config
from mmseg.models import build_segmentor
import model  # noqa: F401
from dataset import get_dataloader
from dataset.utils import custom_collate_fn_temporal
from tools.center_support_common import kernel_weights, write_json
from tools.diagnose_opus_v1_gaussian import calibrated_predictions, file_sha256

RUN = ROOT / 'work_dirs/empty_decision_mini'
CONFIG = 'config/center_support_contribution1/a0_original.py'
CHECKPOINT = 'work_dirs/center_support_contribution1/train/a0_original_seed928/epoch_5.pth'
MANIFEST = 'work_dirs/opus_gaussian_empty_scope/training_calibration_manifest.json'
REGIONS = ['camera', 'camera_complement', 'all_valid']
THRESHOLDS = [1e-6, 1e-3, 1e-2, 1e-1]
FEATURES = ['margin', 'x', 'y', 'z', 'range_xy', 'log_support', 'log_neff',
            'weighted_opacity', 'weighted_scale', 'fg_top_gap', 'fg_entropy',
            'log_fg_total', 'vote_agreement', 'vote_entropy',
            'individual_semantic_entropy', 'frustum_views', 'image_norm_mean',
            'image_norm_max', 'image_crossview_cosine', 'multi_view_available']
EVENTS = ['gt_occupied', 'gt_empty', 'native_fn', 'native_fn_fg_correct',
          'native_fn_fg_wrong', 'native_semantic_correct', 'native_semantic_wrong',
          'native_fp', 'flips', 'correct_rescue', 'wrong_class_exposed', 'new_false_occupied']


def entropy(p):
    return -(p * p.clamp_min(1e-20).log()).sum(-1)


def transition_masks(gt, fg, native_pred, shifted_pred):
    """Disjoint transitions for a positive shift, foreground wins exact ties."""
    flip = (native_pred == 17) & (shifted_pred != 17)
    return dict(flips=flip, correct_rescue=flip & (gt < 17) & (fg == gt),
                wrong_class_exposed=flip & (gt < 17) & (fg != gt),
                new_false_occupied=flip & (gt == 17))


def select_uniform(mask, limit, generator):
    ids = torch.where(mask)[0]
    population = ids.numel()
    if population > limit:
        # CPU RNG separates sampling from GPU kernel implementation details.
        choice = torch.from_numpy(generator.choice(population, limit, replace=False)).to(ids.device)
        ids = ids[choice]
    return ids, population / max(len(ids), 1)


def render(head, xyz, args, semantic, opacity=None, inverse=None):
    means, alpha, _, scales, inv = args
    return head.aggregator(xyz.float().contiguous(), means.contiguous(),
        (alpha if opacity is None else opacity).reshape(1, -1).contiguous(),
        semantic.contiguous(), scales.contiguous(), (inv if inverse is None else inverse).contiguous())


def evidence(head, output, check=False):
    """Use production binning, covariances and truncation, excluding empty G."""
    gaussian = output['gaussian']
    complete_args = head.prepare_gaussian_args(gaussian)
    args = tuple(x[:, :-1] for x in complete_args)
    means, alpha, semantic, scales, inverse = args
    xyz = output['sampled_xyz']
    sem = semantic[0, :, :17]
    if sem.min() < 0:
        raise ValueError('This experiment requires nonnegative softplus semantics')
    votes = F.one_hot(sem.argmax(-1), 18).to(sem)
    votes[:, 17] = 1.
    weighted_votes = render(head, xyz, args, votes[None])
    support = weighted_votes[:, 17].clone()
    normalized_votes = weighted_votes[:, :17] / support[:, None].clamp_min(1e-20)
    vote_agreement = normalized_votes.max(-1).values
    vote_entropy = entropy(normalized_votes)
    del weighted_votes
    moments = torch.zeros_like(semantic)
    moments[0, :, 0] = alpha.reshape(-1)
    moments[0, :, 1] = scales[0].mean(-1)
    semprob = sem / sem.sum(-1, keepdim=True).clamp_min(1e-20)
    moments[0, :, 2] = entropy(semprob)
    weighted_moments = render(head, xyz, args, moments)
    averages = weighted_moments[:, :3] / support[:, None].clamp_min(1e-20)
    del weighted_moments
    ones = torch.ones_like(semantic)
    square_mass = render(head, xyz, args, ones, opacity=alpha.square(), inverse=inverse * 2)[:, 0].clone()
    neff = support.square() / square_mass.clamp_min(1e-30)
    audit = {}
    if check:
        rendered = render(head, xyz, args, semantic)
        reference = output['pred_occ'][-1][0, :17].T
        audit['max_foreground_render_error'] = float((rendered[:, :17] - reference).abs().max())
        if not torch.allclose(rendered[:, :17], reference, atol=1e-5, rtol=1e-5):
            raise AssertionError(audit)
        del rendered
        ids = torch.linspace(0, xyz.shape[1]-1, 24, device=xyz.device).long()
        ss = torch.zeros(len(ids), device=xyz.device)
        qq = torch.zeros_like(ss)
        for k in range(0, means.shape[1], 2048):
            w = kernel_weights(xyz[0, ids], means[0, k:k+2048], alpha[0, k:k+2048],
                scales[0, k:k+2048], inverse[0, k:k+2048], head.aggregator.pc_min[0],
                head.aggregator.grid_size, head.aggregator.scale_multiplier)
            ss += w.sum(-1)
            qq += w.square().sum(-1)
        audit['max_reference_support_error'] = float((ss-support[ids]).abs().max())
        audit['max_reference_square_mass_error'] = float((qq-square_mass[ids]).abs().max())
        assert torch.allclose(ss, support[ids], atol=2e-5, rtol=2e-5), audit
        assert torch.allclose(qq, square_mass[ids], atol=2e-5, rtol=2e-5), audit
    return dict(support=support, neff=neff, averages=averages,
                vote_agreement=vote_agreement, vote_entropy=vote_entropy), audit


def image_evidence(xyz, output, batch):
    """Projected FPN proxies, WITHOUT GT depth, masks, or semantic labels."""
    projection = batch['projection_mat'][0].float()
    wh = batch['image_wh'][0].float()
    homogeneous = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1)
    projected = torch.einsum('cij,nj->cni', projection, homogeneous)
    depth = projected[..., 2]
    uv = projected[..., :2] / depth[..., None].clamp_min(1e-5)
    valid = (depth > 1e-5) & (uv[..., 0] >= 0) & (uv[..., 1] >= 0)
    valid &= (uv[..., 0] < wh[:, None, 0]) & (uv[..., 1] < wh[:, None, 1])
    grid = uv / wh[:, None] * 2 - 1
    fmap = output['ms_img_feats'][0][0].float()
    sampled = F.grid_sample(fmap, grid[:, :, None], align_corners=False).squeeze(-1).permute(0, 2, 1)
    count = valid.sum(0).float()
    norm = sampled.norm(dim=-1)
    mean_norm = (norm*valid).sum(0) / count.clamp_min(1)
    max_norm = (norm*valid).max(0).values
    unit = F.normalize(sampled, dim=-1) * valid[..., None]
    # Sum of ordered cross-view dot products / number of ordered valid pairs.
    cross = (unit.sum(0).square().sum(-1) - count) / (count * (count-1)).clamp_min(1)
    cross = torch.where(count >= 2, cross, torch.zeros_like(cross))
    return torch.stack([count, mean_norm, max_norm, cross, (count >= 2).float()], -1)


def features(ids, output, batch, ev):
    logits = output['pred_occ'][-1][0]
    fg = logits[:17, ids].T
    maxima = fg.topk(2, -1).values
    xyz = output['sampled_xyz'][0, ids]
    p = fg.clamp_min(0) / fg.sum(-1, keepdim=True).clamp_min(1e-20)
    semantic = torch.stack([maxima[:, 0]-maxima[:, 1], entropy(p),
        torch.log1p(fg.sum(-1).clamp_min(0)), ev['vote_agreement'][ids],
        ev['vote_entropy'][ids], ev['averages'][ids, 2]], -1)
    support = torch.stack([torch.log1p(ev['support'][ids]), torch.log1p(ev['neff'][ids]),
                          ev['averages'][ids, 0], ev['averages'][ids, 1]], -1)
    position = torch.cat([xyz, xyz[:, :2].norm(dim=-1, keepdim=True)], -1)
    margin = maxima[:, 0] - logits[17, ids]
    result = torch.cat([margin[:, None], position, support, semantic,
                        image_evidence(xyz, output, batch)], -1)
    assert result.shape[1] == len(FEATURES) and torch.isfinite(result).all()
    return result


def process(split, limit=None):
    random.seed(928)
    np.random.seed(928)
    torch.manual_seed(928)
    torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(CONFIG)
    dcfg = copy.deepcopy(cfg.val_dataset_config)
    if split == 'train':
        dcfg.imageset = cfg.train_dataset_config.imageset
    _, loader = get_dataloader(cfg.train_dataset_config, dcfg, cfg.train_loader,
        dict(cfg.val_loader, batch_size=1, num_workers=4), val_only=True, seed=928)
    dataset = loader.dataset
    indices = json.loads((ROOT / MANIFEST).read_text())['indices'] if split == 'train' else list(range(len(dataset)))
    if limit:
        indices = indices[:limit]
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=4,
        collate_fn=custom_collate_fn_temporal, pin_memory=True)
    network = build_segmentor(cfg.model).cuda().eval()
    checkpoint = torch.load(ROOT / CHECKPOINT, map_location='cpu', weights_only=False)
    network.load_state_dict(checkpoint['state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    bias = float(network.head.empty_bias.detach())
    shifts = [0., 1., 4., -bias]
    scene_ids = sorted(set(dataset.keyframes[i][0] for i in indices))
    output_dir = RUN / (split + ('_smoke' if limit else ''))
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(config=CONFIG, checkpoint=CHECKPOINT, checkpoint_sha256=file_sha256(ROOT / CHECKPOINT),
        resolved_config_sha256=hashlib.sha256(cfg.pretty_text.encode()).hexdigest(),
        imageset=str(dcfg.imageset), imageset_sha256=file_sha256(dcfg.imageset),
        source_sha256=file_sha256(__file__), checkpoint_epoch=epoch, learned_bias=bias,
        shifts=shifts, selected_extra_shift=1., features=FEATURES, sample_limit_per_region_kind=2048,
        regions=REGIONS, support_thresholds=THRESHOLDS, split=split, seed=928)
    write_json(output_dir / 'protocol.json', protocol)
    frames, audits = [], []
    totals = {r: {k: 0 for k in EVENTS} for r in REGIONS}
    support_totals = {r: {str(t): [0, 0, 0] for t in THRESHOLDS} for r in REGIONS}
    conf = {r: {str(b): np.zeros((18, 18), dtype=np.int64) for b in shifts} for r in REGIONS}
    start = time.monotonic()
    with torch.inference_mode():
        for frame_id, (index, batch) in enumerate(zip(indices, loader)):
            scene, scene_frame = dataset.keyframes[index]
            stem = output_dir / f'frame_{frame_id:04d}'
            if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
                frame = json.loads(stem.with_suffix('.json').read_text())
                assert frame['protocol_sha256'] == hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            else:
                batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                output = network(imgs=batch.pop('img'), metas=batch)
                logits = output['pred_occ'][-1][0]
                maxfg, fg = logits[:17].max(0)
                margin = maxfg-logits[17]
                gt = output['sampled_label'][0].long()
                valid = (gt >= 0) & (gt < 18)
                camera = batch['occ_cam_mask'][0].reshape(-1).bool()
                masks = [valid & camera, valid & ~camera, valid]
                predictions = {b: calibrated_predictions(maxfg, fg, logits[17], b) for b in shifts}
                pred = predictions[0.]
                assert torch.equal(pred, output['final_occ'][0].reshape(-1))
                ev, audit = evidence(network.head, output, check=frame_id < 2)
                transitions = transition_masks(gt, fg, pred, predictions[1.])
                common = dict(gt_occupied=gt < 17, gt_empty=gt == 17,
                    native_fn=(gt < 17) & (pred == 17),
                    native_fn_fg_correct=(gt < 17) & (pred == 17) & (fg == gt),
                    native_fn_fg_wrong=(gt < 17) & (pred == 17) & (fg != gt),
                    native_semantic_correct=(gt < 17) & (pred == gt),
                    native_semantic_wrong=(gt < 17) & (pred < 17) & (pred != gt),
                    native_fp=(gt == 17) & (pred < 17), **transitions)
                frame = dict(index=index, scene=scene, scene_frame=int(scene_frame),
                    protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                    counts={}, support={}, confusion={}, numerical_audit=audit,
                    native_parity=True)
                samples = {k: [] for k in ['X', 'gt', 'fg', 'region', 'kind', 'weight', 'voxel_id']}
                rng = np.random.default_rng(928 + index + (0 if split == 'train' else 100000))
                for ri, (name, mask) in enumerate(zip(REGIONS, masks)):
                    frame['counts'][name] = {k: int((v & mask).sum()) for k, v in common.items()}
                    frame['support'][name] = {str(t): [
                        int((common['native_fn'] & mask & (ev['support'] > t)).sum()),
                        int((common['native_fn_fg_correct'] & mask & (ev['support'] > t)).sum()),
                        int((common['correct_rescue'] & mask & (ev['support'] > t)).sum())]
                        for t in THRESHOLDS}
                    frame['confusion'][name] = {}
                    for b in shifts:
                        pp = predictions[b]
                        cc = torch.bincount(gt[mask]*18 + pp[mask], minlength=324).reshape(18, 18)
                        frame['confusion'][name][str(b)] = cc.cpu().tolist()
                    if ri == 2:
                        continue
                    for kind, sample_mask in [(0, mask), (1, mask & transitions['flips'])]:
                        ids, weight = select_uniform(sample_mask, 2048, rng)
                        if not len(ids):
                            continue
                        samples['X'].append(features(ids, output, batch, ev).cpu().numpy())
                        samples['gt'].append(gt[ids].cpu().numpy().astype(np.int8))
                        samples['fg'].append(fg[ids].cpu().numpy().astype(np.int8))
                        samples['region'].append(np.full(len(ids), ri, np.int8))
                        samples['kind'].append(np.full(len(ids), kind, np.int8))
                        samples['weight'].append(np.full(len(ids), weight, np.float64))
                        samples['voxel_id'].append(ids.cpu().numpy().astype(np.int32))
                for key in EVENTS:
                    assert frame['counts']['all_valid'][key] == sum(frame['counts'][r][key] for r in REGIONS[:2])
                assert frame['counts']['all_valid']['flips'] == sum(frame['counts']['all_valid'][k] for k in
                    ['correct_rescue', 'wrong_class_exposed', 'new_false_occupied'])
                np.savez_compressed(stem.with_suffix('.npz'), **{k: np.concatenate(v) for k, v in samples.items()})
                write_json(stem.with_suffix('.json'), frame)
                del output, ev
            frames.append(dict(index=index, scene=scene, scene_frame=int(scene_frame), path=str(stem.relative_to(ROOT))))
            if frame['numerical_audit']:
                audits.append(frame['numerical_audit'])
            for r in REGIONS:
                for k, value in frame['counts'][r].items():
                    totals[r][k] += value
                for t, values in frame['support'][r].items():
                    support_totals[r][t] = [a+b for a, b in zip(support_totals[r][t], values)]
                for b in shifts:
                    conf[r][str(b)] += frame['confusion'][r][str(b)]
            if (frame_id+1) % 32 == 0 or frame_id == 0:
                print(f'{split} {frame_id+1}/{len(indices)} elapsed={time.monotonic()-start:.1f}s', flush=True)
                write_json(RUN / 'status.json', dict(status='running', split=split,
                    completed_frames=frame_id+1, expected_frames=len(indices), pid=__import__('os').getpid()))
    for b in shifts:
        assert np.array_equal(conf['all_valid'][str(b)], conf['camera'][str(b)]+conf['camera_complement'][str(b)])
    # Match the independently completed, hashed mini-set sweep exactly.
    reference_path = ROOT / f'work_dirs/opus_gaussian_empty_scope/current_{split}_initial.json'
    parity = False
    if not limit:
        reference = json.loads(reference_path.read_text())
        assert reference['checkpoint_sha256'] == protocol['checkpoint_sha256']
        for r, rr in [('camera', 'camera'), ('all_valid', 'all_valid'), ('camera_complement', 'camera_invisible')]:
            for b in [0., 1., 4.]:
                assert np.array_equal(conf[r][str(b)], reference['masks'][rr]['empty_logit_shift_metrics'][str(b)]['confusion'])
        parity = True
    result = dict(status='complete', protocol=protocol, frames=frames, scene_ids=scene_ids,
        counts=totals, support_counts=support_totals, confusion={r: {b: v.tolist() for b, v in cc.items()} for r, cc in conf.items()},
        samples=len(frames), elapsed_seconds=time.monotonic()-start, numerical_audits=audits,
        previous_sweep_exact_parity=parity, finished_at=datetime.now().astimezone().isoformat())
    write_json(output_dir / 'summary.json', result)
    assert file_sha256(ROOT / CHECKPOINT) == protocol['checkpoint_sha256']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both')
    parser.add_argument('--smoke-frames', type=int)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'work_dirs/center_support_contribution1/execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for split in (['train', 'val'] if args.split == 'both' else [args.split]):
                process(split, args.smoke_frames)
            write_json(RUN / 'status.json', dict(status='extraction_complete', split=args.split, smoke=bool(args.smoke_frames)))
        except BaseException as exc:
            write_json(RUN / 'status.json', dict(status='failed', error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
