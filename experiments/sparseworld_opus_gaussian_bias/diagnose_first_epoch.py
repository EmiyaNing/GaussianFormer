"""Read-only first-epoch diagnostics on fixed held-out samples.

Alternative rendering results are diagnostic probes, not new model scores or
calibration choices. This module never writes into training work directories.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from loss import OPENOCC_LOSS
from model.encoder.opus_encoder import _opus_decode_points
from .metrics import CLASS_NAMES
from .train import atomic_json, move_batch


def summarize(hist):
    tp = np.diag(hist).astype(float)
    union = hist.sum(0) + hist.sum(1) - tp
    iou = np.divide(tp, union, out=np.zeros(18), where=union > 0) * 100
    occupied_tp = float(hist[:17, :17].sum())
    occupied_gt = float(hist[:17].sum())
    occupied_pred = float(hist[:, :17].sum())
    return dict(miou=float(iou[:17].mean()), class_iou=iou.tolist(),
                occupied_iou=100 * occupied_tp / max(occupied_gt + occupied_pred - occupied_tp, 1),
                occupied_precision=100 * occupied_tp / max(occupied_pred, 1),
                occupied_recall=100 * occupied_tp / max(occupied_gt, 1),
                predicted_occupied_percent=100 * occupied_pred / hist.sum())


def quantiles(value):
    value = value.detach().float().reshape(-1)
    if not value.numel():
        return None
    return torch.quantile(value, value.new_tensor([0., .1, .5, .9, 1.])).cpu().tolist()


def semantic_gradient_probe(model, cfg, state, current, gaussian, xyz, metas):
    """Compare actual warm-up set focal and voxel gradients on final logits.

    Decoder features, means and Gaussian attributes are held fixed. These are
    gradients at the output logits, not at shared model parameters.
    """
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    with torch.enable_grad():
        raw = state['all_cls_scores'][-1].detach().clone().requires_grad_(True)
        points = _opus_decode_points(state['all_refine_pts'][-1], model.pc_range).flatten(1, 2)
        targets = loss_fn.strict._expanded_targets(metas, 0, points)
        set_loss, _ = loss_fn.strict._set_loss(
            points[0], raw[0].flatten(0, 1), *targets, temporal_reweight=True)
        set_grad = torch.autograd.grad(set_loss, raw)[0][:, current].flatten(-2)
        altered = gaussian._replace(semantics=F.softplus(raw[:, current]))
        rendered = model.gaussian_head.render_state(altered, xyz)
        voxel_loss = loss_fn.occupancy.loss_voxel(
            [rendered], xyz, metas['occ_label'].flatten(1), metas['occ_cam_mask'])
        voxel_grad = torch.autograd.grad(voxel_loss, raw)[0][:, current].flatten(-2)
    dot = (set_grad * voxel_grad).sum()
    child_dot = (set_grad.unflatten(-1, (-1, 17)) *
                 voxel_grad.unflatten(-1, (-1, 17))).sum(-1)
    active = child_dot != 0
    return dict(scope='Final current-query raw semantic logits only; fixed decoder output and Gaussian geometry.',
                temporal_final_stage_focal=float(set_loss.detach()),
                gaussian_voxel_loss=float(voxel_loss.detach()),
                set_gradient_norm=float(set_grad.norm()),
                voxel_gradient_norm=float(voxel_grad.norm()),
                gradient_cosine=float(dot / (set_grad.norm() * voxel_grad.norm()).clamp_min(1e-20)),
                opposed_children_percent=100 * float((child_dot[active] < 0).float().mean()),
                active_children=int(active.sum()),
                camera_visible_percent=100 * float(metas['occ_cam_mask'].float().mean()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--gradient-probe', action='store_true')
    parser.add_argument('--gaussian-only', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = dict(epoch=1, class_names=CLASS_NAMES,
                  note='Fixed small-sample diagnostics; alternative renderings are not formal evaluation scores.',
                  quantile_probabilities=[0., .1, .5, .9, 1.], models={})
    names = ('sparseworld_opus_gaussian_bias',) if args.gaussian_only else (
        'sparseworld_strict', 'sparseworld_opus_gaussian_bias')
    for name in names:
        cfg = Config.fromfile(f'config/sparseworld_opus_gaussian_bias/{name}_4096_1024_15e.py')
        model = build_segmentor(cfg.model)
        model.init_weights()
        model.set_epoch(0)
        checkpoint = Path(f'out/sparseworld_opus_gaussian_bias_4096_1024_15e/{name}/epoch_01.pth')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.cuda().eval()
        model.pts_bbox_head.ind_stamps_all = saved['query_stamps'].cuda()
        model.pts_bbox_head.reset_mask()
        del saved
        dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
        indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
        histograms, records = {}, []
        for index in indices:
            images, metas = move_batch(custom_collate_fn_temporal([dataset[index]]), 'cuda')
            ego = model.plan_head(metas['temporal_ego_states'].to(images).reshape(1, 1, -1))
            scale = (torch.tanh(model.points_scale_branch(ego)) + 1) * .35 + .8
            state = model.pts_bbox_head(model.extract_feat(images, metas), metas, points_scale=scale)
            current = model.pts_bbox_head.ind_stamps_all == 0
            points = state['all_refine_pts'][-1][:, current]
            classes = state['all_cls_scores'][-1][:, current]
            inside = torch.isfinite(points).all(-1) & ((points >= 0) & (points < 1)).all(-1)
            valid_scores = classes.sigmoid().amax(-1)[inside]
            modes = dict(point_rasterizer=model.pts_bbox_head.get_occ(
                dict(cls_scores=classes, refine_pts=points)))
            record = dict(index=index, children=points.shape[2],
                          in_range_percent=100 * float(inside.float().mean()),
                          valid_point_score_quantiles=quantiles(valid_scores))
            if name == 'sparseworld_opus_gaussian_bias':
                gaussian = state['representation'][-1]['gaussian']
                gaussian = type(gaussian)(*(v[:, current] if v is not None else None for v in gaussian))
                xyz = metas['occ_xyz'].float().flatten(1, 3)
                logits = model.gaussian_head.render_state(gaussian, xyz)
                modes['gaussian_default'] = logits.argmax(1).view(1, 200, 200, 16)
                for delta in (-1., 1., 2.):
                    altered = logits.clone()
                    altered[:, 17] -= delta
                    modes[f'empty_bias_delta_{delta:+g}'] = altered.argmax(1).view(1, 200, 200, 16)
                altered = gaussian._replace(opacities=inside[..., None].to(gaussian.opacities))
                modes['opacity_one'] = model.gaussian_head.render_state(altered, xyz).argmax(1).view(1, 200, 200, 16)
                altered = gaussian._replace(scales=(gaussian.scales * 1.5).clamp(max=.4))
                modes['scales_times_1_5'] = model.gaussian_head.render_state(altered, xyz).argmax(1).view(1, 200, 200, 16)
                gt = metas['occ_label'].flatten(1)
                foreground = logits[:, :17].amax(1)
                record.update(empty_bias=float(model.gaussian_head.empty_bias),
                              empty_scalar=float(model.gaussian_head.empty_scalar),
                              scales_quantiles=quantiles(gaussian.scales[inside]),
                              opacity_quantiles=quantiles(gaussian.opacities[inside]),
                              softplus_semantic_max_quantiles=quantiles(gaussian.semantics.amax(-1)[inside]),
                              foreground_logits_on_gt_occupied_quantiles=quantiles(foreground[gt != 17]),
                              empty_logits_on_gt_occupied_quantiles=quantiles(logits[:, 17][gt != 17]))
                if args.gradient_probe and index == indices[0]:
                    record['semantic_gradient_probe'] = semantic_gradient_probe(
                        model, cfg, state, current, gaussian, xyz, metas)
            gt = metas['occ_label'].long().reshape(-1)
            for mode, prediction in modes.items():
                hist = torch.bincount(gt * 18 + prediction.long().reshape(-1), minlength=324)
                histograms.setdefault(mode, np.zeros((18, 18), dtype=np.int64))
                histograms[mode] += hist.reshape(18, 18).cpu().numpy()
            records.append(record)
            print(name, index, 'done', flush=True)
            model.memory.clear()
            model.memory_queue.clear()
        output['models'][name] = dict(checkpoint=str(checkpoint), indices=indices,
                                     records=records, metrics={k: summarize(v) for k, v in histograms.items()})
        del model, dataset, state, modes, images, metas
        torch.cuda.empty_cache()
    atomic_json(args.output, output)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
