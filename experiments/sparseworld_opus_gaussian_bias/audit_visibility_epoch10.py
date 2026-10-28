"""Full held-out visibility audit at fixed checkpoints; no parameter updates."""
import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from .diagnose_first_epoch import summarize
from .train import atomic_json, digest, loader, move_batch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    MMLogger.get_current_instance().setLevel(logging.ERROR)
    output = dict(epoch=args.epoch, num_samples=1024, evaluation='fixed-checkpoint visibility audit', models={})
    if args.resume and Path(args.output).exists():
        output = json.loads(Path(args.output).read_text())
        if output['epoch'] != args.epoch or output['num_samples'] != 1024:
            raise ValueError('resume audit protocol differs')
    for name in ('sparseworld_strict', 'sparseworld_opus_gaussian_bias'):
        checkpoint = Path(f'out/sparseworld_opus_gaussian_bias_4096_1024_15e/{name}/epoch_{args.epoch:02d}.pth')
        if name in output['models']:
            if output['models'][name]['checkpoint_sha256'] != digest(checkpoint):
                raise ValueError('resume audit checkpoint changed')
            print('Preserved completed audit for', name, flush=True)
            continue
        started = time.monotonic()
        cfg = Config.fromfile(f'config/sparseworld_opus_gaussian_bias/{name}_4096_1024_15e.py')
        set_random_seed(cfg.seed)
        model = build_segmentor(cfg.model)
        model.init_weights()
        model.set_epoch(args.epoch - 1)
        directory = Path(f'out/sparseworld_opus_gaussian_bias_4096_1024_15e/{name}')
        checkpoint = directory / f'epoch_{args.epoch:02d}.pth'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'], strict=True)
        model.cuda().eval()
        model.pts_bbox_head.ind_stamps_all = saved['query_stamps'].cuda()
        model.pts_bbox_head.reset_mask()
        del saved
        dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
        if len(dataset) != 1024:
            raise ValueError('audit requires the complete 1024-sample held-out subset')
        histograms, samples = {}, 0
        model.memory.clear()
        model.memory_queue.clear()
        for data in loader(dataset, cfg.val_loader):
            images, metas = move_batch(data, 'cuda')
            result = model(imgs=images, metas=metas)
            grids = [result['final_occ_grid']] + result['future_occ_grids']
            if len(grids) != 7:
                raise ValueError('audit requires all seven horizons')
            for horizon, prediction in enumerate(grids):
                target = metas['occ_label'] if horizon == 0 else metas['future_occ_labels'][:, horizon - 1]
                cam = metas['occ_cam_mask'] if horizon == 0 else metas['future_occ_cam_masks'][:, horizon - 1]
                # CPU integer bincount avoids repeated GPU atomics on the
                # dominant empty-class bin. Hidden=full-visible is exact.
                gt = target.cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                pred = prediction.cpu().numpy().reshape(-1).astype(np.int64, copy=False)
                visible = cam.cpu().numpy().reshape(-1).astype(bool, copy=False)
                valid = (gt >= 0) & (gt < 18)
                ids = gt[valid] * 18 + pred[valid]
                full_hist = np.bincount(ids, minlength=324).reshape(18, 18)
                cam_hist = np.bincount(ids[visible[valid]], minlength=324).reshape(18, 18)
                for region, hist in [('full', full_hist), ('camera', cam_hist),
                                     ('noncamera', full_hist - cam_hist)]:
                    key = f'h{horizon}/{region}'
                    histograms.setdefault(key, np.zeros((18, 18), dtype=np.int64))
                    histograms[key] += hist
            samples += images.shape[0]
            if samples % 64 == 0:
                print(name, samples, 'elapsed', round(time.monotonic() - started, 1), flush=True)
        if samples != 1024:
            raise ValueError('incomplete audit')
        formal = np.load(directory / f'epoch_{args.epoch:02d}_confusion.npz')['confusion']
        recheck = np.stack([histograms[f'h{h}/full'] for h in range(7)])
        for h in range(7):
            if not np.array_equal(histograms[f'h{h}/full'],
                                  histograms[f'h{h}/camera'] + histograms[f'h{h}/noncamera']):
                raise ValueError('visibility regions do not partition the full grid')
        metrics = {}
        for key, hist in histograms.items():
            occupied_tp = int(hist[:17, :17].sum())
            false_positive = int(hist[17, :17].sum())
            false_negative = int(hist[:17, 17].sum())
            metrics[key] = dict(summarize(hist), voxel_count=int(hist.sum()),
                               occupied_tp=occupied_tp, occupied_fp=false_positive,
                               occupied_fn=false_negative,
                               gt_occupied=int(hist[:17].sum()))
        output['models'][name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                                     exact_formal_confusion_match=bool(np.array_equal(formal, recheck)),
                                     formal_confusion_absolute_difference=int(np.abs(formal - recheck).sum()),
                                     seconds=time.monotonic() - started, metrics=metrics)
        atomic_json(args.output, output)
        del model, dataset, result, images, metas, grids
        torch.cuda.empty_cache()
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
