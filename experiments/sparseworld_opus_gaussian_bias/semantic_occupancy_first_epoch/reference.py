"""Re-evaluate C without modifying C files or advancing its training state."""
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from .. import train as runner
from ..ab_first_epoch.report import read
from ..fullspace_diagnostics import partition_histograms
from .diagnostics import save_scene_counts, scene_name
from .protocol import ANALYSIS, C_ANALYSIS, C_WORK, ROOT


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    audit = read(ANALYSIS / 'configuration_audit.json')['arms']['C_reference_eval']
    cfg = Config.fromfile(audit['config'])
    if runner.digest(audit['config']) != audit['config_sha256']:
        raise ValueError('C reference configuration changed')
    boundary = read(C_WORK / 'first_epoch_completed.json')
    checkpoint_path = C_WORK / 'epoch_01.pth'
    if runner.digest(checkpoint_path) != boundary['checkpoint_sha256']:
        raise ValueError('preserved C checkpoint changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    set_random_seed(cfg.seed)
    model = build_segmentor(cfg.model)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda()
    model.set_epoch(0)
    model.pts_bbox_head.num_stamps_all.copy_(checkpoint['state_dict']['pts_bbox_head.num_stamps_all'].cuda())
    model.pts_bbox_head.ind_stamps_all = checkpoint['query_stamps'].cuda()
    model.pts_bbox_head.reset_mask()
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    scenes, sample_counts, cursor = {}, {}, 0

    @torch.no_grad()
    def observe(module, args, kwargs, result):
        nonlocal cursor
        name = scene_name(dataset, cursor)
        cursor += 1
        metas = kwargs['metas']
        scenes.setdefault(name, np.zeros((7, 18, 18), dtype=np.int64))
        sample_counts[name] = sample_counts.get(name, 0) + 1
        for h, grid in enumerate([result['final_occ_grid']] + result['future_occ_grids']):
            target = metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h-1]
            cam = metas['occ_cam_mask'] if h == 0 else metas['future_occ_cam_masks'][:, h-1]
            scenes[name][h] += partition_histograms(grid, target, cam)['full']

    work = ROOT / 'C_reference_eval'
    work.mkdir(parents=True, exist_ok=True)
    handle = model.register_forward_hook(observe, with_kwargs=True)
    try:
        metrics = runner.evaluate(model, dataset, cfg, work, 1)
    finally:
        handle.remove()
    with np.load(C_WORK / 'epoch_01_confusion.npz', allow_pickle=False) as data:
        expected = data['confusion']
    if not np.array_equal(sum(scenes.values()), expected):
        raise ValueError('C reference re-evaluation does not exactly reproduce preserved C predictions')
    save_scene_counts(work / 'epoch_01_scene_confusion.npz', scenes, sample_counts)
    target = ANALYSIS / 'C_reference_eval'
    runner.atomic_json(target / 'reference_verified.json', dict(status='passed', exact_formal_confusion_matches_C=True,
        samples=cursor, scenes=len(scenes), checkpoint_sha256=boundary['checkpoint_sha256'],
        source_status_sha256=runner.digest(C_ANALYSIS / 'status.json'), checkpoint_not_modified=True,
        training_not_advanced=True, metrics=metrics))
    print(f'C_REFERENCE_VERIFIED samples={cursor} scenes={len(scenes)}', flush=True)


if __name__ == '__main__':
    main()
