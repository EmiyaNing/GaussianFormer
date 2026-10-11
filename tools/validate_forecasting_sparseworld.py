"""One-batch CUDA integration validation for ForecastingSparseWorld."""
import argparse
import os
import sys

import torch
from mmengine import Config
from mmseg.models import build_segmentor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model  # noqa: F401
import loss  # noqa: F401
from dataset import get_dataloader
from loss import OPENOCC_LOSS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    network = build_segmentor(cfg.model)
    network.init_weights()
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')
    result = network.load_state_dict(checkpoint_data['state_dict'], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(result)
    network = network.cuda().eval()
    objective = OPENOCC_LOSS.build(cfg.loss).cuda().eval()
    _, loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, val_only=True)
    batch = next(iter(loader))
    images = batch.pop('img').cuda()
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=cfg.get('amp', False)):
        output = network(imgs=images, metas=batch)
        total, metrics = objective(dict(
            forecasting_output=output['forecasting_output'], metas=batch,
            global_iter=checkpoint_data.get('global_iter', 0)))
    horizons = [item['horizon'] + 1
                for item in output['forecasting_output']['future_predictions']]
    print(dict(
        checkpoint_epoch=checkpoint_data.get('epoch'),
        global_iter=checkpoint_data.get('global_iter'),
        future_horizons=horizons,
        final_occ_shape=tuple(output['final_occ_grid'].shape),
        total_loss=float(total),
        current_total=float(metrics['current_total']),
        future_total=float(metrics['future_total']),
        latent_alignment=float(metrics['latent_alignment']),
        pyramid_alignment=float(metrics['pyramid_alignment']),
        peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024 ** 3))


if __name__ == '__main__':
    main()
