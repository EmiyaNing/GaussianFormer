"""One real Occ3D sample through the single-frame OPUS-V2 sparse head."""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import get_dataloader
from loss import OPENOCC_LOSS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', type=int, default=8)
    args = parser.parse_args()
    config = Config.fromfile('config/mini_occ3d_4096_1024/opusv2.py')
    config.model.lifter.num_queries = args.queries
    config.val_loader.num_workers = 0
    _, loader = get_dataloader(
        config.train_dataset_config, config.val_dataset_config,
        config.train_loader, config.val_loader, val_only=True)
    batch = next(iter(loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    imgs = batch.pop('img')
    network = build_segmentor(config.model).cuda().train()
    prediction = network(imgs=imgs, metas=batch)
    loss_module = OPENOCC_LOSS.build(config.loss).cuda()
    arguments = {'metas': batch}
    for argument, output_key in config.loss_input_convertion.items():
        arguments[argument] = prediction[output_key]
    loss, detail = loss_module(arguments)
    loss.backward()
    gradient = network.head.densifiers[-1].classifier[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    print('final_occ', tuple(prediction['final_occ'].shape),
          'sparse_logits', tuple(prediction['v2_sparse_logits'][-1].shape))
    print('loss', float(loss.detach()), detail,
          'head_grad', float(gradient.abs().sum()),
          'peak_mib', round(torch.cuda.max_memory_allocated() / 1048576))


if __name__ == '__main__':
    main()
