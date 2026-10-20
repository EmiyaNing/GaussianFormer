"""One real Occ3D mini sample through GaussianFormer and occupancy backward."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from mmengine import Config
from mmseg.models import build_segmentor

import model  # noqa: F401
from dataset import get_dataloader
from loss import OPENOCC_LOSS


def main():
    config = Config.fromfile('config/mini_occ3d_4096_1024/gaussianformer.py')
    config.val_loader.num_workers = 0
    _, loader = get_dataloader(
        config.train_dataset_config, config.val_dataset_config,
        config.train_loader, config.val_loader, val_only=True)
    batch = next(iter(loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    images = batch.pop('img')
    network = build_segmentor(config.model).cuda().train()
    prediction = network(imgs=images, metas=batch)
    criterion = OPENOCC_LOSS.build(config.loss).cuda()
    inputs = {'metas': batch}
    for name, key in config.loss_input_convertion.items():
        inputs[name] = prediction[key]
    loss, parts = criterion(inputs)
    loss.backward()
    assert torch.isfinite(loss)
    print('final_occ', tuple(prediction['final_occ'].shape),
          'loss', float(loss.detach()), 'components', parts,
          'peak_mib', round(torch.cuda.max_memory_allocated() / 1048576))


if __name__ == '__main__':
    main()
