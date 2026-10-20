"""One real Occ3D sample through OPUS-V1 Gaussian rendering and backward."""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from mmengine import Config
from mmseg.models import build_segmentor

import model  # noqa: F401 - registers the custom modules
from dataset import get_dataloader
from loss import OPENOCC_LOSS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default=
                        'config/mini_occ3d_4096_1024/opusv1_gaussian_sparse_render.py')
    parser.add_argument('--queries', type=int, default=8)
    parser.add_argument('--full-loss', action='store_true')
    args = parser.parse_args()
    config = Config.fromfile(args.py_config)
    config.model.lifter.num_queries = args.queries
    config.val_loader.batch_size = 1
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
    output = network(imgs=imgs, metas=batch)
    logits = output['pred_occ'][-1]
    labels = output['sampled_label'].long()
    assert logits.shape == (1, 18, 640000), logits.shape
    assert output['opus_pred_points'][-1].shape == (1, args.queries * 128, 3)
    if args.full_loss:
        loss_module = OPENOCC_LOSS.build(config.loss).cuda()
        inputs = {'metas': batch}
        for name, key in config.loss_input_convertion.items():
            inputs[name] = output[key]
        loss, detail = loss_module(inputs)
        print('loss_components', detail)
    else:
        loss = F.cross_entropy(logits, labels)
    loss.backward()
    branch = network.encoder.attribute_branches['5'].net[-1]
    point = network.encoder.layers[-1].reg_branch[-1]
    semantic = network.encoder.layers[-1].cls_branch[-1]
    for name, parameter in [('attributes', branch.weight),
                            ('point', point.weight), ('semantic', semantic.weight)]:
        gradient = parameter.grad
        assert gradient is not None and torch.isfinite(gradient).all(), name
        print(name, float(gradient.abs().sum()))
    if hasattr(network.head, 'empty_bias'):
        gradient = network.head.empty_bias.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        print('empty_bias_gradient', float(gradient.abs().sum()))
    assert branch.weight.grad.abs().sum() > 0
    grouped_bias_gradient = branch.bias.grad.view(128, 8).abs().sum(dim=0)
    print('attribute_bias_gradient_by_channel', grouped_bias_gradient.tolist())
    print('loss', float(loss.detach()), 'peak_mib',
          round(torch.cuda.max_memory_allocated() / 1048576))


if __name__ == '__main__':
    main()
