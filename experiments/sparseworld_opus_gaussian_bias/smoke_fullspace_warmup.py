"""GPU check of warm-up backward and its temporal-expanded gradient probe."""
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from . import train as runner
from .queue_fullspace import CONFIG, ROOT
from .train_fullspace import evaluate_with_diagnostics


def main():
    cfg = Config.fromfile(CONFIG)
    directory = ROOT / 'smoke_warmup'
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_random_seed(cfg.seed)
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = runner.load_image_initialization(model, cfg.train_load_from)
    model.cuda().train()
    model.set_epoch(0)
    dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
    dataset.set_epoch(0)
    batch = next(iter(runner.loader(dataset, cfg.train_loader)))
    images, metas = runner.move_batch(batch, 'cuda')
    result = model(imgs=images, metas=metas)
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    loss, details = loss_fn(dict(result, metas=metas, global_iter=0))
    if not torch.isfinite(loss):
        raise ValueError('warm-up loss is not finite')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
    model.zero_grad(set_to_none=True)
    del images, metas, result, loss_fn
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    before = runner.capture_rng()
    metrics = evaluate_with_diagnostics(model, validation, cfg, directory, 1, limit=1)
    after = runner.capture_rng()
    if (before['python'] != after['python'] or
            not torch.equal(before['torch'], after['torch']) or
            any(not torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda']))):
        raise ValueError('diagnostic modified training random state')
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('diagnostic accumulated model gradients')
    runner.atomic_json(directory / 'smoke_success.json', dict(
        status='passed', stage='temporal_pretrain', initialization=initialization,
        loss=float(loss.detach()), loss_details={k: float(v) for k, v in details.items()},
        metrics=metrics, diagnostics_preserved_rng=True, diagnostics_accumulated_parameter_grad=False))
    print('WARMUP_SMOKE passed', flush=True)


if __name__ == '__main__':
    main()
