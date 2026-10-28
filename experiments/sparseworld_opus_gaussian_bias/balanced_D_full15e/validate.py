"""Validate resumed optimizer/model and per-epoch diagnostics at early/late stages."""
import gc

import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from ..train import atomic_json, loader, make_optimizer, move_batch, set_lr
from .diagnostics import evaluate
from .protocol import ANALYSIS, CONFIG, ROOT, WORK


def main():
    cfg = Config.fromfile(str(CONFIG))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_random_seed(cfg.seed)
    saved = torch.load(WORK/'epoch_01.pth', map_location='cpu', weights_only=False)
    model = build_segmentor(cfg.model)
    model.init_weights()
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cuda()
    optimizer = make_optimizer(model, cfg)
    optimizer.load_state_dict(saved['optimizer'])
    set_lr(optimizer, saved['updates'], 15360, cfg.warmup_iters, cfg.min_lr_ratio)
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    train = OPENOCC_DATASET.build(cfg.train_dataset_config)
    val = OPENOCC_DATASET.build(cfg.val_dataset_config)
    checks = []
    del saved
    for epoch_index in (1, 14):
        print(f'D_CONTINUATION_SMOKE epoch={epoch_index+1}', flush=True)
        model.train()
        model.set_epoch(epoch_index)
        if epoch_index == 1 and not torch.equal(model.pts_bbox_head.num_stamps_all, torch.ones_like(model.pts_bbox_head.num_stamps_all)):
            raise ValueError('new pretraining epoch did not reset query statistics')
        train.set_epoch(epoch_index)
        images, metas = move_batch(next(iter(loader(train, cfg.train_loader, seed=cfg.seed+epoch_index))), 'cuda')
        torch.cuda.reset_peak_memory_stats()
        result = model(imgs=images, metas=metas)
        loss, _ = loss_fn(dict(result, metas=metas, global_iter=epoch_index*2048))
        if not torch.isfinite(loss):
            raise ValueError('resumed smoke loss is not finite')
        horizons = sorted(result['gaussian_logits_by_horizon'])
        if horizons != ([0] if epoch_index==1 else [0,2,4,6]):
            raise ValueError('Gaussian future supervision curriculum changed')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        value, memory = float(loss.detach()), torch.cuda.max_memory_allocated()/2**20
        del result, loss, images, metas
        gc.collect()
        directory = ROOT/'smoke'/f'epoch_{epoch_index+1:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        metric = evaluate(model, val, cfg, directory, epoch_index+1, limit=1)
        if metric['num_samples']!=1:
            raise ValueError('diagnostic smoke sample count differs')
        checks.append(dict(epoch=epoch_index+1, loss=value, supervised_horizons=horizons,
                           peak_cuda_mib=memory, validation_samples=1, diagnostics_saved=True))
    atomic_json(ANALYSIS/'gpu_validation.json', dict(status='passed', checks=checks,
        resumed_model_and_optimizer=True, validation_only_in_memory=True, formal_checkpoint_unchanged=True))
    print('D continuation GPU checks passed.', flush=True)


if __name__ == '__main__':
    main()
