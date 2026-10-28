"""Check both warm-up objectives and A's full future curriculum on GPU."""
import json

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from .. import train as runner
from .diagnostics import evaluate
from .protocol import ARMS, ROOT
from .train import model_fingerprint


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fingerprints = {}
    for arm, epoch_index in [('A_softmax_focal', 0), ('B_ce_free_0_1', 0), ('A_softmax_focal', 14)]:
        cfg = Config.fromfile(str(ARMS[arm]))
        work = ROOT / 'smoke' / f'{arm}_epoch_{epoch_index+1:02d}'
        work.mkdir(parents=True, exist_ok=True)
        set_random_seed(cfg.seed)
        model = build_segmentor(cfg.model)
        model.init_weights()
        initialization = runner.load_image_initialization(model, cfg.train_load_from)
        initial_hash = model_fingerprint(model)
        fingerprints[f'{arm}/{epoch_index}'] = initial_hash
        model.cuda().train()
        model.set_epoch(epoch_index)
        dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
        dataset.set_epoch(epoch_index)
        images, metas = runner.move_batch(next(iter(runner.loader(dataset, cfg.train_loader))), 'cuda')
        loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
        result = model(imgs=images, metas=metas)
        loss, details = loss_fn(dict(result, metas=metas, global_iter=epoch_index * 2048))
        if not torch.isfinite(loss):
            raise ValueError('smoke loss is not finite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
        model.zero_grad(set_to_none=True)
        value = float(loss.detach())
        horizons = sorted(result['gaussian_logits_by_horizon'])
        del images, metas, result, loss_fn, loss
        validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
        before = runner.capture_rng()
        metric = evaluate(model, validation, cfg, work, epoch_index + 1, limit=1)
        after = runner.capture_rng()
        if (before['python'] != after['python'] or not torch.equal(before['torch'], after['torch']) or
                any(not torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda'])) or
                not np.array_equal(before['numpy'][1], after['numpy'][1]) or
                before['numpy'][2:] != after['numpy'][2:]):
            raise ValueError('diagnostics modified random states')
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('diagnostics accumulated parameter gradients')
        runner.atomic_json(work / 'smoke_success.json', dict(
            status='passed', arm=arm, epoch_index=epoch_index, loss=value, metrics=metric,
            supervised_gaussian_horizons=horizons, initial_model_sha256=initial_hash,
            initialization=initialization, diagnostic_rng_preserved=True,
            diagnostic_parameter_gradients_untouched=True,
            config_sha256=runner.digest(ARMS[arm])))
        print(f'SMOKE_PASSED {arm} epoch_index={epoch_index} loss={value:.6f}', flush=True)
        del model, dataset, validation, before, after, details
        torch.cuda.empty_cache()
    if len(set(fingerprints.values())) != 1:
        raise ValueError('A and B did not start from identical model states')
    runner.atomic_json(ROOT / 'smoke' / 'all_passed.json', dict(status='passed', initial_model_sha256=fingerprints))
    print('ALL_AB_SMOKES_PASSED', flush=True)


if __name__ == '__main__':
    main()
