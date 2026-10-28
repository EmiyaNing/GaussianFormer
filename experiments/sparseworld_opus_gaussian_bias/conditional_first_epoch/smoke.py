"""Verify current/future C objectives and complete new diagnostics on GPU."""
import json

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from .. import train as runner
from ..ab_first_epoch.train import model_fingerprint
from .diagnostics import evaluate
from .loss import conditional_semantic_ce
from .protocol import BASE_CONFIG, BASE_WORK, CONFIG, ROOT


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    expected = json.loads((BASE_WORK / 'initialization.json').read_text())['initial_model_sha256']
    checks = []
    for epoch_index in (0, 14):
        print(f'C_SMOKE_START epoch={epoch_index+1}', flush=True)
        cfg = Config.fromfile(str(CONFIG))
        work = ROOT / 'smoke' / f'epoch_{epoch_index+1:02d}'
        work.mkdir(parents=True, exist_ok=True)
        set_random_seed(cfg.seed)
        model = build_segmentor(cfg.model)
        model.init_weights()
        initialization = runner.load_image_initialization(model, cfg.train_load_from)
        fingerprint = model_fingerprint(model)
        if fingerprint != expected:
            raise ValueError('C initial model differs from B')
        model.cuda().train()
        model.set_epoch(epoch_index)
        data = OPENOCC_DATASET.build(cfg.train_dataset_config)
        data.set_epoch(epoch_index)
        images, metas = runner.move_batch(next(iter(runner.loader(data, cfg.train_loader))), 'cuda')
        loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
        baseline = OPENOCC_LOSS.build(Config.fromfile(str(BASE_CONFIG)).loss.occupancy_loss).cuda()
        result = model(imgs=images, metas=metas)
        loss, details = loss_fn(dict(result, metas=metas, global_iter=epoch_index * 2048))
        terms = []
        for h, logits in result['gaussian_logits_by_horizon'].items():
            labels = (metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h - 1]).flatten(1)
            actual = loss_fn.occupancy.loss_voxel([logits], metas['occ_xyz'].flatten(1, 3), labels)
            base = baseline.loss_voxel([logits], metas['occ_xyz'].flatten(1, 3), labels)
            auxiliary = conditional_semantic_ce(logits, labels, loss_fn.occupancy.class_weights)
            torch.testing.assert_close(actual, base + auxiliary, atol=1e-4, rtol=1e-5)
            terms.append(dict(horizon=h, base=float(base.detach()), auxiliary=float(auxiliary.detach()), total=float(actual.detach())))
            del actual, base, auxiliary
        if not torch.isfinite(loss):
            raise ValueError('C smoke loss is not finite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
        model.zero_grad(set_to_none=True)
        loss_value = float(loss.detach())
        horizons = sorted(result['gaussian_logits_by_horizon'])
        del result, images, metas, loss, loss_fn, baseline, details, logits, labels
        validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
        before = runner.capture_rng()
        # Install the same ordinary evaluator used by the verified boundary entry.
        runner.evaluate_original = runner.evaluate
        try:
            metrics = evaluate(model, validation, cfg, work, epoch_index + 1, limit=1)
        finally:
            del runner.evaluate_original
        after = runner.capture_rng()
        if (before['python'] != after['python'] or not torch.equal(before['torch'], after['torch']) or
                any(not torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda'])) or
                not np.array_equal(before['numpy'][1], after['numpy'][1]) or before['numpy'][2:] != after['numpy'][2:]):
            raise ValueError('C diagnostics modified RNG states')
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('C diagnostics accumulated model parameter gradients')
        record = dict(status='passed', epoch_index=epoch_index, total_loss=loss_value,
                      supervised_gaussian_horizons=horizons, conditional_terms=terms,
                      metrics=metrics, initialization=initialization, initial_model_sha256=fingerprint,
                      initial_model_matches_B=True, config_sha256=runner.digest(CONFIG),
                      diagnostic_rng_preserved=True, diagnostic_parameter_gradients_untouched=True)
        runner.atomic_json(work / 'smoke_success.json', record)
        checks.append(record)
        print(f'C_SMOKE_PASSED epoch={epoch_index+1} horizons={horizons} loss={loss_value:.6f}', flush=True)
        del model, data, validation, before, after
        torch.cuda.empty_cache()
    runner.atomic_json(ROOT / 'smoke' / 'all_passed.json', dict(status='passed', checks=checks,
                      initial_model_sha256=expected, config_sha256=runner.digest(CONFIG)))
    print('ALL_C_SMOKES_PASSED', flush=True)


if __name__ == '__main__':
    main()
