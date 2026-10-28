"""Verify both first-epoch and full-curriculum routes before formal training."""
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from .. import train as runner
from ..ab_first_epoch.train import model_fingerprint
from ..ab_first_epoch.report import read
from ..conditional_first_epoch.loss import conditional_semantic_ce
from .diagnostics import evaluate
from .model import predict_factorized
from .protocol import ANALYSIS, ARMS, C_CONFIG, C_WORK, D, E, NEW_BRANCH_PREFIX, ROOT
from .train import initial_fingerprint


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    expected = read(C_WORK / 'initialization.json')['initial_model_sha256']
    checks = []
    for name, config in ARMS.items():
        for epoch_index in (0, 14):
            print(f'DE_SMOKE_START arm={name} epoch={epoch_index+1}', flush=True)
            cfg = Config.fromfile(str(config))
            work = ROOT / 'smoke' / f'{name}_epoch_{epoch_index+1:02d}'
            work.mkdir(parents=True, exist_ok=True)
            set_random_seed(cfg.seed)
            model = build_segmentor(cfg.model)
            model.init_weights()
            initialization = runner.load_image_initialization(model, cfg.train_load_from)
            shared = initial_fingerprint(model)
            if shared != expected:
                raise ValueError(f'{name} changed an unchanged C initialization tensor')
            full = model_fingerprint(model)
            model.cuda().train()
            model.set_epoch(epoch_index)
            data = OPENOCC_DATASET.build(cfg.train_dataset_config)
            data.set_epoch(epoch_index)
            images, metas = runner.move_batch(next(iter(runner.loader(data, cfg.train_loader))), 'cuda')
            loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
            result = model(imgs=images, metas=metas)
            loss, details = loss_fn(dict(result, metas=metas, global_iter=epoch_index * 2048))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite route loss')
            if name == D:
                base = OPENOCC_LOSS.build(Config.fromfile(str(C_CONFIG)).loss.occupancy_loss).cuda()
                for h, logits in result['gaussian_logits_by_horizon'].items():
                    labels = (metas['occ_label'] if h == 0 else metas['future_occ_labels'][:, h-1]).flatten(1)
                    actual = loss_fn.occupancy.loss_voxel([logits], metas['occ_xyz'].flatten(1, 3), labels)
                    expected_loss = (base.loss_voxel([logits], metas['occ_xyz'].flatten(1, 3), labels)
                        - conditional_semantic_ce(logits, labels, base.class_weights)
                        + conditional_semantic_ce(logits, labels, loss_fn.occupancy.conditional_class_weights))
                    torch.testing.assert_close(actual, expected_loss, atol=1e-4, rtol=1e-5)
                    del actual, expected_loss, logits, labels
                del base
            else:
                for h, logits in result['gaussian_logits_by_horizon'].items():
                    torch.testing.assert_close(logits.exp().sum(1), torch.ones_like(logits[:, 0]), atol=2e-6, rtol=2e-6)
                    grid = result['final_occ_grid'] if h == 0 else result['future_occ_grids'][h-1]
                    torch.testing.assert_close(predict_factorized(logits).view_as(grid), grid)
                    del logits, grid
            loss.backward()
            if name == E:
                heads = [p for n, p in model.named_parameters() if n.startswith(NEW_BRANCH_PREFIX)]
                if not heads or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads) or not any(p.grad.abs().sum() > 0 for p in heads):
                    raise ValueError('independent occupancy branch received no finite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True)
            model.zero_grad(set_to_none=True)
            value, horizons = float(loss.detach()), sorted(result['gaussian_logits_by_horizon'])
            expected_horizons = [0] if epoch_index == 0 else [0, 2, 4, 6]
            if horizons != expected_horizons:
                raise ValueError('future Gaussian curriculum changed')
            del result, images, metas, loss, loss_fn, details
            val = OPENOCC_DATASET.build(cfg.val_dataset_config)
            before = runner.capture_rng()
            parameter_before = model_fingerprint(model)
            metrics = evaluate(model, val, cfg, work, epoch_index+1, limit=1)
            after = runner.capture_rng()
            if (before['python'] != after['python'] or not torch.equal(before['torch'], after['torch']) or
                    any(not torch.equal(a, b) for a, b in zip(before['cuda'], after['cuda'])) or
                    not np.array_equal(before['numpy'][1], after['numpy'][1]) or before['numpy'][2:] != after['numpy'][2:]):
                raise ValueError('diagnostics changed RNG state')
            # Timestamp buffers are evaluator state; parameters must stay intact.
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError('diagnostics accumulated parameter gradients')
            record = dict(status='passed', arm=name, epoch_index=epoch_index, total_loss=value,
                supervised_gaussian_horizons=horizons, metrics=metrics, initialization=initialization,
                shared_initial_model_sha256=shared, initial_model_sha256=full, config_sha256=runner.digest(config),
                diagnostic_rng_preserved=True, diagnostic_parameter_gradients_untouched=True)
            runner.atomic_json(work / 'smoke_success.json', record)
            checks.append(record)
            print(f'DE_SMOKE_PASSED arm={name} epoch={epoch_index+1} loss={value:.6f}', flush=True)
            del model, data, val, before, after
            torch.cuda.empty_cache()
    runner.atomic_json(ROOT / 'smoke/all_passed.json', dict(status='passed', checks=checks,
        configuration_audit_sha256=runner.digest(ANALYSIS / 'configuration_audit.json')))
    print('ALL_DE_SMOKES_PASSED', flush=True)


if __name__ == '__main__':
    main()

