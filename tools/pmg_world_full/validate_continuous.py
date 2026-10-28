"""Exercise real accumulation, eval/train switching and complete-state reload."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime
from .train import evaluate


def memory(stage, update, microbatch):
    torch.cuda.synchronize()
    row = dict(stage=stage, update=update, microbatch=microbatch,
               allocated_mib=torch.cuda.memory_allocated() / 2**20,
               reserved_mib=torch.cuda.memory_reserved() / 2**20,
               peak_mib=torch.cuda.max_memory_allocated() / 2**20)
    print("MEMORY " + json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output = Path(args.work_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "validation.json").exists():
        raise ValueError("use a new validation directory")
    dataset = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = runtime.load_image_initialization(model, cfg.train_load_from)
    model.cuda().train()
    model.set_epoch(0)
    optimizer = runtime.make_optimizer(model, cfg)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    order = np.random.default_rng(cfg.seed).permutation(len(dataset)).tolist()
    records, loss_values = [], []
    iterator = iter(runtime.loader(dataset, cfg.train_loader, order, cfg.seed))
    total_updates = int(np.ceil(np.ceil(len(dataset) / cfg.train_loader.batch_size) / cfg.gradient_accumulation)) * 15
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for update in range(10):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(cfg.gradient_accumulation):
            data = next(iterator)
            images, metas = runtime.move_batch(data, "cuda")
            records.append(memory("before_forward", update, micro))
            result = model(imgs=images, metas=metas)
            assert len(result["forecast_points_list"]) == 6
            assert set(result["gaussian_logits_by_horizon"]) == set(range(7))
            records.append(memory("after_forward", update, micro))
            loss, details = objective(dict(result, metas=metas, global_iter=update * cfg.gradient_accumulation + micro))
            assert torch.isfinite(loss)
            records.append(memory("after_loss", update, micro))
            (loss / cfg.gradient_accumulation).backward()
            records.append(memory("after_backward", update, micro))
            loss_values.append(float(loss.detach()))
            del result, loss, details, images, metas, data
            records.append(memory("after_cleanup", update, micro))
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True))
        runtime.set_lr(optimizer, update, total_updates, cfg.warmup_iters, cfg.min_lr_ratio)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        records.append(memory("after_update", update, -1))
        runtime.atomic_json(output / "progress.json", dict(status="validating", updates=update + 1,
                             mean_loss=float(np.mean(loss_values[-cfg.gradient_accumulation:])), gradient_norm=norm))
        if update == 7:
            evaluate(model, validation, cfg, output, 1, lambda: False, limit=1)
            records.append(memory("after_evaluation", update, -1))
        if update == 8:
            checkpoint = output / "reload_validation.pth"
            expected = runtime.model_fingerprint(model)
            runtime.atomic_save(checkpoint, dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                rng=runtime.capture_rng(), query_stamps=model.pts_bbox_head.ind_stamps_all,
                next_sample=9 * cfg.gradient_accumulation * cfg.train_loader.batch_size))
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = build_segmentor(cfg.model).cuda()
            model.load_state_dict(saved["state_dict"], strict=True)
            optimizer = runtime.make_optimizer(model, cfg)
            optimizer.load_state_dict(saved["optimizer"])
            counts = model.pts_bbox_head.num_stamps_all.clone()
            model.set_epoch(0)
            model.pts_bbox_head.num_stamps_all.copy_(counts)
            model.pts_bbox_head.ind_stamps_all = saved["query_stamps"].cuda()
            model.pts_bbox_head.reset_mask()
            runtime.restore_rng(saved["rng"])
            assert runtime.model_fingerprint(model) == expected
            assert saved["next_sample"] == 9 * cfg.gradient_accumulation * cfg.train_loader.batch_size
            del saved
            gc.collect()
            records.append(memory("after_reload", update, -1))
    runtime.atomic_json(output / "memory_records.json", records)
    runtime.atomic_json(output / "validation.json", dict(
        passed=True, updates=10, microbatches=10 * cfg.gradient_accumulation,
        real_gradient_accumulation=cfg.gradient_accumulation,
        microbatch_size=cfg.train_loader.batch_size, active_horizons=list(range(7)),
        evaluation_then_training=True, complete_state_reload_then_training=True,
        peak_cuda_mib=max(r["peak_mib"] for r in records),
        after_update_allocated_mib=[r["allocated_mib"] for r in records if r["stage"] == "after_update"],
        seconds=time.monotonic() - started, config_sha256=runtime.digest(args.config),
        initialization=initialization))
    print("VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
