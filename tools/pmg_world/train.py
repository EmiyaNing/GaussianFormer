"""Matched 15-epoch PMG ablations with resumable, per-epoch diagnosis.

Only newly created experiment directories are written. Stateful resume checks
configuration, immutable manifests, implementation hashes and optimizer state.
"""

import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.protocol import training_contract_sha256
from .evaluate import evaluate, refresh


def implementation_identity():
    paths = [Path("model/segmentor/pmg_sparse_opus_world.py"),
             Path("model/encoder/pmg_shape_projection.py"),
             Path("model/segmentor/sparse_opus_gausisan_world.py"),
             Path("model/segmentor/sparseworld_strict.py"),
             Path("model/head/sparse_opus_gausisan_world_head.py"),
             Path("model/encoder/sparse_opus_gausisan_world_encoder.py"),
             Path("model/encoder/opus_encoder.py"),
             Path("model/encoder/opus_v1_gaussian_encoder.py"),
             Path("loss/sparse_opus_gausisan_world_loss.py"),
             Path("loss/sparseworld_strict_loss.py"),
             Path("dataset/sparse_opus_gausisan_world_dataset.py"),
             Path("dataset/sparseworld_strict.py")]
    paths += sorted(Path("tools/pmg_world").glob("*.py"))
    for suffix in ("*.py", "*.cpp", "*.cu", "*.h"):
        paths += sorted(Path("model/ops/sparse_localagg").rglob(suffix))
    from model.ops.sparse_localagg import _C
    paths.append(Path(_C.__file__))
    return {str(path): runtime.digest(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--stop-after-epoch", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.stop_after_epoch <= 15:
        parser.error("stop-after-epoch must be in 1..15")
    if "_base_" in Path(args.config).read_text():
        raise ValueError("standalone configuration required")
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    if cfg.max_epochs != 15 or cfg.model.finetune_epoch != 0:
        raise ValueError("all arms use 15 formal epochs starting at epoch 1")
    directory, report = Path(args.work_dir), Path(cfg.report_directory)
    directory.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = runtime.load_image_initialization(model, cfg.train_load_from)
    initialization["initial_model_sha256"] = runtime.model_fingerprint(model)
    model.cuda()
    optimizer = runtime.make_optimizer(model, cfg)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train_data = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation_data = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if (len(train_data), len(validation_data)) != (4096, 1024):
        raise ValueError("the stage-1 comparison must retain 4096/1024 samples")
    batches = math.ceil(len(train_data) / cfg.train_loader.batch_size)
    total_updates = math.ceil(batches / cfg.gradient_accumulation) * 15
    if total_updates != 15360:
        raise ValueError("optimizer-update budget changed")
    identity = dict(training_contract=training_contract_sha256(cfg),
                    manifests=runtime.audit_manifests(cfg), source_sha256=implementation_identity())
    latest = directory / "latest.pth"
    epoch, next_sample, updates, best, phase = 0, 0, 0, -1., "train"
    resume_rng, resume_stamps = None, None
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("resume implementation/configuration/data identity changed")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        epoch, next_sample, updates, best, phase = (saved[k] for k in
                                                 ("epoch", "next_sample", "updates", "best", "phase"))
        resume_rng, resume_stamps = saved["rng"], saved["query_stamps"]
        initialization = saved["initialization"]
        del saved
    else:
        # An independent generator makes sampling order unrelated to model construction.
        # All modes have identical parameter sets and initialized tensor hashes.
        runtime.atomic_json(directory / "initialization.json", initialization)
    cfg.dump(str(directory / "resolved_config.py"))
    runtime.atomic_json(directory / "protocol.json", dict(identity, arm=cfg.pmg_experiment.arm,
                         total_updates=total_updates, planned_epochs=15,
                         prediction_inputs="five historical frames only", distillation=False))

    def save(progress_epoch, sample, progress_phase):
        runtime.atomic_save(latest, dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                           epoch=progress_epoch, next_sample=sample, updates=updates, best=best,
                           phase=progress_phase, rng=runtime.capture_rng(),
                           query_stamps=model.pts_bbox_head.ind_stamps_all,
                           initialization=initialization, identity=identity))

    while epoch < min(args.stop_after_epoch, 15):
        model.train()
        within_epoch = runtime.should_restore_query_statistics(resume_rng, next_sample, phase)
        statistics = model.pts_bbox_head.num_stamps_all.clone() if within_epoch else None
        model.set_epoch(epoch)
        if statistics is not None:
            model.pts_bbox_head.num_stamps_all.copy_(statistics)
            model.pts_bbox_head.ind_stamps_all = resume_stamps.cuda()
            model.pts_bbox_head.reset_mask()
        train_data.set_epoch(epoch)
        if resume_rng is not None:
            runtime.restore_rng(resume_rng)
            resume_rng = None
        order = np.random.default_rng(cfg.seed + epoch).permutation(len(train_data)).tolist()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses = []
        if phase != "evaluate":
            for local_index, data in enumerate(runtime.loader(
                    train_data, cfg.train_loader, order[next_sample:], cfg.seed + epoch)):
                sample = next_sample + local_index * cfg.train_loader.batch_size
                processed = min(sample + cfg.train_loader.batch_size, len(train_data))
                batch_index = sample // cfg.train_loader.batch_size
                images, metas = runtime.move_batch(data, "cuda")
                result = model(imgs=images, metas=metas)
                inputs = dict(result, metas=metas, global_iter=epoch * batches + batch_index)
                loss, details = objective(inputs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss in epoch {epoch+1}, sample {sample}")
                (loss / cfg.gradient_accumulation).backward()
                losses.append(float(loss.detach()))
                boundary = (batch_index + 1) % cfg.gradient_accumulation == 0
                gradient_norm = None
                if boundary:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm,
                                                          error_if_nonfinite=True)
                    gradient_norm = float(norm)
                    runtime.set_lr(optimizer, updates, total_updates, cfg.warmup_iters, cfg.min_lr_ratio)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                # Tensor values never escape their batch graph; logs contain scalars.
                del result, inputs, loss, details, images, metas, data
                if (batch_index + 1) % cfg.print_freq == 0:
                    row = dict(status="training", arm=cfg.pmg_experiment.arm, epoch=epoch + 1,
                               samples=processed, updates=updates, planned_epochs=15,
                               mean_loss=float(np.mean(losses[-cfg.print_freq:])),
                               gradient_norm=gradient_norm,
                               seconds_per_microbatch=(time.monotonic() - started) / (local_index + 1),
                               peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                               allocated_after_cleanup_mib=torch.cuda.memory_allocated() / 2**20)
                    print("TRAIN " + json.dumps(row), flush=True)
                    runtime.atomic_json(directory / "progress.json", row)
                if boundary and updates % cfg.checkpoint_interval_steps == 0:
                    save(epoch, processed, "train")
            save(epoch, len(train_data), "evaluate")
        metrics = evaluate(model, validation_data, cfg, report, epoch + 1)
        score = metrics["selection_mean_miou"]
        if score > best:
            best = score
            runtime.atomic_save(directory / "best.pth", dict(state_dict=model.state_dict(),
                                epoch_index=epoch, metrics=metrics,
                                query_stamps=model.pts_bbox_head.ind_stamps_all, identity=identity))
        save(epoch + 1, 0, "train")
        refresh(cfg)
        print("EPOCH", epoch + 1, "selection_mean_miou", score, flush=True)
        epoch, next_sample, phase = epoch + 1, 0, "train"
        runtime.atomic_json(directory / "progress.json", dict(status="epoch_completed", epoch=epoch,
                             arm=cfg.pmg_experiment.arm, updates=updates, best=best, planned_epochs=15))
    if epoch == 15:
        if updates != total_updates:
            raise ValueError("completed epochs without the required updates")
        runtime.atomic_json(directory / "progress.json", dict(status="verifying_best_checkpoint",
                             epoch=15, arm=cfg.pmg_experiment.arm, updates=updates))
        del model, optimizer, objective
        torch.cuda.empty_cache()
        from .verify_best import verify_best
        verification = verify_best(cfg, directory, validation_data)
        runtime.atomic_json(directory / "completed.json", dict(status="completed", epochs=15,
                             updates=updates, best=best, arm=cfg.pmg_experiment.arm,
                             best_checkpoint_verified=verification["passed"]))


if __name__ == "__main__":
    main()
