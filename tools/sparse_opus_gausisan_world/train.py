"""Train the integrated D model with the original 15-epoch optimization protocol.

Run from the repository root using ``python -m tools.sparse_opus_gausisan_world.train``.
All epochs include full validation and the same fixed-sample diagnostics as D.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from loss import OPENOCC_LOSS
from .evaluation import evaluate
from .protocol import MODEL_NAME, check_resume_identity, training_contract_sha256
from .report import refresh
from .runtime import (
    atomic_json,
    atomic_save,
    audit_manifests,
    capture_rng,
    configure_runtime,
    digest,
    loader,
    load_image_initialization,
    make_optimizer,
    model_fingerprint,
    move_batch,
    restore_rng,
    set_lr,
    should_restore_query_statistics,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="full-horizon forward/backward and one validation sample",
    )
    parser.add_argument(
        "--resume-from", help="Resume an original D or integrated checkpoint"
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Pause after diagnostics; keep the full 15-epoch LR budget",
    )
    args = parser.parse_args()
    if args.smoke and args.resume_from:
        parser.error("--smoke starts a scratch check; omit --resume-from")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= 15:
        parser.error("--stop-after-epoch must be between 1 and 15")
    cfg = Config.fromfile(args.config)
    if "_base_" in Path(args.config).read_text():
        raise ValueError("experiment configurations must be standalone")
    configure_runtime(cfg)
    directory = Path(args.work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if args.smoke and (directory / "latest.pth").exists():
        raise ValueError("smoke checks require a separate work directory")
    run_lock = open(directory / "run.lock", "w")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("this model work directory already has an active runner")
    run_lock.write(str(os.getpid()))
    run_lock.flush()
    config_hash = digest(args.config)
    manifest_hashes = audit_manifests(cfg)
    contract_hash = training_contract_sha256(cfg)
    cfg.dump(str(directory / "resolved_config.py"))
    model = build_segmentor(cfg.model)
    model.init_weights()
    initialization = load_image_initialization(model, cfg.train_load_from)
    atomic_json(directory / "initialization.json", initialization)
    initialization.update(
        initial_model_sha256=model_fingerprint(model), model_name=MODEL_NAME
    )
    atomic_json(directory / "initialization.json", initialization)
    model.cuda()
    optimizer = make_optimizer(model, cfg)
    loss_fn = OPENOCC_LOSS.build(cfg.loss).cuda()
    train_data = OPENOCC_DATASET.build(cfg.train_dataset_config)
    val_data = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if (len(train_data), len(val_data)) != (4096, 1024):
        raise ValueError("experiment dataset sizes differ from 4096/1024")
    batch_size = cfg.train_loader.batch_size
    batches_per_epoch = math.ceil(len(train_data) / batch_size)
    total_updates = (
        math.ceil(batches_per_epoch / cfg.gradient_accumulation) * cfg.max_epochs
    )
    if cfg.max_epochs != 15 or total_updates != 15360:
        raise ValueError(
            "D continuation must retain the original fifteen-epoch optimization budget"
        )
    epoch, next_iter, updates, best = 0, 0, 0, -1.0
    phase = "train"
    latest = directory / "latest.pth"
    resume_rng = None
    resume_stamps = None
    resume_path = Path(args.resume_from) if args.resume_from else latest
    if (args.resume_from or latest.exists()) and not args.smoke:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        identity = check_resume_identity(saved, cfg, config_hash, manifest_hashes)
        print(f"CHECKPOINT_IDENTITY {identity}", flush=True)
        required = {
            "optimizer",
            "epoch",
            "next_iter",
            "updates",
            "best_score",
            "phase",
            "rng",
            "query_stamps",
        }
        if not required.issubset(saved):
            raise ValueError(
                "training resume requires a complete latest/epoch checkpoint; selected weights are evaluation-only"
            )
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        epoch, next_iter, updates = saved["epoch"], saved["next_iter"], saved["updates"]
        best, phase, resume_rng = saved["best_score"], saved["phase"], saved["rng"]
        resume_stamps = saved["query_stamps"]
        atomic_json(
            directory / "resume_audit.json",
            dict(
                checkpoint=str(resume_path),
                checkpoint_sha256=digest(resume_path),
                identity=identity,
                epoch_index=epoch,
                next_sample_index=next_iter,
                updates=updates,
                phase=phase,
                optimizer_loaded=True,
                model_name=MODEL_NAME,
                training_contract_sha256=contract_hash,
            ),
        )
        del saved
        print(
            f"RESUME epoch={epoch+1} next_iter={next_iter} updates={updates} phase={phase}",
            flush=True,
        )

    def save(progress_epoch, progress_iter, progress_phase, path=latest):
        atomic_save(
            path,
            dict(
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=progress_epoch,
                next_iter=progress_iter,
                phase=progress_phase,
                updates=updates,
                best_score=best,
                rng=capture_rng(),
                model_name=MODEL_NAME,
                training_contract_sha256=contract_hash,
                config_sha256=config_hash,
                manifest_sha256=manifest_hashes,
                query_stamps=model.pts_bbox_head.ind_stamps_all,
                initialization=initialization,
            ),
        )

    if args.smoke:
        epoch = cfg.max_epochs - 1
    while epoch < cfg.max_epochs:
        model.train()
        # A completed epoch must take the same set_epoch path as uninterrupted training.
        # Only interrupted training/evaluation within an epoch restores its query statistics.
        within_epoch = should_restore_query_statistics(resume_rng, next_iter, phase)
        statistics = (
            model.pts_bbox_head.num_stamps_all.clone() if within_epoch else None
        )
        model.set_epoch(epoch)
        if statistics is not None:
            model.pts_bbox_head.num_stamps_all.copy_(statistics)
            model.pts_bbox_head.ind_stamps_all = resume_stamps.cuda()
            if not model.pretrain:
                model.pts_bbox_head.reset_mask()
        train_data.set_epoch(epoch)
        if resume_rng is not None:
            restore_rng(resume_rng)
            resume_rng = None
        order = (
            np.random.default_rng(cfg.seed + epoch)
            .permutation(len(train_data))
            .tolist()
        )
        data_loader = loader(
            train_data, cfg.train_loader, order[next_iter:], cfg.seed + epoch
        )
        started = time.monotonic()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        if phase != "evaluate":
            for local_index, data in enumerate(data_loader):
                index = next_iter + local_index * batch_size
                processed = min(index + batch_size, len(train_data))
                batch_index = index // batch_size
                images, metas = move_batch(data, "cuda")
                result = model(imgs=images, metas=metas)
                loss, details = loss_fn(
                    dict(
                        result,
                        metas=metas,
                        global_iter=epoch * batches_per_epoch + batch_index,
                    )
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch+1}, sample {index}"
                    )
                accumulation = min(
                    cfg.gradient_accumulation,
                    batches_per_epoch
                    - (batch_index // cfg.gradient_accumulation)
                    * cfg.gradient_accumulation,
                )
                (loss / accumulation).backward()
                losses.append(float(loss.detach()))
                boundary = (
                    batch_index + 1
                ) % cfg.gradient_accumulation == 0 or processed == len(train_data)
                if boundary or args.smoke:
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_max_norm, error_if_nonfinite=True
                    )
                    set_lr(
                        optimizer,
                        updates,
                        total_updates,
                        cfg.warmup_iters,
                        cfg.min_lr_ratio,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                if (local_index + 1) % cfg.print_freq == 0 or args.smoke:
                    elapsed = time.monotonic() - started
                    row = dict(
                        epoch=epoch + 1,
                        sample=processed,
                        updates=updates,
                        mean_loss=float(np.mean(losses[-cfg.print_freq :])),
                        gradient_norm_before_clipping=(
                            float(norm.detach()) if boundary or args.smoke else None
                        ),
                        seconds_per_sample=elapsed / (local_index + 1) / batch_size,
                        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                        empty_bias=(
                            float(model.gaussian_head.empty_bias.detach())
                            if hasattr(model, "gaussian_head")
                            else None
                        ),
                    )
                    print("TRAIN " + json.dumps(row), flush=True)
                    atomic_json(
                        directory / "progress.json", dict(status="training", **row)
                    )
                if args.smoke:
                    break
                if boundary and updates % cfg.checkpoint_interval_steps == 0:
                    save(epoch, processed, "train")
            if not args.smoke:
                save(epoch, len(train_data), "evaluate")
        metrics = evaluate(
            model, val_data, cfg, directory, epoch + 1, 1 if args.smoke else None
        )
        print("EPOCH " + json.dumps(metrics), flush=True)
        if args.smoke:
            atomic_json(
                directory / "smoke_success.json",
                dict(
                    status="passed",
                    metrics=metrics,
                    peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
                ),
            )
            return
        score = metrics["selection_mean_miou"]
        if score > best:
            best = score
            atomic_save(
                directory / "best_mIoU.pth",
                dict(
                    state_dict=model.state_dict(),
                    epoch_index=epoch,
                    metrics=metrics,
                    config_sha256=config_hash,
                    model_name=MODEL_NAME,
                    training_contract_sha256=contract_hash,
                    query_stamps=model.pts_bbox_head.ind_stamps_all,
                    manifest_sha256=manifest_hashes,
                    initialization=initialization,
                ),
            )
            atomic_json(directory / "best.json", metrics)
        save(epoch + 1, 0, "train", directory / f"epoch_{epoch+1:02d}.pth")
        shutil.copyfile(
            directory / f"epoch_{epoch+1:02d}.pth", directory / "latest.pth.tmp"
        )
        os.replace(directory / "latest.pth.tmp", latest)
        refresh(directory, Path(cfg.report_directory))
        if (
            args.stop_after_epoch is not None
            and epoch + 1 >= args.stop_after_epoch
            and epoch + 1 < cfg.max_epochs
        ):
            atomic_json(
                directory / "paused.json",
                dict(
                    status="paused_after_epoch",
                    trained_epochs=epoch + 1,
                    planned_epochs=cfg.max_epochs,
                    updates=updates,
                    model_name=MODEL_NAME,
                    checkpoint=str(latest),
                ),
            )
            return
        epoch, next_iter, phase = epoch + 1, 0, "train"
    if updates != total_updates:
        raise ValueError("D completion requires all 15360 optimizer updates")
    atomic_json(
        directory / "completed.json",
        dict(
            status="completed",
            epochs=15,
            updates=updates,
            best_score=best,
            config_sha256=config_hash,
            manifest_sha256=manifest_hashes,
        ),
    )


if __name__ == "__main__":
    main()
