"""Short controlled adaptation: freeze one T-A0, train only the position correction head."""

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
from tools.pmg_world.evaluate import evaluate
from tools.sparse_opus_gausisan_world import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output, report = Path(args.work_dir), Path(cfg.report_directory)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        return
    model = build_segmentor(cfg.model)
    saved = torch.load(cfg.adaptation_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    incompatible = model.load_state_dict(saved["state_dict"], strict=False)
    if incompatible.unexpected_keys or any(not k.startswith("motion_correction.") for k in incompatible.missing_keys):
        raise ValueError("frozen baseline parameter namespace changed")
    model.cuda()
    counts = model.pts_bbox_head.num_stamps_all.clone()
    model.set_epoch(int(saved["epoch_index"]))
    model.pts_bbox_head.num_stamps_all.copy_(counts)
    model.pts_bbox_head.ind_stamps_all = saved["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    del saved
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("motion_correction."))
    optimizer = torch.optim.AdamW(model.motion_correction.parameters(), lr=2e-4, weight_decay=.01)
    objective = OPENOCC_LOSS.build(cfg.loss).cuda()
    train = OPENOCC_DATASET.build(cfg.train_dataset_config)
    validation = OPENOCC_DATASET.build(cfg.val_dataset_config)
    order = np.random.default_rng(20261023).permutation(len(train)).tolist()
    accumulation = cfg.gradient_accumulation
    planned_updates = 1 if args.validate_only else cfg.adaptation_updates
    required = planned_updates * accumulation
    if required > len(train):
        raise ValueError("the short adaptation must not silently repeat samples")
    order = order[:required]
    cfg.dump(str(output / "resolved_config.py"))
    runtime.atomic_json(output / "protocol.json", dict(
        arm=cfg.pmg_experiment.arm, source_checkpoint=cfg.adaptation_checkpoint,
        source_sha256=runtime.digest(cfg.adaptation_checkpoint), config_sha256=runtime.digest(args.config),
        actual_updates=planned_updates, validation_only=args.validate_only, microbatch_size=1, accumulation=accumulation,
        training_sample_indices=order, sample_order_seed=20261023,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_native=True, native_dropout_disabled=True, distillation=False,
        initialization_note="Same pretrained T-A0 state; identical correction-head initialization across arms",
        objective="Original strict/world/trajectory and dense-time voxel losses; no GT speed or instance inputs"))
    # Native modules remain deterministic and frozen; only top-level training
    # enables future differentiable rendering and range masks.
    model.train()
    for name, child in model.named_children():
        if name != "motion_correction":
            child.eval()
    optimizer.zero_grad(set_to_none=True)
    losses, started, updates = [], time.monotonic(), 0
    for microbatch, data in enumerate(runtime.loader(train, cfg.train_loader, order, cfg.seed)):
        images, metas = runtime.move_batch(data, "cuda")
        result = model(imgs=images, metas=metas)
        if set(result["gaussian_logits_by_horizon"]) != set(range(7)):
            raise ValueError("all seven voxel losses must remain active")
        inputs = dict(result, metas=metas, global_iter=microbatch, update_stamp_statistics=False)
        loss, details = objective(inputs)
        assert torch.isfinite(loss)
        (loss / accumulation).backward()
        losses.append(float(loss.detach()))
        del result, inputs, loss, details, images, metas, data
        if (microbatch + 1) % accumulation:
            continue
        norm = float(torch.nn.utils.clip_grad_norm_(model.motion_correction.parameters(), 5., error_if_nonfinite=True))
        factor = (1/3 + 2/3 * updates/32) if updates < 32 else (.001 + .999 * .5 * (1 + np.cos(np.pi * (updates-32)/max(cfg.adaptation_updates-32,1))))
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * float(factor)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        if updates % 16 == 0:
            progress = dict(status="training", updates=updates, planned_updates=cfg.adaptation_updates,
                mean_loss=float(np.mean(losses[-16 * accumulation:])), gradient_norm=norm,
                elapsed_seconds=time.monotonic()-started,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            print("ADAPTATION " + json.dumps(progress), flush=True)
            runtime.atomic_json(output / "progress.json", progress)
    assert updates == planned_updates
    runtime.atomic_save(output / "latest.pth", dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
        epoch_index=model.curr_epoch, updates=updates, query_stamps=model.pts_bbox_head.ind_stamps_all,
        rng=runtime.capture_rng()))
    del optimizer, objective
    gc.collect()
    torch.cuda.empty_cache()
    metrics = evaluate(model, validation, cfg, output if args.validate_only else report, 1, limit=2 if args.validate_only else None)
    runtime.atomic_json(output / ("validation.json" if args.validate_only else "completed.json"), dict(status="passed" if args.validate_only else "completed", updates=updates,
        selection_mean_miou=metrics["selection_mean_miou"], elapsed_seconds=time.monotonic()-started))
    print("ADAPTATION_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
