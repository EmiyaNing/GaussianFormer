"""Fixed-checkpoint position/semantic/amplitude interventions with exact regional counts."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from tools.pmg_world.regions import HORIZONS, REGIONS, histogram, load_masks, token_at
from tools.sparse_opus_gausisan_world import runtime
from .interventions import VARIANTS, associate, make_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 1024:
        parser.error("samples must be in 1..1024")
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output = Path(args.output)
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    if len(dataset) != 1024:
        raise ValueError("fixed diagnostic validation manifest changed")
    model = build_segmentor(cfg.model).cuda()
    saved = torch.load(cfg.diagnostic_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    epoch = runtime.load_evaluation_checkpoint(model, saved)
    del saved
    model.eval()
    model.requires_grad_(False)
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
    identity = dict(checkpoint=cfg.diagnostic_checkpoint, checkpoint_sha256=runtime.digest(cfg.diagnostic_checkpoint),
                    checkpoint_epoch=epoch, config_sha256=runtime.digest(args.config),
                    manifests=runtime.audit_manifests(cfg), variants=list(VARIANTS),
                    sample_indices=indices, horizons=list(HORIZONS),
                    source_sha256={str(p):runtime.digest(p) for p in sorted(Path("tools/pmg_world_stage3").glob("*.py"))},
                    training_updates=0, gt_assisted_diagnostic=True)
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != identity:
        raise ValueError("existing diagnostic protocol changed")
    runtime.atomic_json(protocol, identity)
    original_render = model.gaussian_head.render_state
    captured = []

    def capture(state, xyz):
        captured.append(type(state)(*(value.detach() if value is not None else None for value in state)))
        return original_render(state, xyz)

    model.gaussian_head.render_state = capture
    remaining = [i for i in indices if not (samples_dir / f"{i:04d}.npz").exists()]
    started = time.monotonic()
    with torch.no_grad():
        for index, data in zip(remaining, runtime.loader(dataset, cfg.val_loader, remaining)):
            captured.clear()
            model.memory.clear()
            model.memory_queue.clear()
            images, metas = runtime.move_batch(data, "cuda")
            result = model(imgs=images, metas=metas)
            assert len(captured) == 7
            count = int((result["query_stamps"] == 0).sum())
            assert count == 720
            token = token_at(dataset, index, 0)
            masks_path = Path(cfg.pmg_experiment.reference_directory) / "regions" / (token + ".npz")
            grids = [result["final_occ_grid"], *result["future_occ_grids"]]
            targets = [metas["occ_label"], *metas["future_occ_labels"].unbind(1)]
            xyz = metas["occ_xyz"].float().flatten(1, 3)
            numpy_xyz = xyz[0].cpu().numpy()
            global_counts = np.zeros((len(VARIANTS), 7, 18, 18), np.int64)
            region_counts = np.zeros((len(VARIANTS), 3, len(REGIONS), 18, 18), np.int64)
            partitions = np.zeros((len(VARIANTS), 3, 4), np.int64)
            coverage = []
            for horizon in range(7):
                gt = targets[horizon][0].long().reshape(-1).cpu().numpy()
                native = grids[horizon][0].reshape(-1).cpu().numpy()
                if horizon not in HORIZONS:
                    global_counts[:, horizon] = histogram(native, gt)
                    continue
                slot = HORIZONS.index(horizon)
                masks = load_masks(masks_path, horizon)
                mapping = associate(captured[horizon], numpy_xyz, gt, masks["fast"], count, radius=2.)
                states, checks = make_variants(captured[horizon], mapping, count, seed=20261023 + index * 7 + horizon)
                n = len(mapping["indices"])
                coverage.append(dict(index=index, horizon=horizon, fast_voxels=int(mapping["fast_voxels"]),
                    active_inherited_gaussians=int(mapping["active_gaussians"]), matched_gaussians=n,
                    matched_distinct_fast_voxels=int(len(np.unique(mapping["gt_indices"]))),
                    mean_association_distance_m=float(mapping["distances"].mean()) if n else None,
                    **checks))
                fast_ids = np.flatnonzero(masks["fast"])
                fast_tensor = torch.as_tensor(fast_ids, device=xyz.device, dtype=torch.long)
                fast_gt = torch.as_tensor(gt[fast_ids], device=xyz.device, dtype=torch.long)
                for variant_index, name in enumerate(VARIANTS):
                    logits = original_render(states[name], xyz)
                    prediction = logits.argmax(1)[0].cpu().numpy()
                    if name == "D00" and not np.array_equal(prediction, native):
                        raise ValueError("D00 re-render differs from native model output")
                    global_counts[variant_index, horizon] = histogram(prediction, gt)
                    for region_index, region in enumerate(REGIONS):
                        region_counts[variant_index, slot, region_index] = histogram(prediction, gt, masks[region])
                    scores = logits[0, :, fast_tensor]
                    fg, label = scores[:17].max(0)
                    missed = scores[17] > fg
                    correct = label == fast_gt
                    partitions[variant_index, slot] = [
                        int((~missed & correct).sum()), int((~missed & ~correct).sum()),
                        int((missed & correct).sum()), int((missed & ~correct).sum())]
                    del logits, scores
                del states
            np.savez_compressed(samples_dir / f"{index:04d}.npz", global_confusion=global_counts,
                                region_confusion=region_counts, readout_partitions=partitions)
            runtime.atomic_json(samples_dir / f"{index:04d}.json", dict(index=index, token=token,
                scene=str(dataset.keyframes[index][0]), coverage=coverage,
                native_prediction_parity=True, invariants_passed=True))
            del result, images, metas, grids, targets, data
            captured.clear()
            model.memory.clear()
            model.memory_queue.clear()
            done = sum((samples_dir / f"{i:04d}.npz").exists() for i in indices)
            row = dict(status="running", completed_samples=done, planned_samples=len(indices),
                       elapsed_seconds=time.monotonic() - started,
                       peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20)
            runtime.atomic_json(output / "progress.json", row)
            if done % 8 == 0 or done == len(indices):
                print("DIAGNOSIS " + json.dumps(row), flush=True)
    model.gaussian_head.render_state = original_render
    from .report import summarize
    summarize(output)
    runtime.atomic_json(output / "completed.json", dict(status="completed", num_samples=len(indices),
        checkpoint_epoch=epoch, training_updates=0, elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
