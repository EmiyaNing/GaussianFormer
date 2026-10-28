"""Frozen Gaussian F/P diagnosis and fixed evaluation regions; no training."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from tools.sparse_opus_gausisan_world import runtime
from tools.sparse_opus_gausisan_world.metrics import HorizonMetrics
from .regions import (HORIZONS, REGIONS, attributes_for, histogram, pack_masks,
                      record_at, region_result, speed_masks, token_at)


def future_observation(dataset, index, horizon):
    """Build a true five-frame P window, without requiring future-of-P labels."""
    scene, _ = dataset.keyframes[index]
    info = record_at(dataset, index, horizon)
    frame = next(i for i, item in enumerate(dataset.scene_infos[scene]) if item.get("token") == info["token"])
    result = dataset.get_data_info(deepcopy(info), scene_token=scene, frame_index=frame)
    keys = dataset.keyframe_indices_by_scene[scene]
    position = keys.index(frame)
    if position < 4:
        raise ValueError("P window lacks real history")
    result["history_context"]["history_indices"] = [keys[position - step] for step in range(1, 5)]
    result["aug_configs"] = dataset._sample_augmentation()
    for transform in dataset.pipeline:
        if transform.__class__.__name__ != "LoadSparseWorldFutureOccupancy":
            result = transform(result)
    names = ("img", "projection_mat", "image_wh", "occ_label", "occ_xyz")
    return custom_collate_fn_temporal([{name: result[name] for name in names}])


def perception_grid(model, images, metas, source_ego):
    ego = model.plan_head(source_ego.to(images).reshape(images.shape[0], 1, -1))
    scale = (torch.tanh(model.points_scale_branch(ego)) + 1) * .35 + .8
    outputs = model.pts_bbox_head(model.extract_feat(images, metas), metas, points_scale=scale)
    source = outputs["representation"][-1]["gaussian"]
    current = model.pts_bbox_head.ind_stamps_all == 0
    state = type(source)(*(value[:, current] if value is not None else None for value in source))
    return model.gaussian_head.render_state(state, metas["occ_xyz"].float().flatten(1, 3)).argmax(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "regions").mkdir(exist_ok=True)
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    model = build_segmentor(cfg.model).cuda()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime.load_evaluation_checkpoint(model, saved)
    del saved
    model.eval()
    if hasattr(model.gaussian_head, "analytic_empty"):
        model.gaussian_head.analytic_empty = True
    metric = HorizonMetrics()
    region_counts = np.zeros((3, len(REGIONS), 18, 18), dtype=np.int64)
    perception_counts = np.zeros_like(region_counts)
    coverage = []
    started = time.monotonic()
    protocol = dict(config=args.config, checkpoint=args.checkpoint,
                    checkpoint_sha256=runtime.digest(args.checkpoint),
                    manifests=runtime.audit_manifests(cfg),
                    samples=args.limit or len(dataset),
                    future_images="diagnosis_only", training_updates=0,
                    source="Gaussian D; separate from the original SparseWorld report",
                    attribute_root=cfg.pmg_experiment.attribute_root)
    runtime.atomic_json(output / "protocol.json", protocol)
    with torch.no_grad():
        for index, data in enumerate(runtime.loader(dataset, cfg.val_loader)):
            if args.limit and index >= args.limit:
                break
            images, metas = runtime.move_batch(data, "cuda")
            result = model(imgs=images, metas=metas)
            metric.add(result, metas)
            token = token_at(dataset, index, 0)
            attributes = attributes_for(dataset, index, cfg.pmg_experiment.attribute_root)
            arrays, sample_regions = {}, []
            for slot, horizon in enumerate(HORIZONS):
                observation = future_observation(dataset, index, horizon)
                future_images, future_metas = runtime.move_batch(observation, "cuda")
                pred_p = perception_grid(model, future_images, future_metas,
                                         metas["temporal_ego_states"]).cpu().numpy().reshape(-1)
                pred_f = result["future_occ_grids"][horizon - 1].cpu().numpy().reshape(-1)
                target = metas["future_occ_labels"][0, horizon - 1].cpu().numpy().reshape(-1)
                if not np.array_equal(future_metas["occ_label"].cpu().numpy().reshape(-1), target):
                    raise ValueError("F/P targets differ in native future coordinates")
                masks, denominators = speed_masks(
                    dataset, index, horizon, target,
                    metas["occ_xyz"][0].cpu().numpy(), attributes)
                occupied = (target >= 0) & (target < 17)
                correct_f, correct_p = pred_f == target, pred_p == target
                masks.update(F_only=occupied & correct_f & ~correct_p,
                             P_only=occupied & correct_p & ~correct_f,
                             both_correct=occupied & correct_f & correct_p,
                             both_wrong=occupied & ~correct_f & ~correct_p)
                arrays.update({f"h{horizon}_{key}": value for key, value in pack_masks(masks).items()})
                for region_index, name in enumerate(REGIONS):
                    region_counts[slot, region_index] += histogram(pred_f, target, masks[name])
                    perception_counts[slot, region_index] += histogram(pred_p, target, masks[name])
                sample_regions.append(dict(horizon=horizon, **denominators))
                del future_images, future_metas, observation
            temporary = output / "regions" / (token + ".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, output / "regions" / (token + ".npz"))
            coverage.append(dict(token=token, horizons=sample_regions))
            del result, images, metas, data
            if (index + 1) % 25 == 0 or args.limit:
                print("REFERENCE", index + 1, "seconds", round(time.monotonic() - started, 1), flush=True)
                runtime.atomic_json(output / "progress.json", dict(samples=index + 1, planned=protocol["samples"]))
    final = dict(protocol, status="completed", global_metrics=metric.results(),
                 regions=[dict(seconds=horizon * .5, **{
                     name: dict(F=region_result(region_counts[slot, i]),
                                P=region_result(perception_counts[slot, i]))
                     for i, name in enumerate(REGIONS)}) for slot, horizon in enumerate(HORIZONS)],
                 coverage=coverage, seconds=time.monotonic() - started)
    runtime.atomic_json(output / "results.json", final)
    np.savez_compressed(output / "counts.npz", F=region_counts, P=perception_counts,
                        global_confusion=metric.confusion)
    runtime.atomic_json(output / "completed.json", dict(status="completed", samples=metric.samples))
    runtime.atomic_json(output / "progress.json", dict(status="completed", samples=metric.samples,
                         planned=protocol["samples"]))


if __name__ == "__main__":
    main()
