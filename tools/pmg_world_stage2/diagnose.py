"""Same-weight interventions with fixed centers; no optimizer updates or future images.

Distance statistics use every positive-opacity Gaussian center and three
predeclared distance thresholds. These are coverage proxies, not an exact
Gaussian support decomposition or instance-pose benchmark.
"""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.sparseworld_strict import _ego_pose
from model.encoder.opus_encoder import _opus_decode_points
from model.encoder.pmg_shape_projection import centered, fit_yaw
from model.segmentor.pmg_sparse_opus_world import transport_renderer_quaternion
from tools.sparse_opus_gausisan_world import runtime
from tools.pmg_world.regions import (
    HORIZONS,
    REGIONS,
    attributes_for,
    histogram,
    load_masks,
    record_at,
    token_at,
)


ROOT = Path("analysis/pmg_world_stage2")
VARIANTS = (
    "native",
    "redecode",
    "persist_semantics",
    "persist_shape",
    "persist_opacity",
    "persist_all",
)


def owner_indices(points, instances, pose):
    """Associate predicted centers inside GT oriented boxes; exclude overlaps."""
    from pyquaternion import Quaternion

    world = points @ pose[:3, :3].T + pose[:3, 3]
    owner = np.full(len(points), -1, dtype=np.int32)
    ambiguous = np.zeros(len(points), dtype=bool)
    for number, item in enumerate(instances):
        local = (world - np.asarray(item["translation"])) @ Quaternion(
            item["rotation"]
        ).rotation_matrix
        half = np.asarray(item["size"])[[1, 0, 2]] * 0.5
        inside = (np.abs(local) <= half + 1e-5).all(-1)
        ambiguous |= inside & (owner >= 0)
        owner[inside] = number
    owner[ambiguous] = -1
    return owner


def merge_history(base, historical, fields, count):
    values = {}
    for field in fields:
        old, new = getattr(base, field), getattr(historical, field)
        values[field] = torch.cat((new[:, :count], old[:, count:]), 1)
    return base._replace(**values)


def stats_for_points(state, xyz, target, prediction, fast, gate, reg, velocity):
    means = state.means[0].flatten(0, 1).cpu().numpy()
    opacity = state.opacities[0].reshape(-1).cpu().numpy()
    active = np.isfinite(means).all(-1) & (opacity > 0)
    flat_xyz = xyz[0].cpu().numpy()
    ids = np.flatnonzero(fast)
    result = dict(fast_voxels=len(ids), active_gaussians=int(active.sum()))
    if not len(ids) or not active.any():
        return result
    distance, nearest = cKDTree(means[active]).query(flat_xyz[ids], workers=1)
    nearest = np.flatnonzero(active)[nearest]
    missed = prediction[ids] == 17
    result.update(
        nearest_distance_sum_m=float(distance.sum()),
        nearest_distance_p50_m=float(np.median(distance)),
        nearest_distance_p90_m=float(np.quantile(distance, 0.9)),
        missed_voxels=int(missed.sum()),
        missed_nearest_distance_sum_m=float(distance[missed].sum()),
    )
    for threshold in (0.4, 0.8, 1.2):
        result[f"distance_gt_{threshold}m"] = int((distance > threshold).sum())
        result[f"missed_distance_gt_{threshold}m"] = int(
            (missed & (distance > threshold)).sum()
        )
    fg = state.semantics[0].flatten(0, 1).argmax(-1).cpu().numpy()[nearest]
    result.update(
        nearest_semantic_correct=int((fg == target[ids]).sum()),
        nearest_gate_closed=int((~gate.reshape(-1)[nearest]).sum()),
        nearest_reg_norm_sum_m=float(
            np.linalg.norm(reg.reshape(-1, 3)[nearest], axis=-1).sum()
        ),
        nearest_gated_velocity_norm_sum_m=float(
            np.linalg.norm(velocity.reshape(-1, 2)[nearest], axis=-1).sum()
        ),
    )
    return result


def aggregate(records):
    output = {}
    for horizon in HORIZONS:
        rows = [r for r in records if r["horizon"] == horizon]
        numeric = set.union(*(set(r) for r in rows)) if rows else set()
        sums = {
            k: sum(r.get(k, 0) for r in rows)
            for k in numeric
            if k not in ("index", "horizon")
            and "p50" not in k
            and "p90" not in k
            and not k.startswith("projection_")
        }
        n = sums.get("fast_voxels", 0)
        missed = sums.get("missed_voxels", 0)
        output[str(horizon * 0.5)] = dict(
            sums=sums,
            fast_voxels=n,
            missed_farther_than_0_8m_pct=(
                100 * sums.get("missed_distance_gt_0.8m", 0) / missed
                if missed
                else None
            ),
            nearest_gate_closed_pct=(
                100 * sums.get("nearest_gate_closed", 0) / n if n else None
            ),
            nearest_distance_mean_m=(
                sums.get("nearest_distance_sum_m", 0) / n if n else None
            ),
        )
        if rows and "projection_displacement_mean_m" in rows[0]:
            output[str(horizon * 0.5)]["projection_displacement_mean_m"] = float(
                np.mean([r["projection_displacement_mean_m"] for r in rows])
            )
            output[str(horizon * 0.5)]["projection_displacement_max_m"] = float(
                max(r["projection_displacement_max_m"] for r in rows)
            )
    return output


def run_arm(arm, indices, variants):
    cfg = Config.fromfile(
        f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py"
    )
    runtime.configure_runtime(cfg)
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    checkpoint = f"out/pmg_world_stage1/{arm}/best.pth"
    model = build_segmentor(cfg.model).cuda()
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    epoch = runtime.load_evaluation_checkpoint(model, saved)
    del saved
    model.eval()
    captured, reg_outputs, vel_outputs = [], [], []
    original = model.gaussian_head.render_state

    def capture(state, xyz):
        captured.append(state)
        return original(state, xyz)

    model.gaussian_head.render_state = capture
    handles = [
        model.reg_branch.register_forward_hook(
            lambda m, a, v: reg_outputs.append(v.detach())
        ),
        model.vel_branch.register_forward_hook(
            lambda m, a, v: vel_outputs.append(v.detach())
        ),
    ]
    confusions = {v: np.zeros((7, 18, 18), np.int64) for v in variants}
    region_counts = {v: np.zeros((3, len(REGIONS), 18, 18), np.int64) for v in variants}
    coverage, association = [], []
    started = time.monotonic()
    for index, batch in zip(indices, runtime.loader(dataset, cfg.val_loader, indices)):
        with torch.no_grad():
            images, metas = runtime.move_batch(batch, "cuda")
            captured.clear()
            reg_outputs.clear()
            vel_outputs.clear()
            result = model(imgs=images, metas=metas)
            states = list(captured)
            assert len(states) == 7 and len(reg_outputs) == 6
            count = int((result["query_stamps"] == 0).sum())
            source = states[0]
            encoder = model.pts_bbox_head.transformer.encoder
            _, _, intrinsic = encoder.attribute_branches[str(len(encoder.layers) - 1)](
                result["query_feat"]
            )
            intrinsic = intrinsic[:, result["query_stamps"] == 0]
            query_valid = result["representation"][-1]["query_valid_mask"][
                :, result["query_stamps"] == 0
            ]
            intrinsic = intrinsic * query_valid[:, :, None, None]
            template = centered(
                _opus_decode_points(
                    result["all_refine_pts"][-1][:, result["query_stamps"] == 0],
                    model.pc_range,
                )
            )
            xyz = metas["occ_xyz"].float().flatten(1, 3)
            masks_path = (
                Path(cfg.pmg_experiment.reference_directory)
                / "regions"
                / (token_at(dataset, index, 0) + ".npz")
            )
            attributes = attributes_for(
                dataset, index, cfg.pmg_experiment.attribute_root
            )
            for horizon in range(7):
                target = (
                    (
                        metas["occ_label"]
                        if horizon == 0
                        else metas["future_occ_labels"][:, horizon - 1]
                    )
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                native_prediction = (
                    (
                        result["final_occ_grid"]
                        if horizon == 0
                        else result["future_occ_grids"][horizon - 1]
                    )
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                masks = load_masks(masks_path, horizon) if horizon in HORIZONS else None
                if horizon:
                    points, classes, features = (
                        result[name][horizon - 1]
                        for name in (
                            "forecast_points_list",
                            "forecast_semantics_list",
                            "forecast_features_list",
                        )
                    )
                    decoded = encoder._make_gaussian(
                        features,
                        points,
                        classes,
                        torch.ones(
                            features.shape[:2], device=features.device, dtype=torch.bool
                        ),
                        len(encoder.layers) - 1,
                    )
                    angle = fit_yaw(
                        template,
                        centered(
                            _opus_decode_points(points[:, :count], model.pc_range)
                        ),
                    )
                    valid = torch.isfinite(points[:, :count]).all(-1) & (
                        (points[:, :count] >= 0) & (points[:, :count] < 1)
                    ).all(-1)
                    persistent = source._replace(
                        means=decoded.means[:, :count],
                        rotations=transport_renderer_quaternion(
                            source.rotations, angle
                        ),
                        opacities=intrinsic * valid[..., None],
                    )
                for variant in variants:
                    if horizon == 0 or variant == "native":
                        prediction = native_prediction
                    else:
                        fields = {
                            "persist_semantics": ("semantics",),
                            "persist_shape": ("scales", "rotations"),
                            "persist_opacity": ("opacities",),
                            "persist_all": (
                                "semantics",
                                "scales",
                                "rotations",
                                "opacities",
                            ),
                        }.get(variant, ())
                        state = merge_history(decoded, persistent, fields, count)
                        prediction = (
                            original(state, xyz).argmax(1).cpu().numpy().reshape(-1)
                        )
                    confusions[variant][horizon] += histogram(prediction, target)
                    if arm == "A0" and variant == "redecode":
                        assert np.array_equal(
                            prediction, native_prediction
                        ), "instrumented A0 re-decode differs from native output"
                    if masks is not None:
                        for j, name in enumerate(REGIONS):
                            region_counts[variant][
                                HORIZONS.index(horizon), j
                            ] += histogram(prediction, target, masks[name])
                if horizon in HORIZONS:
                    labels = result["forecast_semantics_list"][horizon - 1][0].argmax(
                        -1
                    )
                    gate = ((labels >= 2) & (labels <= 10)).cpu().numpy()
                    reg = (
                        (reg_outputs[horizon - 1][0].unflatten(-1, (-1, 3)) * 0.5)
                        .cpu()
                        .numpy()
                    )
                    vel = (
                        vel_outputs[horizon - 1][0].unflatten(-1, (-1, 2))
                    ).cpu().numpy() * gate[..., None]
                    row = dict(
                        index=index,
                        horizon=horizon,
                        **stats_for_points(
                            states[horizon],
                            xyz,
                            target,
                            native_prediction,
                            masks["fast"],
                            gate,
                            reg,
                            vel,
                        ),
                    )
                    if arm in ("A4", "C1"):
                        previous = (
                            result["all_refine_pts"][-1][:, result["query_stamps"] == 0]
                            if horizon == 1
                            else result["forecast_points_list"][horizon - 2][:, :count]
                        )
                        candidate = _opus_decode_points(previous, model.pc_range)[
                            0
                        ] + torch.from_numpy(
                            np.concatenate(
                                (reg[:count, :, :2] + vel[:count], reg[:count, :, 2:]),
                                -1,
                            )
                        ).to(
                            points
                        )
                        actual = _opus_decode_points(points[:, :count], model.pc_range)[
                            0
                        ]
                        row["projection_displacement_mean_m"] = float(
                            (actual - candidate).norm(dim=-1).mean()
                        )
                        row["projection_displacement_max_m"] = float(
                            (actual - candidate).norm(dim=-1).max()
                        )
                    coverage.append(row)
                    instances = attributes[HORIZONS.index(horizon)]
                    means = states[horizon].means[0].cpu().numpy()
                    owner = owner_indices(
                        means.reshape(-1, 3),
                        instances,
                        _ego_pose(record_at(dataset, index, horizon)),
                    ).reshape(means.shape[:2])
                    associated = owner >= 0
                    associated &= states[horizon].opacities[0, ..., 0].cpu().numpy() > 0
                    owner[~associated] = -1
                    fast_owner = np.asarray(
                        [
                            bool(
                                i.get("speed") is not None
                                and np.isfinite(i["speed"])
                                and i["speed"] >= 2
                            )
                            for i in instances
                        ]
                    )
                    fast_points = (
                        associated & fast_owner[np.maximum(owner, 0)]
                        if len(instances)
                        else associated
                    )
                    eligible = associated.sum(-1) >= 3
                    mixed = np.asarray(
                        [len(set(o[o >= 0].tolist())) > 1 for o in owner]
                    )
                    direction = np.sum(reg[..., :2] * vel, axis=-1)
                    association.append(
                        dict(
                            index=index,
                            horizon=horizon,
                            total_children=int(owner.size),
                            associated_children=int(associated.sum()),
                            fast_associated_children=int(fast_points.sum()),
                            fast_gate_closed_children=int((fast_points & ~gate).sum()),
                            eligible_queries=int(eligible.sum()),
                            multi_instance_queries=int((eligible & mixed).sum()),
                            fast_reg_velocity_opposed=int(
                                (fast_points & (direction < 0)).sum()
                            ),
                        )
                    )
            del result, states, captured[:], images, metas, batch
        if len(coverage) % 24 == 0:
            print(
                "DIAG",
                arm,
                index,
                "seconds",
                round(time.monotonic() - started, 1),
                flush=True,
            )
        runtime.atomic_json(
            ROOT / f"diagnosis_{arm}_progress.json",
            dict(
                status="running",
                arm=arm,
                last_index=index,
                samples_done=len(coverage) // 3,
                total_samples=len(indices),
            ),
        )
    for handle in handles:
        handle.remove()
    np.savez_compressed(
        ROOT / f"diagnosis_{arm}_counts.npz",
        **{f"global_{k}": v for k, v in confusions.items()},
        **{f"regions_{k}": v for k, v in region_counts.items()},
    )
    output = dict(
        status="completed",
        arm=arm,
        checkpoint=checkpoint,
        checkpoint_sha256=runtime.digest(checkpoint),
        checkpoint_epoch=epoch,
        sample_indices=indices,
        training_updates=0,
        coverage=coverage,
        coverage_summary=aggregate(coverage),
        association=association,
        variants=list(variants),
        seconds=time.monotonic() - started,
        caveats=[
            "Same-weight attribute interventions hold centers fixed but can be out of distribution.",
            "Available A0/A4/C1 weights have different completed epochs; cross-arm comparisons are exploratory.",
            "Nearest-center statistics include both historical and birth Gaussians; distance is not Gaussian support.",
            "GT boxes are used for diagnosis only; association at predicted positions is coverage-biased.",
        ],
    )
    runtime.atomic_json(ROOT / f"diagnosis_{arm}.json", output)
    print("FINISHED", arm, output["coverage_summary"], flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--arms", nargs="+", default=["A0", "A4", "C1"])
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, 1023, args.samples, dtype=int).tolist()
    for arm in args.arms:
        run_arm(arm, indices, ("native",) if args.native_only else VARIANTS)


if __name__ == "__main__":
    main()
