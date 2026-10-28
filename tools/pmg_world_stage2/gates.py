"""Evaluation-only interventions on the semantic velocity gate.

The native-mode implementation is checked against each original model before
soft/open gates are evaluated. Class logits and rendered semantic sources are
not overwritten. No GT enters the forecast or gate.
"""

import argparse
import gc
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from dataset.sparseworld_strict import _ego_pose
from model.encoder.opus_encoder import _opus_decode_points, _opus_encode_points
from model.encoder.pmg_shape_projection import centered, structured_update
from tools.sparse_opus_gausisan_world import runtime
from tools.pmg_world.regions import (
    HORIZONS,
    REGIONS,
    histogram,
    load_masks,
    token_at,
    attributes_for,
    record_at,
)
from .diagnose import owner_indices


@torch.no_grad()
def diagnostic_forecast(self, outputs, ego_feature, metas):
    if self.training:
        raise RuntimeError("diagnostic gates are evaluation-only")
    batch = ego_feature.shape[0]
    stamps = self.pts_bbox_head.ind_stamps_all
    current = stamps == 0
    all_features = outputs["query_feat"]
    all_points = outputs["all_refine_pts"][-1]
    features = all_features[:, current]
    points = all_points[:, current].detach()
    template = centered(_opus_decode_points(points, self.pc_range)).detach()
    timestamp = points.new_zeros(batch, self.num_query, self.num_refines, 1)
    indices = current.nonzero(as_tuple=True)[0]
    (
        forecast_points,
        forecast_classes,
        forecast_features,
        trajectories,
        angles,
        orders,
    ) = ([], [], [], [], [], [])
    for interval in range(self.num_fu_frames):
        fused, _ = self.ego_cross_attn(
            ego_feature.new_ones(batch, 1, 3) * 0.5,
            ego_feature,
            points.detach(),
            features.detach(),
        )
        trajectories.append(self.traj_head(fused))
        future = stamps == interval + 1
        features = torch.cat((features, all_features[:, future]), 1)
        points = torch.cat((points, all_points[:, future]), 1).detach()
        indices = torch.cat((indices, future.nonzero(as_tuple=True)[0]))
        timestamp = torch.cat(
            (
                timestamp,
                points.new_full((batch, int(future.sum()), self.num_refines, 1), 0.5),
            ),
            1,
        )
        position = self.position_encoder(
            torch.cat((points, timestamp), -1).flatten(2, 3)
        )
        features = features + fused + position
        offsets = self.reg_branch(features).unflatten(-1, (-1, 3)) * 0.5
        classes = self.cls_branch(features).unflatten(-1, (-1, 17))
        velocity = self.vel_branch(features).unflatten(-1, (-1, 2))
        labels = classes.argmax(-1)
        if self.diagnostic_gate_mode == "native":
            gate = ((labels >= 2) & (labels <= 10)).to(offsets)
        elif self.diagnostic_gate_mode == "soft":
            gate = classes.softmax(-1)[..., 2:11].sum(-1)
        elif self.diagnostic_gate_mode == "open":
            gate = torch.ones_like(labels, dtype=offsets.dtype)
        else:
            raise ValueError(self.diagnostic_gate_mode)
        offsets = torch.cat(
            (offsets[..., :2] + velocity * gate[..., None], offsets[..., 2:]), -1
        )
        previous = _opus_decode_points(points, self.pc_range)
        if self.geometry_mode == "baseline":
            world = previous.mean(-2, keepdim=True) + offsets
            angle = offsets.new_zeros(batch, points.shape[1])
        else:
            old, old_angle = structured_update(
                previous[:, : self.num_query],
                template,
                offsets[:, : self.num_query],
                self.geometry_mode,
                self.shape_penalty,
                self.shape_radius_rms,
            )
            born = (
                previous[:, self.num_query :].mean(-2, keepdim=True)
                + offsets[:, self.num_query :]
            )
            world = torch.cat((old, born), 1)
            angle = torch.cat((old_angle, old_angle.new_zeros(batch, born.shape[1])), 1)
        points = _opus_encode_points(world, self.pc_range)
        forecast_points.append(points)
        forecast_classes.append(classes)
        forecast_features.append(features)
        angles.append(angle)
        orders.append(indices)
    outputs["pmg_angles"] = angles
    outputs["pmg_state_order"] = orders
    return forecast_points, forecast_classes, forecast_features, trajectories, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    root = Path("analysis/pmg_world_stage2")
    indices = np.linspace(0, 1023, args.samples, dtype=int).tolist()
    for arm in ("A0", "C1"):
        cfg = Config.fromfile(
            f"config/sparse_opus_gaussian_world/pmg_{arm}_4096_1024_direct15e.py"
        )
        runtime.configure_runtime(cfg)
        dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
        model = build_segmentor(cfg.model).cuda()
        saved = torch.load(
            f"out/pmg_world_stage1/{arm}/best.pth",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        epoch = runtime.load_evaluation_checkpoint(model, saved)
        del saved
        model.eval()
        global_counts = {
            v: np.zeros((7, 18, 18), np.int64) for v in ("native", "soft", "open")
        }
        regions = {
            v: np.zeros((3, len(REGIONS), 18, 18), np.int64) for v in global_counts
        }
        disagreement = []
        for index, batch in zip(
            indices, runtime.loader(dataset, cfg.val_loader, indices)
        ):
            with torch.no_grad():
                images, metas = runtime.move_batch(batch, "cuda")
                if index == indices[0]:
                    native = model(imgs=images, metas=metas)
                    baseline_grids = [
                        g.clone()
                        for g in [native["final_occ_grid"], *native["future_occ_grids"]]
                    ]
                    del native
                model._forecast = MethodType(diagnostic_forecast, model)
                token = token_at(dataset, index, 0)
                path = (
                    Path(cfg.pmg_experiment.reference_directory)
                    / "regions"
                    / (token + ".npz")
                )
                attributes = attributes_for(
                    dataset, index, cfg.pmg_experiment.attribute_root
                )
                for variant in global_counts:
                    model.diagnostic_gate_mode = variant
                    result = model(imgs=images, metas=metas)
                    grids = [result["final_occ_grid"], *result["future_occ_grids"]]
                    if variant == "native":
                        current = result["query_stamps"] == 0
                        initial = (
                            result["all_cls_scores"][-1][0, current]
                            .argmax(-1)
                            .cpu()
                            .numpy()
                        )
                        count = initial.shape[0]
                        for h in HORIZONS:
                            forecast = (
                                result["forecast_semantics_list"][h - 1][0, :count]
                                .argmax(-1)
                                .cpu()
                                .numpy()
                            )
                            rendered = initial if arm == "C1" else forecast
                            rendered_movable = (rendered >= 2) & (rendered <= 10)
                            gate_open = (forecast >= 2) & (forecast <= 10)
                            points = (
                                _opus_decode_points(
                                    result["forecast_points_list"][h - 1][0, :count],
                                    model.pc_range,
                                )
                                .cpu()
                                .numpy()
                            )
                            instances = attributes[HORIZONS.index(h)]
                            owner = owner_indices(
                                points.reshape(-1, 3),
                                instances,
                                _ego_pose(record_at(dataset, index, h)),
                            ).reshape(count, -1)
                            valid = (owner >= 0) & (
                                (points >= np.array([-40.0, -40.0, -1.0]))
                                & (points < np.array([40.0, 40.0, 5.4]))
                            ).all(-1)
                            fast = np.asarray(
                                [
                                    i.get("speed") is not None
                                    and np.isfinite(i["speed"])
                                    and i["speed"] >= 2
                                    for i in instances
                                ]
                            )
                            labels = np.asarray([i["label"] for i in instances])
                            associated_fast = (
                                valid & fast[np.maximum(owner, 0)]
                                if len(instances)
                                else np.zeros_like(valid)
                            )
                            correct = (
                                associated_fast
                                & (rendered == labels[np.maximum(owner, 0)])
                                if len(instances)
                                else np.zeros_like(valid)
                            )
                            disagreement.append(
                                dict(
                                    index=index,
                                    horizon=h,
                                    current_children=rendered.size,
                                    rendered_movable_children=int(
                                        rendered_movable.sum()
                                    ),
                                    rendered_movable_gate_closed=int(
                                        (rendered_movable & ~gate_open).sum()
                                    ),
                                    fast_associated=int(associated_fast.sum()),
                                    fast_render_correct=int(correct.sum()),
                                    fast_render_correct_gate_closed=int(
                                        (correct & ~gate_open).sum()
                                    ),
                                )
                            )
                    if index == indices[0] and variant == "native":
                        assert all(
                            torch.equal(a, b) for a, b in zip(grids, baseline_grids)
                        ), "gate instrumentation changed native output"
                        del baseline_grids
                    for h, grid in enumerate(grids):
                        target = (
                            (
                                metas["occ_label"]
                                if h == 0
                                else metas["future_occ_labels"][:, h - 1]
                            )
                            .cpu()
                            .numpy()
                            .reshape(-1)
                        )
                        prediction = grid.cpu().numpy().reshape(-1)
                        global_counts[variant][h] += histogram(prediction, target)
                        if h in HORIZONS:
                            masks = load_masks(path, h)
                            for j, name in enumerate(REGIONS):
                                regions[variant][HORIZONS.index(h), j] += histogram(
                                    prediction, target, masks[name]
                                )
                    del result, grids
                del images, metas, batch
            if (indices.index(index) + 1) % 8 == 0:
                print("GATES", arm, index, flush=True)
        np.savez_compressed(
            root / f"gates_{arm}_counts.npz",
            **{f"global_{k}": v for k, v in global_counts.items()},
            **{f"regions_{k}": v for k, v in regions.items()},
        )
        runtime.atomic_json(
            root / f"gates_{arm}.json",
            dict(
                status="completed",
                arm=arm,
                checkpoint_epoch=epoch,
                sample_indices=indices,
                variants=list(global_counts),
                native_instrumentation_exact=True,
                training_updates=0,
                semantic_motion_disagreement=disagreement,
                caveat="Soft/open gates change rollout geometry under fixed weights and can be out of distribution; no result establishes how a trained replacement performs.",
            ),
        )
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
