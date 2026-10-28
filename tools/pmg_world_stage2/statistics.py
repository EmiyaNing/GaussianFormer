"""Paired scene bootstrap after the selected ten-epoch experiment finishes."""

import json
from pathlib import Path

import numpy as np

from .report import ROOT
from tools.sparse_opus_gausisan_world.runtime import atomic_json


def miou_batch(hist):
    diagonal = np.diagonal(hist, axis1=-2, axis2=-1)
    union = hist.sum(-1) + hist.sum(-2) - diagonal
    iou = (
        np.divide(
            diagonal, union, out=np.zeros_like(diagonal, dtype=float), where=union > 0
        )
        * 100
    )
    return iou[..., :17].mean(-1), iou[..., [0, 1, 11, 12, 13, 14, 15, 16]].mean(-1)


def ci(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def final_statistics():
    path = ROOT / "time_supervision_comparison.json"
    if not path.exists():
        return
    comparison = json.loads(path.read_text())
    conditions = {
        "A0_sparse": Path("analysis/pmg_world_stage1/A0"),
        "C1_sparse": Path("analysis/pmg_world_stage1/C1"),
        "A0_dense": ROOT / "T_A0_dense",
        "C1_dense": ROOT / "T_C1_dense",
    }
    weights = None
    reference_scenes = None
    scores = {}
    statics = {}
    regional = {}
    for name, folder in conditions.items():
        epoch = comparison["selected"][name]["epoch"]
        with np.load(folder / f"epoch_{epoch:02d}_counts.npz") as arrays:
            scenes = arrays["scenes"]
            if reference_scenes is None:
                reference_scenes = scenes
                generator = np.random.default_rng(20261020)
                weights = generator.multinomial(
                    len(scenes), np.full(len(scenes), 1 / len(scenes)), size=2000
                )
            assert np.array_equal(reference_scenes, scenes)
            scene = arrays["scene_confusion"]
            sampled = (weights @ scene.reshape(len(scenes), -1)).reshape(
                2000, 7, 18, 18
            )
            scores[name], statics[name] = miou_batch(sampled)
            regions = arrays["scene_region_confusion"][:, :, [0, 6, 5]]
            parts = np.stack(
                (
                    regions[..., :17, :].sum((-1, -2)),
                    regions[..., :17, :17].sum((-1, -2)),
                    regions[..., :, :17].sum((-1, -2)),
                    np.trace(regions[..., :17, :17], axis1=-2, axis2=-1),
                ),
                axis=-1,
            )
            summed = (weights @ parts.reshape(len(scenes), -1)).reshape(2000, 3, 3, 4)
            occupied, recalled, predicted, correct = np.moveaxis(summed, -1, 0)
            accuracy = (
                np.divide(
                    correct,
                    occupied,
                    out=np.zeros_like(correct, dtype=float),
                    where=occupied > 0,
                )
                * 100
            )
            union = occupied + predicted - recalled
            iou = (
                np.divide(
                    recalled,
                    union,
                    out=np.zeros_like(recalled, dtype=float),
                    where=union > 0,
                )
                * 100
            )
            regional[name] = dict(
                fast_accuracy=accuracy[:, :, 0],
                F_only=accuracy[:, :, 1],
                fast_neighborhood_iou=iou[:, :, 2],
            )
            del sampled
    results = {}
    for left, right in (
        ("A0_dense", "A0_sparse"),
        ("C1_dense", "C1_sparse"),
        ("C1_dense", "A0_dense"),
        ("C1_sparse", "A0_sparse"),
    ):
        delta = scores[left] - scores[right]
        results[left + "_minus_" + right] = dict(
            primary_95ci_pp=ci(delta[:, [0, 2, 4, 6]].mean(-1)),
            per_horizon_95ci_pp=[ci(delta[:, h]) for h in range(7)],
            static_123_95ci_pp=[
                ci((statics[left] - statics[right])[:, h]) for h in (2, 4, 6)
            ],
            regions={
                key: [
                    ci((regional[left][key] - regional[right][key])[:, h])
                    for h in range(3)
                ]
                for key in regional[left]
            },
        )
    did = (scores["C1_dense"] - scores["A0_dense"]) - (
        scores["C1_sparse"] - scores["A0_sparse"]
    )
    atomic_json(
        ROOT / "final_scene_statistics.json",
        dict(
            common_epoch_budget=comparison["common_epoch_budget"],
            scene_count=len(reference_scenes),
            iterations=2000,
            seed=20261020,
            comparisons=results,
            difference_in_differences_95ci_pp=[ci(did[:, h]) for h in range(7)],
            caveat="Conditional on checkpoint selection on this validation set; one training seed, repeated tuning and multiple comparisons are not corrected.",
        ),
    )
    lines = [
        "# Stage2 场景配对统计",
        "",
        "2000次场景配对bootstrap，95%区间，单位pp。以已经选中的checkpoint为条件，未校正选轮、多重比较和训练种子波动。",
        "",
        "| 对照 | 原主指标差值95%区间 |",
        "|---|---|",
    ]
    for name, row in results.items():
        lines.append(f"| {name} | {row['primary_95ci_pp']} |")
    lines += ["", "| 秒 | 时间监督对C1−A0差值的影响95%区间 |", "|---|---|"]
    for h in range(7):
        lines.append(f"| {h*.5:g} | {ci(did[:,h])} |")
    lines += [
        "",
        "逐时刻、静态类别与快速/F-only/邻域区间见[JSON](final_scene_statistics.json)。",
    ]
    (ROOT / "final_scene_statistics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    final_statistics()
