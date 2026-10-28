"""Independently reload and evaluate an arm's selected checkpoint."""

from pathlib import Path

import numpy as np
import torch
from mmseg.models import build_segmentor

from tools.sparse_opus_gausisan_world import runtime
from tools.pmg_world.evaluate import evaluate


def verify_best(cfg, work_directory, dataset):
    work_directory = Path(work_directory)
    output = Path(cfg.report_directory) / "best_checkpoint_verification"
    completed = output / "verification.json"
    checkpoint_path = work_directory / "best.pth"
    checkpoint_sha = runtime.digest(checkpoint_path)
    if completed.exists():
        import json

        previous = json.loads(completed.read_text())
        if previous.get("checkpoint_sha256") == checkpoint_sha and previous.get(
            "passed"
        ):
            return previous
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_segmentor(cfg.model).cuda()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.set_epoch(saved["epoch_index"])
    model.pts_bbox_head.ind_stamps_all = saved["query_stamps"].cuda()
    model.pts_bbox_head.reset_mask()
    epoch = saved["epoch_index"] + 1
    del saved
    actual = evaluate(model, dataset, cfg, output, epoch)
    original_path = Path(cfg.report_directory) / f"epoch_{epoch:02d}_counts.npz"
    with np.load(original_path) as original, np.load(
        output / f"epoch_{epoch:02d}_counts.npz"
    ) as repeated:
        differences = {}
        for key in ("confusion", "region_confusion"):
            delta = repeated[key] - original[key]
            differences[key] = dict(
                exact=bool(np.array_equal(repeated[key], original[key])),
                absolute_sum=int(np.abs(delta).sum()),
                maximum=int(np.abs(delta).max()),
            )
        base = original["confusion"]
        candidate = repeated["confusion"]

        def class_iou(counts):
            diagonal = counts.diagonal(axis1=-2, axis2=-1).astype(float)
            union = counts.sum(-1) + counts.sum(-2) - diagonal
            return (
                np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union > 0)
                * 100
            )

        maximum_class_iou_delta = float(
            np.abs(class_iou(base) - class_iou(candidate)).max()
        )
    exact = all(x["exact"] for x in differences.values())
    # Small CUDA accumulation differences are disclosed, not silently described
    # as exact parity. Larger count or class-metric changes invalidate the check.
    passed = exact or (
        differences["confusion"]["absolute_sum"] <= 64
        and differences["region_confusion"]["absolute_sum"] <= 640
        and maximum_class_iou_delta < 0.001
    )
    report = dict(
        passed=passed,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        epoch=epoch,
        samples=actual["num_samples"],
        exact_confusion=exact,
        differences=differences,
        maximum_class_iou_delta_pp=maximum_class_iou_delta,
    )
    runtime.atomic_json(completed, report)
    if not passed:
        raise ValueError("independent best-checkpoint verification differs materially")
    del model
    return report
