"""Reconstruct mutually exclusive fast-target errors from frozen stage1 counts."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("analysis/pmg_world_stage2")
SELECTED = {"A0": 9, "A1": 9, "A3": 9, "A4": 10, "C1": 9}


def error_partition(hist):
    occupied = int(hist[:17].sum())
    correct = int(np.trace(hist[:17, :17]))
    missed = int(hist[:17, 17].sum())
    wrong = occupied - correct - missed
    return dict(
        voxels=occupied,
        correct=correct,
        missed=missed,
        wrong_nonempty=wrong,
        correct_pct=100 * correct / occupied,
        missed_pct=100 * missed / occupied,
        wrong_nonempty_pct=100 * wrong / occupied,
        missed_share_of_errors_pct=100 * missed / (occupied - correct),
    )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for arm, epoch in SELECTED.items():
        path = Path(f"analysis/pmg_world_stage1/{arm}/epoch_{epoch:02d}_counts.npz")
        with np.load(path) as data:
            counts = data["region_confusion"]
            for slot, seconds in enumerate((1, 2, 3)):
                rows.append(
                    dict(
                        arm=arm,
                        epoch=epoch,
                        seconds=seconds,
                        **error_partition(counts[slot, 0]),
                    )
                )
        sources.append(str(path))
    with (ROOT / "fast_error_partition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "offline_diagnosis.json").write_text(
        json.dumps(
            dict(
                status="completed",
                validation_samples=1024,
                selection="one checkpoint per arm; best original metric within epochs 1..10",
                rows=rows,
                sources=sources,
                training_updates=0,
                limitation="Predicted empty does not identify whether geometry, opacity, semantic amplitude or empty competition caused the miss.",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Stage2：快速目标错误分解",
        "",
        "固定使用第一阶段前10轮选中checkpoint的完整1024样本计数；不重新加载当前已覆盖的best.pth。",
        "",
        "| 组别 | 轮次 | 秒 | 正确类别% | 错误非空类% | 预测为空% | 错误中漏检占比% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['epoch']} | {r['seconds']} | {r['correct_pct']:.3f} | {r['wrong_nonempty_pct']:.3f} | {r['missed_pct']:.3f} | {r['missed_share_of_errors_pct']:.3f} |"
        )
    lines += [
        "",
        "A0三个时刻约85%的快速错误表现为预测为空。C1的快速召回更低；2/3秒错误非空类别反而减少，仍不足以抵消漏检增加。",
        "",
        "这支持先检查覆盖和占据贡献，但不能直接证明运动误差或空类压制是根因。需要同权重属性替换与Gaussian位置诊断。",
        "",
        "这三个结果互斥且总和为100%；它们只针对GT快速占据，区域外误报必须另看邻域IoU/FP。",
        "",
        "原始计数与CSV：[JSON](offline_diagnosis.json)、[CSV](fast_error_partition.csv)。",
    ]
    (ROOT / "offline_diagnosis.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(status="completed", report=str(ROOT / "offline_diagnosis.md")),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
