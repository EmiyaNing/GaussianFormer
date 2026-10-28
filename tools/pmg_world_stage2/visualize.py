"""Export the fixed-weight readout and fast-error evidence as report figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import ROOT
from tools.pmg_world.summarize import miou


def main():
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for ax, arm in zip(axes[:2], ("A0", "A4")):
        with np.load(ROOT / f"diagnosis_{arm}_counts.npz") as data:
            baseline = np.asarray([miou(h) for h in data["global_redecode"]])[1:]
            delta = (
                np.asarray(
                    [
                        [miou(h) for h in data[f"global_{v}"]][1:]
                        for v in (
                            "persist_semantics",
                            "persist_shape",
                            "persist_opacity",
                        )
                    ]
                )
                - baseline
            )
        ax.imshow(delta, cmap="RdBu", vmin=-5.5, vmax=5.5, aspect="auto")
        ax.set_xticks(range(6), ["0.5s", "1s", "1.5s", "2s", "2.5s", "3s"])
        ax.set_yticks(range(3), ["Semantics", "Scale / rotation", "Opacity"])
        ax.set_title(
            f"{arm}: attribute persistence minus re-decoding, fixed centers (mIoU pp)"
        )
        for row in range(3):
            for col in range(6):
                ax.text(
                    col,
                    row,
                    f"{delta[row,col]:+.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(delta[row, col]) > 3 else "black",
                )
    rows = json.loads((ROOT / "offline_diagnosis.json").read_text())["rows"]
    rows = [r for arm in ("A0", "C1") for r in rows if r["arm"] == arm]
    x = np.arange(len(rows))
    base = np.zeros(len(rows))
    for name, label, color in (
        ("correct_pct", "Correct", "#2a9d8f"),
        ("wrong_nonempty_pct", "Wrong nonempty class", "#e9c46a"),
        ("missed_pct", "Predicted empty", "#e76f51"),
    ):
        values = np.asarray([r[name] for r in rows])
        axes[2].bar(x, values, bottom=base, label=label, color=color)
        base += values
    axes[2].set_xticks(x, [f"{r['arm']} {r['seconds']}s" for r in rows])
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel("GT fast voxels (%)")
    axes[2].set_title(
        "Full 1024-sample fast-error partition at best-within-10 checkpoints"
    )
    axes[2].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.15))
    figure.savefig(ROOT / "diagnostic_overview.png", dpi=170)
    figure.savefig(ROOT / "diagnostic_overview.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
