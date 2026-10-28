"""Standalone figures for the completed query-gate experiment."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.pmg_world_native_adapt.gate_report import ROOT, SOURCES, regional


def main():
    metrics = {name: json.loads((folder / f"{prefix}_metrics.json").read_text())
               for name, folder, prefix in SOURCES}
    comparisons = json.loads((ROOT / "comparison_statistics.json").read_text())["comparisons"]
    gates = json.loads((ROOT / "gate_statistics.json").read_text())["rows"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    colors = {"T-A0": "#555555", "N-native": "#e58b31", "N-half": "#3985ad", "G1": "#af3a63"}
    time = np.arange(7) * .5
    horizons = np.arange(1, 4)
    for name, color in colors.items():
        metric = metrics[name]
        axes[0, 0].plot(time, [h["miou"] for h in metric["horizons"]], "o-", label=name, color=color)
        axes[0, 1].plot(horizons, regional(metric, "fast", "occupied_semantic_accuracy"), "o-", color=color)
        axes[0, 2].plot(horizons, regional(metric, "fast", "occupied_recall"), "o-", color=color)
    axes[0, 0].set(title="Global mIoU", ylabel="mIoU (%)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Fast-target semantic accuracy", ylabel="Accuracy (%)", xticks=horizons)
    axes[0, 2].set(title="Fast-target occupancy recall", ylabel="Recall (%)", xticks=horizons)
    for axis, region, ci_name, title in (
        (axes[1, 0], "F_only", "F_only", "F-only accuracy change vs T-A0"),
        (axes[1, 1], "static_classes", "static_classes_accuracy", "Static-class GT accuracy change vs T-A0"),
    ):
        base = regional(metrics["T-A0"], region, "occupied_semantic_accuracy")
        for name in ("N-native", "N-half", "G1"):
            delta = regional(metrics[name], region, "occupied_semantic_accuracy") - base
            if name == "G1":
                ci = np.asarray(comparisons["G1_minus_T-A0"][ci_name]["per_horizon_delta_95ci_pp"])
                axis.errorbar(horizons, delta, yerr=np.stack((delta-ci[:, 0], ci[:, 1]-delta)),
                              fmt="o-", capsize=3, color=colors[name], label=name)
            else:
                axis.plot(horizons, delta, "o--", color=colors[name], label=name)
        axis.axhline(0, color="#777777", linewidth=.8)
        axis.set(title=title, ylabel="Change (percentage points)", xticks=horizons)
    gt = np.asarray([r["seconds"] for r in gates])
    mean = np.asarray([r["mean"] for r in gates])
    std = np.asarray([r["std"] for r in gates])
    axes[1, 2].plot(gt, mean, "o-", color=colors["G1"])
    axes[1, 2].fill_between(gt, mean-std, mean+std, color=colors["G1"], alpha=.2)
    axes[1, 2].axhline(1, color="#777777", linewidth=.8)
    axes[1, 2].set(title="Query gate: mean and one SD", ylabel="Gate coefficient")
    for axis in axes.flat:
        axis.set_xlabel("Forecast horizon (s)")
        axis.grid(alpha=.15)
    fig.suptitle("G1 minimal recurrent gate: matched 512-update development experiment", fontsize=14)
    fig.savefig(ROOT / "comparison_overview.png", dpi=180)
    fig.savefig(ROOT / "comparison_overview.pdf")
    plt.close(fig)
    print(str(ROOT / "comparison_overview.png"))


if __name__ == "__main__":
    main()
