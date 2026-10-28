"""Audit reconstructed fast-region denominators against retained metadata."""

import json
from pathlib import Path

import numpy as np

from tools.sparse_opus_gausisan_world.runtime import atomic_json
from .regions import HORIZONS


def main():
    root = Path("analysis/pmg_world_stage1")
    manifest = json.loads(Path("data/sparseworld_opus_gaussian_bias_4096_1024/val.json").read_text())
    legacy = Path("out/sparseworld_region_comparison/20261010_full_nuscenes/attributes")
    compared, missing, differences = 0, [], []
    actual_total = np.zeros(3, dtype=np.int64)
    expected_total = np.zeros_like(actual_total)
    scene_counts = {}
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    for sample in manifest["samples"]:
        token = sample["token"]
        path = root / "reference" / "regions" / (token + ".npz")
        if not path.exists():
            missing.append(token)
            continue
        attributes = json.loads((legacy / (token + ".json")).read_text())["instances"]
        scene = sample["scene"]
        scene_counts.setdefault(scene, np.zeros((3, 3), dtype=np.int64))
        with np.load(path) as arrays:
            for i, horizon in enumerate(HORIZONS):
                actual = int(np.unpackbits(arrays[f"h{horizon}_fast"], count=640000).sum())
                expected = sum(x["voxels"] for x in attributes[i]
                               if x.get("speed") is not None and x["speed"] >= 2)
                actual_total[i] += actual
                expected_total[i] += expected
                fast = arrays[f"h{horizon}_fast"]
                both = arrays[f"h{horizon}_both_correct"]
                f_correct = fast & (arrays[f"h{horizon}_F_only"] | both)
                p_correct = fast & (arrays[f"h{horizon}_P_only"] | both)
                scene_counts[scene][i] += [actual, int(popcount[f_correct].sum()), int(popcount[p_correct].sum())]
                if actual != expected:
                    differences.append(dict(token=token, seconds=horizon*.5,
                                            retained_metadata_voxels=expected, reconstructed_voxels=actual))
        compared += 1
    intervals = []
    names = sorted(scene_counts)
    counts = np.stack([scene_counts[s] for s in names]) if names else np.zeros((0, 3, 3))
    if len(names):
        generator = np.random.default_rng(20261019)
        indices = generator.integers(len(names), size=(2000, len(names)))
        bootstrap = counts[indices].sum(1)
        for i, horizon in enumerate(HORIZONS):
            valid = bootstrap[:, i, 0] > 0
            changes = 100. * (bootstrap[valid, i, 2] - bootstrap[valid, i, 1]) / bootstrap[valid, i, 0]
            total = counts[:, i].sum(0)
            intervals.append(dict(seconds=horizon*.5, fast_gt_voxels=int(total[0]),
                                  F_accuracy=100*int(total[1])/max(int(total[0]),1),
                                  P_accuracy=100*int(total[2])/max(int(total[0]),1),
                                  P_minus_F_95ci=np.quantile(changes,[.025,.975]).tolist() if len(changes) else None))
    confirmed = bool(not missing and all(x["P_minus_F_95ci"] is not None and x["P_minus_F_95ci"][0] > 0 for x in intervals) and len(intervals)==3)
    atomic_json(root / "region_denominator_audit.json", dict(samples_compared=compared,
                 missing_samples=len(missing), full_coverage=not missing,
                 expected_fast_voxels=expected_total.tolist(), actual_fast_voxels=actual_total.tolist(),
                 mismatched_sample_horizons=differences,
                 scene_paired_fast_results=intervals, phenomenon_confirmed=confirmed,
                 note="Conservative exclusion of overlapping same-class GT boxes is explicit. Historical voxel-ID NPZ caches were previously deleted; any reconstruction differences are disclosed."))
    if not missing:
        np.savez_compressed(root / "reference_scene_fast_counts.npz", scenes=np.asarray(names), counts=counts)
        from .summarize import refresh
        refresh()
        if not confirmed:
            raise SystemExit("Gaussian fast-region effect did not meet the predefined scene-paired confirmation rule; review before long training")
    print("REGION_AUDIT", compared, "samples;", len(differences), "sample/horizon differences", flush=True)


if __name__ == "__main__":
    main()
