"""Record every eligible sample in each official split, without subsampling."""

import argparse
import json
from pathlib import Path

from mmengine.config import Config

from dataset import OPENOCC_DATASET
from tools.sparse_opus_gausisan_world import runtime


def prepare(cfg):
    source_hashes, scenes, records = {}, {}, {}
    for split in ("train", "val"):
        options = dict(cfg[f"{split}_dataset_config"])
        output = Path(options.pop("selection_manifest"))
        options["type"] = "NuScenesSparseWorldStrictDataset"
        dataset = OPENOCC_DATASET.build(options)
        sources = {}
        for name in ("imageset", "trajectory_path", "admlp_state_path"):
            path = options[name]
            if path not in source_hashes:
                source_hashes[path] = runtime.digest(path)
            sources[path] = source_hashes[path]
        samples = [
            dict(token=dataset.scene_infos[scene][frame]["token"],
                 scene=str(scene), frame=int(frame))
            for scene, frame in dataset.keyframes
        ]
        if not samples or len({item["token"] for item in samples}) != len(samples):
            raise ValueError(f"invalid full-split sample list: {split}")
        scenes[split] = {item["scene"] for item in samples}
        payload = dict(split=split, scope="all_eligible_official_split_samples",
                       num_samples=len(samples), num_scenes=len(scenes[split]),
                       source_files=sources, samples=samples)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if json.loads(output.read_text()) != payload:
                raise ValueError(f"refusing to overwrite a changed manifest: {output}")
        else:
            runtime.atomic_json(output, payload)
        records[split] = dict(num_samples=len(samples), num_scenes=len(scenes[split]),
                              manifest=str(output), sha256=runtime.digest(output))
        del dataset
    if scenes["train"] & scenes["val"]:
        raise ValueError("training and validation scenes overlap")
    report = Path(cfg.report_directory)
    report.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(report / "data_preparation.json", records)
    print(records, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    prepare(Config.fromfile(parser.parse_args().config))


if __name__ == "__main__":
    main()
