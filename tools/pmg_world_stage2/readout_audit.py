"""Partition empty predictions using exact rendered foreground/empty scores."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from tools.pmg_world.regions import HORIZONS, load_masks, token_at
from tools.sparse_opus_gausisan_world import runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--arms", nargs="+", default=["A0", "A4", "C1"])
    args = parser.parse_args()
    root = Path("analysis/pmg_world_stage2")
    indices = np.linspace(0, 1023, args.samples, dtype=int).tolist()
    for arm in args.arms:
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
        records = []
        attributes = []
        original = model.gaussian_head.render_state
        counter = 0
        current_index = None
        targets = None
        masks = None

        def render(state, xyz):
            nonlocal counter
            logits = original(state, xyz)
            h = counter
            counter += 1
            old = state.semantics[:, : model.num_query]
            attributes.append(
                dict(
                    index=current_index,
                    horizon=h,
                    semantic_max_median=float(old.amax(-1).median()),
                    semantic_max_mean=float(old.amax(-1).mean()),
                    scale_mean=float(state.scales[:, : model.num_query].mean()),
                    opacity_mean=float(state.opacities[:, : model.num_query].mean()),
                )
            )
            if h in HORIZONS:
                mask = masks[h]
                target = targets[h][mask]
                selected = logits[0, :, mask]
                foreground, labels = selected[:17].max(0)
                empty = selected[17]
                missed = empty > foreground
                # Argmax ties select a foreground label before free=17.
                correct_fg = labels == target
                records.append(
                    dict(
                        index=current_index,
                        horizon=h,
                        fast_voxels=int(mask.sum()),
                        correct=int((~missed & correct_fg).sum()),
                        wrong_nonempty=int((~missed & ~correct_fg).sum()),
                        empty_foreground_correct=int((missed & correct_fg).sum()),
                        empty_foreground_wrong=int((missed & ~correct_fg).sum()),
                        foreground_score_sum=float(foreground.sum()),
                        empty_score_sum=float(empty.sum()),
                        correct_foreground_score_sum=float(
                            selected.gather(0, target[None]).sum()
                        ),
                    )
                )
            return logits

        model.gaussian_head.render_state = render
        with torch.no_grad():
            for current_index, batch in zip(
                indices, runtime.loader(dataset, cfg.val_loader, indices)
            ):
                images, metas = runtime.move_batch(batch, "cuda")
                path = (
                    Path(cfg.pmg_experiment.reference_directory)
                    / "regions"
                    / (token_at(dataset, current_index, 0) + ".npz")
                )
                targets = {
                    h: metas["future_occ_labels"][0, h - 1].reshape(-1).long()
                    for h in HORIZONS
                }
                masks = {
                    h: torch.from_numpy(load_masks(path, h)["fast"]).to("cuda")
                    for h in HORIZONS
                }
                counter = 0
                result = model(imgs=images, metas=metas)
                assert counter == 7
                del result, images, metas, batch
                if (indices.index(current_index) + 1) % 16 == 0:
                    print("READOUT_AUDIT", arm, current_index, flush=True)
        summary = {}
        for h in HORIZONS:
            selected = [r for r in records if r["horizon"] == h]
            summary[str(h * 0.5)] = {
                k: sum(r[k] for r in selected)
                for k in selected[0]
                if k not in ("index", "horizon")
            }
        runtime.atomic_json(
            root / f"readout_audit_{arm}.json",
            dict(
                status="completed",
                arm=arm,
                checkpoint_epoch=epoch,
                sample_indices=indices,
                records=records,
                summary=summary,
                attributes=attributes,
                training_updates=0,
                note="Exact renderer outputs. Correct foreground class with empty winning is a score competition outcome, not proof that changing a threshold is a suitable trained solution.",
            ),
        )
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
