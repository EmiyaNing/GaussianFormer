"""Fixed-sample raw Gaussian structure diagnosis on the mature D checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from tools.sparse_opus_gausisan_world import runtime
from .evaluate import shape_statistics
from model.encoder.opus_encoder import _opus_decode_points


def semantic_patch_support(points, target, pc_range):
    """Unique-GT-voxel support; semantic purity is not instance purity."""
    query_count = points.shape[0]
    index = torch.floor((points - pc_range[:3]) / .4).long()
    valid = ((index >= 0) & (index < index.new_tensor([200, 200, 16]))).all(-1)
    safe = torch.minimum(index.clamp(min=0), index.new_tensor([199, 199, 15]))
    flat = safe[..., 0]*3200 + safe[..., 1]*16 + safe[..., 2]
    labels = target.reshape(-1)[flat]
    valid &= (labels >= 0) & (labels < 17)
    queries = torch.arange(query_count, device=points.device)[:, None].expand_as(flat)
    keys = torch.unique((queries*640000 + flat)[valid])
    q, voxel = keys // 640000, keys % 640000
    classes = target.reshape(-1)[voxel].long()
    counts = torch.bincount(q*17 + classes, minlength=query_count*17).reshape(query_count, 17)
    support = counts.sum(-1)
    reliable = support >= 3
    purity = counts.max(-1).values[reliable] / support[reliable]
    return dict(query_count=query_count, queries_with_3_unique_occupied_voxels=int(reliable.sum()),
                mean_semantic_purity=float(purity.mean()) if purity.numel() else None,
                fraction_semantic_purity_below_0_6=float((purity < .6).float().mean()) if purity.numel() else None,
                note="GT semantic mixture only; equal classes can still belong to different moving instances")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sparse_opus_gaussian_world/pmg_A0_4096_1024_direct15e.py")
    parser.add_argument("--checkpoint", default="out/sparseworld_opus_gaussian_bias_balanced_D_full15e_4096_1024/D_balanced_conditional_ce/latest.pth")
    parser.add_argument("--output", default="analysis/pmg_world_stage1/reference_geometry.json")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    runtime.configure_runtime(cfg)
    model = build_segmentor(cfg.model).cuda()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    runtime.load_evaluation_checkpoint(model, saved)
    del saved
    model.eval()
    dataset = OPENOCC_DATASET.build(cfg.val_dataset_config)
    selected = np.linspace(0, len(dataset) - 1, 16, dtype=int).tolist()
    geometry, attributes, support = [], [], []
    captured = []
    original = model.gaussian_head.render_state

    def render(state, xyz):
        captured.append(state)
        return original(state, xyz)

    model.gaussian_head.render_state = render
    with torch.no_grad():
        for index, batch in zip(selected, runtime.loader(dataset, cfg.val_loader, selected)):
            images, metas = runtime.move_batch(batch, "cuda")
            captured.clear()
            result = model(imgs=images, metas=metas)
            geometry.extend(shape_statistics(model, result, index))
            current = result["query_stamps"] == 0
            initial_classes = result["all_cls_scores"][-1][:, current].argmax(-1)
            support.append(dict(index=index, **semantic_patch_support(
                _opus_decode_points(result["all_refine_pts"][-1][0, current], model.pc_range),
                metas["occ_label"][0], model.pc_range)))
            for horizon, state in enumerate(captured):
                count = initial_classes.shape[1]
                classes = state.semantics[:, :count].argmax(-1)
                attributes.append(dict(index=index, seconds=horizon * .5,
                                       scope="all_current_query_patches",
                                       mean_opacity=float(state.opacities.mean()),
                                       mean_scale=float(state.scales.mean()),
                                       semantic_flip_fraction=float((classes != initial_classes).float().mean())))
            del result, images, metas, batch
            print("GEOMETRY", index, flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    runtime.atomic_json(args.output, dict(status="completed", checkpoint=args.checkpoint,
                         samples=selected, geometry=geometry, attributes=attributes,
                         semantic_patch_support=support,
                         scope="16 fixed samples; raw patch structure, not GT pose-error decomposition",
                         permutation_invariant_metric="sorted pair-distance distribution",
                         training_updates=0))
    from .summarize import refresh
    refresh()


if __name__ == "__main__":
    main()
