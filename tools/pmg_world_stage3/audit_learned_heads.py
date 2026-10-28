"""Four fixed-sample input-scale, correction-size and zero-head parity audit."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmseg.models import build_segmentor

from dataset import OPENOCC_DATASET
from tools.pmg_world.regions import histogram
from tools.sparse_opus_gausisan_world import runtime


def stats(value):
    flat=value.detach().float().reshape(-1)
    if not flat.numel():
        return None
    return dict(mean=float(flat.mean()),p50=float(flat.median()),p95=float(torch.quantile(flat,.95)),maximum=float(flat.max()))


def main():
    indices=[0,341,682,1023]
    results={}
    for arm in ("current","history"):
        folder=Path(f"out/pmg_world_stage3_20261023/frozen_M_{arm}")
        if not (folder/"completed.json").exists():
            raise ValueError("both arms must finish before the isolated audit")
        cfg=Config.fromfile(f"config/sparse_opus_gaussian_world/pmg_stage3_frozen_M_{arm}_512updates.py")
        runtime.configure_runtime(cfg)
        dataset=OPENOCC_DATASET.build(cfg.val_dataset_config)
        model=build_segmentor(cfg.model).cuda()
        saved=torch.load(folder/"latest.pth",map_location="cpu",mmap=True,weights_only=False)
        runtime.load_evaluation_checkpoint(model,saved)
        del saved
        model.eval().requires_grad_(False)
        render_states=[]
        original_render=model.gaussian_head.render_state

        def capture_state(state,xyz):
            render_states.append(state)
            return original_render(state,xyz)

        model.gaussian_head.render_state=capture_state
        records=[]
        corrections=[]
        current_index=None

        def capture(module,arguments,output):
            value=arguments[0]
            raw=output.unflatten(-1,(model.num_refines,4))
            xy=model.correction_radius_m*torch.tanh(raw[...,:2]/model.correction_radius_m)
            xy=xy/(xy.norm(dim=-1,keepdim=True)/model.correction_radius_m).clamp_min(1.)
            xy=xy*raw[...,3:].sigmoid()
            corrections.append((xy.detach(),raw[...,3:].sigmoid().detach()))
            first=module[0].weight
            records.append(dict(index=current_index,seconds=float(value[0,0,-1]),
                future_feature_norm=stats(value[...,:256].norm(dim=-1)),
                evidence_feature_norm=stats(value[...,256:512].norm(dim=-1)),
                evidence_zero_fraction=float((value[...,256:512].norm(dim=-1)<1e-8).float().mean()),
                future_preactivation_norm=stats(F.linear(value[...,:256],first[:,:256]).norm(dim=-1)),
                evidence_preactivation_norm=stats(F.linear(value[...,256:512],first[:,256:512]).norm(dim=-1)),
                correction_l2_m=stats(xy.norm(dim=-1))))

        handle=model.motion_correction.register_forward_hook(capture)
        parity=[]
        with torch.no_grad():
            for current_index,data in zip(indices,runtime.loader(dataset,cfg.val_loader,indices)):
                images,metas=runtime.move_batch(data,"cuda")
                model.memory.clear();model.memory_queue.clear()
                start=len(records)
                corrections.clear()
                render_states.clear()
                actual=model(imgs=images,metas=metas)
                actual_states=list(render_states)
                render_states.clear()
                assert len(corrections)==6
                for h,(xy,gate) in enumerate(corrections):
                    delta=torch.cat((xy,torch.zeros_like(xy[...,:1])),-1)
                    source=actual['forecast_points_list'][h][:,:model.num_query]-delta/(model.pc_range[3:]-model.pc_range[:3])
                    valid=torch.isfinite(source).all(-1)&((source>=0)&(source<1)).all(-1)
                    corrected=actual['forecast_points_list'][h][:,:model.num_query]
                    corrected_valid=torch.isfinite(corrected).all(-1)&((corrected>=0)&(corrected<1)).all(-1)
                    label=actual['forecast_semantics_list'][h][:,:model.num_query].argmax(-1)
                    moving=(label>=2)&(label<=10)
                    size=xy.norm(dim=-1)
                    records[start+h]['active_correction']=dict(
                        valid_fraction=float(valid.float().mean()),
                        source_valid_to_invalid=int((valid&~corrected_valid).sum()),
                        source_invalid_to_valid=int((~valid&corrected_valid).sum()),
                        fraction_gt_1cm=float((size[valid]>.01).float().mean()),
                        predicted_static=stats(size[valid&~moving]),
                        predicted_movable=stats(size[valid&moving]),
                        gate=stats(gate.squeeze(-1)[valid]))
                current_actual=actual["final_occ_grid"].clone()
                del actual
                correction=model.correction
                model.correction=lambda features:features.new_zeros((*features.shape[:2],model.num_refines,3))
                native=model(imgs=images,metas=metas)
                model.correction=correction
                grids=[native["final_occ_grid"],*native["future_occ_grids"]]
                targets=[metas["occ_label"],*metas["future_occ_labels"].unbind(1)]
                counts=np.stack([histogram(p.cpu().numpy(),t.cpu().numpy()) for p,t in zip(grids,targets)])
                with np.load(Path("analysis/pmg_world_stage3_20261023/diagnosis_1024/samples")/f"{current_index:04d}.npz") as baseline:
                    exact=bool(np.array_equal(counts,baseline["global_confusion"][0]))
                content_exact=all(torch.equal(getattr(a,field),getattr(b,field)) for a,b in zip(actual_states,render_states)
                                  for field in ('semantics','scales','rotations'))
                births_exact=all(torch.equal(getattr(a,field)[:,model.num_query:],getattr(b,field)[:,model.num_query:])
                                 for a,b in zip(actual_states[1:],render_states[1:])
                                 for field in ('means','semantics','scales','rotations','opacities'))
                parity.append(dict(index=current_index,zero_head_all_horizon_counts_exact=exact,
                    gaussian_semantics_and_shape_exact=content_exact,birth_gaussians_exact=births_exact,
                    current_grid_unchanged=bool(torch.equal(current_actual,native["final_occ_grid"]))))
                assert content_exact and births_exact
                del native,grids,targets,images,metas,data,current_actual
                model.memory.clear();model.memory_queue.clear()
                render_states.clear();actual_states.clear()
        handle.remove()
        assert all(row["zero_head_all_horizon_counts_exact"] and row["current_grid_unchanged"] for row in parity)
        results[arm]=dict(records=records,parity=parity,
            scope="four fixed validation samples, all inherited queries; not GT-instance motion statistics")
        del model,dataset
        gc.collect();torch.cuda.empty_cache()
    runtime.atomic_json(Path("analysis/pmg_world_stage3_20261023/learned_head_audit.json"),results)
    print("LEARNED_HEAD_AUDIT_PASSED",flush=True)


if __name__=="__main__":
    main()
