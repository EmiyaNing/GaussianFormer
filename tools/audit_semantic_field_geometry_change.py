"""Supplementary fixed-input GS-slot geometry changes, separate from full-val metrics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.semantic_field_experiment_common import *
from copy import deepcopy
from tools.probe_gaussian_scale_supervision import layout

def main():
    from dataset.utils import custom_collate_fn_temporal
    from dataset import OPENOCC_DATASET
    from loss.semantic_gaussian_field_loss import SemanticGaussianFieldLoss
    indices=json.loads((RUN/'pilot.json').read_text())['training_indices']
    base={};matched={};rows=[];field_rows=[]
    modules={n:SemanticGaussianFieldLoss(mode=n,weight=.5).cuda() for n in ('geometry','semantic')}
    for arm in ('initial',*ARMS):
        checkpoint=INITIAL if arm=='initial' else RUN/'train'/arm/'epoch_1.pth'
        c,net=build('baseline',checkpoint)
        options=deepcopy(c.val_dataset_config);options.imageset=c.train_dataset_config.imageset;options.phase='val'
        dataset=OPENOCC_DATASET.build(options)
        for i in indices:
            row=dataset[i]
            distance,components=cached(np.asarray(row['occ_label']),np.asarray(row['occ_cam_mask']),RUN/'gt_cache')
            row.update(field_distance=distance,field_components=components,
                field_seed=np.array(c.seed*1000000+i,dtype=np.int64))
            batch=to_cuda(custom_collate_fn_temporal([row]))
            with torch.inference_mode():
                out=forward(net,batch);_,v=layout(net.head,out['gaussian']);means,scales=v[:2]
                a=dict(means=means.cpu().numpy(),scales=scales.cpu().numpy())
                inputs=loss_input(c,out,batch);inputs['global_iter']=512
                for mode,module in modules.items():
                    value,diag=module(inputs)
                    field_rows.append(dict(arm=arm,index=i,mode=mode,
                        diagnostics={k:float(v) for k,v in diag.items()}))
                    module.last_raw=None;module.last_geometry=None
                del inputs
            if arm=='initial':base[i]=a
            else:
                if arm=='baseline':matched[i]=a
                references={'initial':base[i]}
                if arm!='baseline':references['baseline']=matched[i]
                for refname,ref in references.items():
                    distance=np.linalg.norm(a['means']-ref['means'],axis=-1)/.4
                    ds=(a['scales']-ref['scales'])/.4
                    rows.append(dict(arm=arm,reference=refname,index=i,
                        center_displacement_voxels=dict(mean=float(distance.mean()),
                            median=float(np.median(distance)),p90=float(np.quantile(distance,.9))),
                        scale_axis_change_voxels=dict(mean_signed=float(ds.mean()),mean_abs=float(abs(ds).mean()),
                            nonzero_fraction=float((abs(ds)>1e-6).mean()))))
            del out,batch
        net.flush_memory_phase();del net;torch.cuda.empty_cache()
    write(RUN/'geometry_change_audit.json',dict(status='complete',indices=indices,rows=rows,paired_field_diagnostics=field_rows,
        interpretation='Same deterministic test transforms at four prespecified train frames. Slot displacement is not object tracking; this does not replace full-val coverage.',
        source_sha256=sha(Path(__file__))))

if __name__=='__main__':main()
