"""Two-update output replay on fixed implementation-audit frames.

These cases check actual AdamW updates and factor replacement. They are not
sampled substitutes for the separate full train/val statistics.
"""
from __future__ import annotations
import copy
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.gaussian_geometry_causal import RUN,ARMS,build,forward,to_cuda,seed_all,write,sha


def render(head,g,xyz):
    z=head._render_batched(xyz,*head.prepare_gaussian_args(g))
    return torch.cat((z[:,:17],z[:,17:]-head.empty_bias.detach()),1)


def measures(z,batch):
    y=batch['occ_label'].flatten().long();cam=batch['occ_cam_mask'].flatten().bool()
    valid=cam&(y>=0)&(y<18);z=z[0,:,valid];y=y[valid];pred=z.argmax(0)
    fg=y<17;empty=y==17;correct=(pred==y)&fg
    return dict(camera_valid=int(valid.sum()),GTfg=int(fg.sum()),GTempty=int(empty.sum()),
        FN=int((fg&(pred==17)).sum()),FP=int((empty&(pred<17)).sum()),
        correct_fg=int(correct.sum()),mean_fg_max=float(z[:17,fg].max(0).values.mean()),
        mean_fg_target=float(z[:,fg].gather(0,y[fg][None]).mean()))


def run():
    seed_all(928);torch.set_num_threads(4)
    common=RUN/'common/standard_s928/final.pth'
    assert json.loads((RUN/'pilot/pairing_audit.json').read_text())['status']=='passed'
    from dataset import OPENOCC_DATASET
    from dataset.utils import custom_collate_fn_temporal
    c,net,opt,loss,state=build(common,freeze_empty=True);net.eval()
    del opt,loss,state
    batches={};before={};records=[]
    for split in ('train','val'):
        setting=copy.deepcopy(c.val_dataset_config);setting.imageset=c[split+'_dataset_config'].imageset
        dataset=OPENOCC_DATASET.build(setting)
        batch=to_cuda(custom_collate_fn_temporal([dataset[0]]));batches[split]=batch
        with torch.inference_mode():
            out=forward(net,batch)
            before[split]=(out['gaussian'],out['sampled_xyz'],out['pred_occ'][-1])
    for arm in ARMS[:4]:
        checkpoint=RUN/'pilot'/f'{arm}_s929/final.pth'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        net.load_state_dict(saved['state_dict'],strict=True);del saved
        for split,batch in batches.items():
            with torch.inference_mode():
                out=forward(net,batch);new=out['gaussian'];old,xyz,baseline=before[split]
                A0=old.semantics.sum(-1,keepdim=True);A1=new.semantics.sum(-1,keepdim=True)
                p0=old.semantics/A0;p1=new.semantics/A1
                geometrical=old._replace(means=new.means,scales=new.scales,rotations=new.rotations)
                amplitude=old._replace(opacities=new.opacities,semantics=old.semantics*(A1/A0))
                direction=old._replace(semantics=p1*A0)
                exact=render(net.head,new,xyz)
                torch.testing.assert_close(exact,out['pred_occ'][-1],rtol=1e-5,atol=2e-5)
                variants=dict(before=baseline,after=out['pred_occ'][-1],
                    geometry_only=render(net.head,geometrical,xyz),
                    amplitude_only=render(net.head,amplitude,xyz),
                    direction_only=render(net.head,direction,xyz))
                records.append(dict(arm=arm,split=split,index=0,actual_updates=2,
                    mean_center_displacement_m=float((new.means-old.means).norm(dim=-1).mean()),
                    mean_scale_abs_change_m=float((new.scales-old.scales).abs().mean()),
                    mean_A_before=float(A0.mean()),mean_A_after=float(A1.mean()),
                    mean_alpha_before=float(old.opacities.mean()),mean_alpha_after=float(new.opacities.mean()),
                    mean_b_before=float((old.opacities*A0).mean()),mean_b_after=float((new.opacities*A1).mean()),
                    mean_class_total_variation=float(.5*(p1-p0).abs().sum(-1).mean()),
                    after_reconstruction_max_error=float((exact-out['pred_occ'][-1]).abs().max()),
                    outcomes={k:measures(v,batch) for k,v in variants.items()}))
    write(RUN/'actual_update_replay.json',dict(status='passed',records=records,
        use='Two fixed implementation-audit frames only; exclude from full-split statistical estimates',
        source_sha256=sha(__file__)))


if __name__=='__main__':
    write(RUN/'replay_protocol.json',dict(source_sha256=sha(__file__),
        probes=[dict(split='train',index=0),dict(split='val',index=0)],
        purpose='Verify actual optimizer output changes and geometry/amplitude/direction replay; not population statistics'))
    while not (RUN/'pilot/pairing_audit.json').exists():time.sleep(5)
    run()
