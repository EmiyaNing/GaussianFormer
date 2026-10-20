"""Exploratory positive/negative channel accounting after registered path probes."""
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage1 import (RUN,EPOCHS,RADII,initialize,build_network,build_loader,seed_all,
    Frame,network_output,to_cuda,write_json,file_sha256,read)


def main():
    p=initialize();dest=RUN/'amplitude_trace';dest.mkdir(exist_ok=True)
    protocol=dest/'protocol.json'
    if not protocol.exists():write_json(protocol,dict(created_at=datetime.now().astimezone().isoformat(),
        exploratory=True,reason='Trace the already observed amplitude pathway into assigned-positive and 16 negative channels.',
        scenes=p['scene_samples'],radii=RADII,primary_radius=.001,source_sha256=file_sha256(__file__),
        primary_protocol_sha256=file_sha256(RUN/'protocol.json'),no_new_selection=True))
    files=[]
    for epoch in EPOCHS:
        seed_all(928);network,mods=build_network(epoch)
        for split in ['train','val']:
            selected=p['scene_samples'][split];_,loader,_=build_loader(split,[x['index'] for x in selected])
            for n,(meta,batch) in enumerate(zip(selected,loader)):
                path=dest/f'epoch{epoch}_{split}_{meta["index"]:04d}.json'
                if path.exists():files.append(str(path.relative_to(ROOT)));continue
                batch=to_cuda(batch)
                with torch.no_grad():out=network_output(network,batch)
                f=Frame(network,out,batch,mods)
                raw=f.u.clone().requires_grad_(True);gradient=torch.autograd.grad(f.set_objective(raw),raw)[0].detach()
                scale=float(raw.detach()[0,f.point_valid].double().norm());norm=float(gradient.double().norm())
                rows=[]
                with torch.no_grad():
                    positive=torch.zeros_like(f.sem);positive[0,f.point_valid]=F.one_hot(f.assigned,17).to(positive)
                    for radius in RADII:
                        updated=F.softplus(f.u-radius*scale/max(norm,1e-30)*gradient)
                        delta=updated-f.sem;pos=(delta*positive).sum(-1);neg=(delta*(1-positive)).sum(-1)
                        payload=torch.zeros_like(f.args[2]);payload[:,:-1,:3]=torch.stack([pos,neg,pos+neg],-1)
                        moment=f.aggregate(payload)[:3];error=float((moment[0]+moment[1]-moment[2]).abs().max())
                        assert error<2e-4
                        cohorts={}
                        for name,mask in f.cohorts.items():
                            valid=mask&(f.support>1e-12);count=int(valid.sum())
                            parts=moment[:,valid]/f.support[valid][None]
                            cohorts[name]=dict(supported=count,positive_sum=float(parts[0].double().sum()),
                                negative_sum=float(parts[1].double().sum()),total_sum=float(parts[2].double().sum()),
                                amplitude_declined=int((parts[2]<0).sum()))
                        valid=f.point_valid
                        rows.append(dict(radius=radius,cohorts=cohorts,rendered_additivity_max_error=error,
                            gaussian=dict(count=int(valid.sum()),before_amplitude_sum=float(f.sem[0,valid].double().sum()),
                                positive_sum=float(pos[0,valid].double().sum()),negative_sum=float(neg[0,valid].double().sum()),
                                total_sum=float((pos+neg)[0,valid].double().sum()),
                                amplitude_declined=int(((pos+neg)[0,valid]<0).sum()),
                                positive_increased=int((pos[0,valid]>0).sum()),negative_decreased=int((neg[0,valid]<0).sum()))))
                    from mmengine import Config
                    cfg=Config.fromfile('config/center_support_contribution1/a0_original.py')
                    assert cfg.loss_input_convertion['occ_mask']=='occ_cam_mask'
                    native_inputs={k:out[v] for k,v in cfg.loss_input_convertion.items() if k in ['pred_occ','sampled_xyz','sampled_label','occ_mask']}
                    ref=mods[1](native_inputs)+mods[2](native_inputs);ce,lovasz=f.field_objectives(f.base)
                    loss_error=float((ref-ce-lovasz).abs());assert loss_error<3e-6
                write_json(path,dict(status='complete',epoch=epoch,split=split,frame=meta,rows=rows,
                    original_training_field_loss_error=loss_error,source_sha256=file_sha256(__file__)))
                files.append(str(path.relative_to(ROOT)));del f,out,batch,raw,gradient
            print('channel trace',epoch,split,len(selected),'complete',flush=True)
        del network,mods
    assert len(files)==144
    write_json(dest/'manifest.json',dict(status='complete',files=files,source_sha256=file_sha256(__file__)))

if __name__=='__main__':main()
