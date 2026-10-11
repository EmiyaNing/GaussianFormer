"""Fixed-target feature audits for end-to-end protocol experiments."""
import json
from pathlib import Path
import torch
from tools.future_distillation_causal_common import matched_targets, feature_loss, teacher_record
from tools.diagnose_teacher_alignment import tree_to
from tools.train_future_predictable_full import image_loader, atomic_torch_save
from tools.experiment_future_predictable_information import save_json


@torch.no_grad()
def audit(cfg,strict,root,manifest,dataset,build):
    root=Path(root);saved=root/'alignment_audit.json'
    result=json.loads(saved.read_text()) if saved.exists() else dict(status='running',cases={},
        protocol='Fixed teacher targets / masks at original baseline query geometry and same query slots; trained positions can move. Also report moving matches. Not identical physical sampling after training.')
    report=json.loads((root/'results.json').read_text())
    checkpoints=[('baseline',None)]
    checkpoints.extend((label,root/label/'latest.pth') for label,row in report['arms'].items() if row.get('status')=='complete')
    for label,checkpoint in checkpoints:
        if label in result['cases']:continue
        instance=build(cfg,strict)
        if checkpoint:
            weights=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            instance.load_state_dict(weights['state_dict'],strict=True)
        instance.eval().requires_grad_(False)
        rows=[]
        for position,data in enumerate(image_loader(dataset,0)):
            data=tree_to(data,'cuda');output=instance(imgs=data.pop('img'),metas=data)
            index=manifest['val']['indices'][position];scene=manifest['val']['scenes'][position]
            teacher=teacher_record(cfg,'val',index,scene)
            moving=matched_targets(instance,output,teacher)
            path=root/'fixed_alignment_targets'/f'{index:06d}.pth'
            if label=='baseline':
                path.parent.mkdir(parents=True,exist_ok=True)
                record=dict(index=index,scene=scene,checkpoint_sha256=cfg.student_checkpoint_sha256,
                            targets={h:(features.cpu(),valid.cpu()) for h,(features,valid) in moving.items()})
                atomic_torch_save(record,path)
            record=torch.load(path,map_location='cpu',weights_only=False)
            if (record['index'],record['scene'],record['checkpoint_sha256'])!=(index,scene,cfg.student_checkpoint_sha256):raise RuntimeError('Fixed target identity changed')
            fixed={h:(features.cuda(),valid.cuda()) for h,(features,valid) in record['targets'].items()}
            rows.append(dict(index=index,scene=scene,fixed=float(feature_loss(output,fixed)),moving=float(feature_loss(output,moving)),
                             valid_fixed=sum(int(v.sum()) for _,v in fixed.values()),valid_moving=sum(int(v.sum()) for _,v in moving.values())))
        result['cases'][label]=dict(fixed_feature_loss=sum(row['fixed'] for row in rows)/len(rows),
                                   moving_feature_loss=sum(row['moving'] for row in rows)/len(rows),rows=rows)
        save_json(saved,result)
        print('[protocol fixed-target audit]',label,result['cases'][label]['fixed_feature_loss'],flush=True)
        del instance;torch.cuda.empty_cache()
    result['status']='complete';save_json(saved,result)
    return result
