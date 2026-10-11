#!/usr/bin/env python3
"""Read-only source-encoder/rollout snapshots for pairs exported before tracing."""
import copy,json,sys,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from mmengine import Config
from torch.utils.data import DataLoader,Subset
from dataset import OPENOCC_DATASET
from dataset.utils import custom_collate_fn_temporal
from tools.sparseworld_full_regions_common import ROOT,make_dataset,WindowFeatureCache,encode_window
from tools.run_sparseworld_full_regions_fastio import collect_lineage,save_lineage,fast_zip_init
from tools.experiment_future_response_distillation import build_student
from tools.diagnose_teacher_alignment import tree_to
from model.response_distillation_forecast import forecast_with_position_gradient
from tools.sparseworld_region_utils import sha256,write_json

def main():
    cv2.setNumThreads(1);zipfile.ZipFile.__init__=fast_zip_init;torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py');manifest=json.loads((ROOT/'manifest.json').read_text())
    pending=[r for r in manifest['samples'] if (ROOT/'pairs'/(r['token']+'.npz')).exists() and not (ROOT/'query_lineage'/(r['token']+'.npz')).exists()]
    print('[lineage backfill] missing existing pairs',len(pending),flush=True)
    model=build_student(cfg,Config.fromfile(cfg.student_model_config));cache=WindowFeatureCache(model);done=0;started=time.monotonic()
    with torch.no_grad():
        for split in ('val','train'):
            local=[r for r in pending if r['split']==split]
            if not local:continue
            original,spec=make_dataset(cfg,split)
            spec.pipeline=[x for x in spec.pipeline if x['type']!='LoadMultiViewImageFuture']
            for x in spec.pipeline:
                if x['type']=='NuScenesAdaptor':x['num_cams']=30
            ds=OPENOCC_DATASET.build(spec);assert ds.keyframes==original.keyframes;del original
            loader=DataLoader(Subset(ds,[r['index'] for r in local]),batch_size=1,num_workers=2,
                collate_fn=custom_collate_fn_temporal,pin_memory=True)
            for rec,batch in zip(local,loader):
                assert batch['sample_idx'][0]==rec['token'];data=tree_to(batch,'cuda');imgs=data.pop('img')
                ego=model.plan_head(data['temporal_ego_states'].float().reshape(1,-1)).unsqueeze(1)
                state=encode_window(model,imgs,data,ego,cache.get(imgs,rec['frame_tokens'][:5]))
                out=forecast_with_position_gradient(model,state['representation'][-1],state['query_stamps'],ego,position_gradient=False)
                raw=collect_lineage(model,state['representation'][-1],state['query_stamps'],out);save_lineage(ROOT,rec,raw);done+=1
                if done%64==0 or done==len(pending):print(f'[lineage backfill] {done}/{len(pending)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                del data,imgs,state,out,raw,batch
    write_json(ROOT/'query_backfill_audit.json',dict(status='complete',sources=done,tokens=[r['token'] for r in pending],
        scope='Same five historical images, original frozen FP32 source encoder and rollout; no future image loading, no model update',code_sha256=sha256(__file__)))

if __name__=='__main__':main()
