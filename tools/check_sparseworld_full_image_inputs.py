#!/usr/bin/env python3
"""Check lossless image I/O on both splits and future-window boundaries."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,cv2
from mmengine import Config
from tools.sparseworld_full_regions_common import ROOT,make_dataset
from tools.diagnose_teacher_alignment import seed_all
from tools.sparseworld_full_image_io import optimize_image_loaders,install_decoded_image_cache
from tools.sparseworld_region_utils import write_json,sha256


def compare(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:compare(a[key],b[key],path+'/'+key)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,torch.Tensor):assert a.dtype==b.dtype and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype,path;np.testing.assert_array_equal(a,b,err_msg=path)
    else:assert a==b,(path,a,b)


def main():
    cv2.setNumThreads(1);torch.set_num_threads(1)
    cfg=Config.fromfile('config/opus/future2current_feature_identity.py');datasets={}
    for split in ('val','train'):
        datasets[split]=make_dataset(cfg,split)[0];print('[image input check] dataset ready',split,flush=True)
    manifest=json.loads((ROOT/'manifest.json').read_text())['samples'];records=[]
    for split in ('val','train'):
        rows=[r for r in manifest if r['split']==split];records+=rows[:8]
        for count in (1,2):records.append(next(r for r in rows if sum(t['available'] for t in r['targets'])==count))
    seed_all(20261010);baseline=[];t=time.monotonic()
    for r in records:
        baseline.append(datasets[r['split']][r['index']]);print('[image input check] baseline',len(baseline),'/20',flush=True)
    a=time.monotonic()-t
    for ds in datasets.values():optimize_image_loaders(ds)
    install_decoded_image_cache();seed_all(20261010);durations=[]
    for i,(r,expected) in enumerate(zip(records,baseline)):
        t=time.monotonic();actual=datasets[r['split']][r['index']];durations.append(time.monotonic()-t)
        compare(expected,actual,r['token']);print('[image input check] exact parity',i+1,'/20',flush=True)
        del actual
    b=sum(durations)
    write_json(ROOT/'image_input_parity.json',dict(status='passed',sources=len(records),tokens=[r['token'] for r in records],
        baseline_seconds=a,cached_seconds=b,scope='Every returned field, array dtype and value bitwise identical; both splits, complete and padded future windows',
        mechanism='uint8 JPEG LRU copies; remove exact uint8-float32-uint8 roundtrip before original PIL resize; final FP32 tensor operations unchanged',
        code_sha256={p:sha256(p) for p in ['tools/sparseworld_full_image_io.py',__file__]}))
    print('All input fields exactly equal; baseline/cached seconds',a,b,flush=True)


if __name__=='__main__':main()
