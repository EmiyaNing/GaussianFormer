#!/usr/bin/env python3
"""Identical full-census exporter with bounded OpenCV threads and fast ZIP I/O."""
import argparse,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tools.sparseworld_region_utils import sha256,write_json
from tools.sparseworld_full_regions_common import ROOT

original_zip_init=zipfile.ZipFile.__init__
RAW={}


def collect_lineage(instance,encoded,stamps,out):
    current=stamps==0
    p=instance._world(encoded['query_points'][:,current])[0]
    z=encoded['opus_logits'][:,current][0]
    result=dict(source_points=p.detach().cpu().numpy(),source_labels=z.argmax(-1).byte().cpu().numpy(),
                source_max_logits=z.amax(-1).float().cpu().numpy())
    assert p.shape==(720,48,3)
    for j,h in enumerate((2,4,6)):
        points=out['points'][h-1][0,:720*48].reshape(720,48,3)
        logits=out['logits'][h-1][0,:720*48].reshape(720,48,17)
        result['future_points_'+str(j)]=points.detach().cpu().numpy()
        result['future_labels_'+str(j)]=logits.argmax(-1).byte().cpu().numpy()
        result['future_max_logits_'+str(j)]=logits.amax(-1).float().cpu().numpy()
    return result


def save_lineage(root,rec,raw):
    directory=root/'query_lineage';directory.mkdir(exist_ok=True)
    dest=directory/(rec['token']+'.npz');tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**raw,token=np.asarray(rec['token']),available=np.asarray([t['available'] for t in rec['targets']]))
    tmp.replace(dest)

def fast_zip_init(self,*args,**kwargs):
    compression=kwargs.get('compression',args[2] if len(args)>2 else zipfile.ZIP_STORED)
    if compression==zipfile.ZIP_DEFLATED and len(args)<5 and kwargs.get('compresslevel') is None:
        kwargs['compresslevel']=1
    return original_zip_init(self,*args,**kwargs)


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--pilot',action='store_true')
    args,_=p.parse_known_args();root=args.root/('smoke' if args.pilot else '')
    cv2.setNumThreads(1);zipfile.ZipFile.__init__=fast_zip_init
    write_json(root/'io_runtime.json',dict(opencv_threads=1,opencv_version=cv2.__version__,zip_compression_level=1,
        scope='Only CPU image scheduling and reversible lossless file compression; native numerical exporter unchanged',wrapper_sha256=sha256(__file__)))
    write_json(root/'query_lineage_protocol.json',dict(version=1,current_parent_queries=720,points_per_query=48,
        horizons=[2,4,6],coordinate='source native ego / corresponding predicted native future ego',
        dtype='FP32 coordinates and original maximum logit; uint8 original argmax class',
        scope='Additional read-only snapshots. All current parents retained, regardless of future confidence; future-born slots analyzed by regional diagnostics, not assigned a false historical parent.',
        wrapper_sha256=sha256(__file__)))
    import tools.export_sparseworld_full_regions as exporter
    forecast=exporter.forecast_with_position_gradient;predict=exporter.predict
    def capture_forecast(instance,encoded,stamps,ego,**kwargs):
        out=forecast(instance,encoded,stamps,ego,**kwargs)
        RAW.update(collect_lineage(instance,encoded,stamps,out));return out
    def capture_predict(model,cache,data,imgs,rec):
        RAW.clear();result=predict(model,cache,data,imgs,rec);save_lineage(root,rec,RAW);RAW.clear();return result
    exporter.forecast_with_position_gradient=capture_forecast;exporter.predict=capture_predict
    exporter.main()

if __name__=='__main__':main()
