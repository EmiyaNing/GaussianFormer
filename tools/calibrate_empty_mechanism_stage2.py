"""Train-only, equal class-threshold calibration of 64-step fork checkpoints."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage2 import RUN,read,write_json,file_sha256,build_loader,to_cuda,seed_all,network_output
from tools.empty_mechanism_stage1 import build_network
from tools.eval_empty_decision_optimization import metrics

CAL=RUN/"equal_bias"
LOW,HIGH,STEPS=-256.,256.,32
EDGES=np.arange(int((HIGH-LOW)*STEPS)+1,dtype=np.float64)/STEPS+LOW
NB=len(EDGES)+1


def initialize():
    CAL.mkdir(exist_ok=True)
    path=CAL/"protocol.json"
    if path.exists():return read(path)
    prior=read(ROOT/"work_dirs/empty_decision_optimization/protocol.json")
    p=dict(at=datetime.now().astimezone().isoformat(),exploratory_followup=True,
        motivation="early native projection gain coexists with worse foreground winner accuracy",
        models=[dict(epoch=e,condition=c) for e in [1,3] for c in ["B0","project_set"]],
        train_frames=4096,validation_frames=1024,fit_scenes=prior["fit_scenes"],
        tune_scenes=prior["tune_scenes"],domain="default camera mask, valid labels",
        threshold_grid=dict(minimum=LOW,maximum=HIGH,step=1/STEPS),
        fitting="maximize each foreground class IoU separately over all fitting voxels; closest-to-zero tie",
        tuning="choose common threshold shift on all training tuning voxels; maximize 17-class mIoU",
        shifts=(np.arange(-32,33)/32).tolist(),
        prediction="foreground winner unchanged; occupied iff native fg-minus-empty margin >= selected class threshold",
        no_validation_selection=True,bootstrap=1000,
        interpretation="strong static-calibration diagnostic; does not reproduce historical C5 regional constraints",
        full_training_screen="require native gains beyond strength controls and positive calibrated mIoU gain at both origins",
        source_sha256=file_sha256(__file__))
    write_json(path,p);return p


def counts_from_hist(hist,thresholds):
    """All edges are exact binary fractions. Right buckets preserve native >= ties."""
    tail=np.cumsum(hist[...,::-1],axis=-1,dtype=np.int64)[...,::-1]
    index=np.rint((np.asarray(thresholds)-LOW)*STEPS).astype(np.int64).clip(0,len(EDGES)-1)
    selected=tail[np.arange(18)[:,None],np.arange(17)[None,:],index[None,:]+1]
    counts=np.zeros((18,18),dtype=np.int64)
    counts[:,:17]=selected
    counts[:,17]=hist.sum((1,2))-selected.sum(1)
    return counts


def fitting_thresholds(hist):
    tail=np.cumsum(hist[...,::-1],axis=-1,dtype=np.int64)[...,::-1]
    truth=hist.sum((1,2))
    selected=tail[...,1:]
    thresholds=[]
    boundary=[]
    for c in range(17):
        tp=selected[c,c]
        fp=selected[:,c].sum(0)-tp
        iou=tp/np.maximum(truth[c]+fp,1)
        best=np.flatnonzero(iou==iou.max())
        i=best[np.abs(EDGES[best]).argmin()]
        thresholds.append(EDGES[i])
        boundary.append(bool(i in [0,len(EDGES)-1]))
    return np.asarray(thresholds),boundary


def tuning_thresholds(hist,thresholds):
    p=initialize()
    # Compute the tail once; 65 x 17 integer queries are then sufficient.
    tail=np.cumsum(hist[...,::-1],axis=-1,dtype=np.int64)[...,::-1]
    truth=hist.sum((1,2));rows=[]
    for shift in p["shifts"]:
        t=np.clip(thresholds-shift,LOW,HIGH)
        index=np.rint((t-LOW)*STEPS).astype(np.int64)
        selected=tail[np.arange(18)[:,None],np.arange(17)[None,:],index[None,:]+1]
        c=np.zeros((18,18),np.int64);c[:,:17]=selected;c[:,17]=truth-selected.sum(1)
        m=metrics(c[None])
        rows.append(dict(shift=shift,miou=float(m["mIoU"][0]),fpr=float(m["empty_FPR"][0])))
    chosen=min(rows,key=lambda r:(-r["miou"],r["fpr"],abs(r["shift"]),r["shift"]))
    return np.clip(thresholds-chosen["shift"],LOW,HIGH),chosen,rows


def fields(output,batch):
    scores=output["pred_occ"][-1][0]
    y=output["sampled_label"][0].long()
    mask=batch["occ_cam_mask"][0].bool().flatten()&(y>=0)&(y<18)
    top,fg=scores[:17,mask].max(0)
    return y[mask],fg,top-scores[17,mask]


def counts(y,pred):
    return torch.bincount(y*18+pred,minlength=324).reshape(18,18)


def run(epoch,condition):
    p=initialize();seed_all(928)
    dest=CAL/f"epoch{epoch}_{condition}";dest.mkdir(exist_ok=True)
    if (dest/"complete.json").exists():return
    checkpoint=RUN/"short"/f"epoch{epoch}"/condition/"final.pth"
    sha=file_sha256(checkpoint)
    net,_=build_network(epoch)
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    net.load_state_dict(state["state_dict"],strict=True);net.eval()
    fit=set(p["fit_scenes"]);tune=set(p["tune_scenes"])
    start=time.monotonic()
    cache=dest/"train_hist.npz"
    if cache.exists():
        a=np.load(cache);assert str(a["checkpoint_sha256"])==sha
        hist=torch.as_tensor(a["hist"],device="cuda")
        native=torch.as_tensor(a["native"],device="cuda")
        frames=a["frames"].tolist();next_index=len(frames)
    else:
        hist=torch.zeros(2,18,17,NB,device="cuda",dtype=torch.int64)
        native=torch.zeros(2,18,18,device="cuda",dtype=torch.int64)
        frames=[];next_index=0
    edge=torch.as_tensor(EDGES,device="cuda",dtype=torch.float32)
    dataset,loader,indices=build_loader("train",range(next_index,4096))
    with torch.inference_mode():
        for index,batch in zip(indices,loader):
            scene=dataset.keyframes[index][0]
            assert (scene in fit) != (scene in tune)
            cohort=0 if scene in fit else 1
            batch=to_cuda(batch);out=network_output(net,batch)
            y,fg,margin=fields(out,batch)
            bucket=torch.bucketize(margin.contiguous(),edge,right=True)
            code=(y*17+fg)*NB+bucket
            hist[cohort].add_(torch.bincount(code,minlength=18*17*NB).reshape(18,17,NB))
            native[cohort].add_(counts(y,torch.where(margin>=0,fg,17)))
            frames.append(index)
            del out,batch,y,fg,margin,bucket,code
            if (index+1)%512==0 or index==4095:
                np.savez_compressed(cache,hist=hist.cpu().numpy(),native=native.cpu().numpy(),
                                    frames=np.asarray(frames),checkpoint_sha256=np.array(sha))
                print(f"equal bias epoch={epoch} {condition} train={index+1}/4096 "
                      f"elapsed={time.monotonic()-start:.1f}s",flush=True)
    h=hist.cpu().numpy();native_cpu=native.cpu().numpy()
    for group in [0,1]:
        assert np.array_equal(counts_from_hist(h[group],np.zeros(17)),native_cpu[group])
    thresholds,boundary=fitting_thresholds(h[0])
    chosen,selection,candidates=tuning_thresholds(h[1],thresholds)
    assert not any(boundary),"Calibration optimum reached broad fitting boundary"
    write_json(dest/"selection.json",dict(thresholds=chosen.tolist(),fitting_thresholds=thresholds.tolist(),
        selection=selection,candidates=candidates,boundary=boundary,checkpoint_sha256=sha,
        protocol_sha256=file_sha256(CAL/"protocol.json"),source_sha256=file_sha256(__file__)))
    del hist,native
    tensor_thresholds=torch.as_tensor(chosen,device="cuda",dtype=torch.float64)
    valset,valloader,indices=build_loader("val")
    raw,calibrated,metadata=[],[],[]
    with torch.inference_mode():
        for index,batch in zip(indices,valloader):
            batch=to_cuda(batch);out=network_output(net,batch)
            y,fg,margin=fields(out,batch)
            raw.append(counts(y,torch.where(margin>=0,fg,17)).cpu().numpy())
            calibrated.append(counts(y,torch.where(margin.double()>=tensor_thresholds[fg],fg,17)).cpu().numpy())
            scene,scene_frame=valset.keyframes[index]
            metadata.append(dict(index=index,scene=scene,scene_frame=int(scene_frame)))
            del out,batch,y,fg,margin
            if (index+1)%256==0:
                print(f"equal bias epoch={epoch} {condition} val={index+1}/1024 "
                      f"elapsed={time.monotonic()-start:.1f}s",flush=True)
    np.savez_compressed(dest/"validation.npz",native=np.stack(raw),calibrated=np.stack(calibrated))
    write_json(dest/"validation_frames.json",metadata)
    mraw=metrics(np.sum(raw,axis=0)[None]);mcal=metrics(np.sum(calibrated,axis=0)[None])
    write_json(dest/"complete.json",dict(status="complete",epoch=epoch,condition=condition,
        train_frames=len(frames),validation_frames=len(raw),selection=selection,
        native={k:float(mraw[k][0]) for k in ["mIoU","IoU","occupied_FNR","empty_FPR"]},
        calibrated={k:float(mcal[k][0]) for k in ["mIoU","IoU","occupied_FNR","empty_FPR"]},
        elapsed_seconds=time.monotonic()-start,checkpoint_sha256=sha,
        protocol_sha256=file_sha256(CAL/"protocol.json"),source_sha256=file_sha256(__file__)))
    print("equal bias complete",epoch,condition,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch",type=int,default=1)
    parser.add_argument("--condition",choices=["B0","project_set"],default="B0")
    parser.add_argument("--initialize",action="store_true")
    args=parser.parse_args()
    if args.initialize:initialize()
    else:run(args.epoch,args.condition)
