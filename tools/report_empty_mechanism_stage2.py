"""Aggregate scene-paired effects from stage-two artifacts, without modifying experiments."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.empty_mechanism_stage2 import RUN,read,write_json,initialize,CONDITIONS
from tools.eval_empty_decision_optimization import metrics


def statistics(outcomes,w):
    counts=np.asarray([o["counts"] for o in outcomes],dtype=np.float64)
    c=(w@counts.reshape(len(counts),-1)).reshape(-1,18,18)
    m=metrics(c)
    result={k:m[v] for k,v in [("fnr","occupied_FNR"),("fpr","empty_FPR"),
                               ("miou","mIoU"),("iou","IoU")]}
    occupied=c[:,:17,:].sum((1,2))
    correct=w@np.asarray([o["foreground_correct"] for o in outcomes])
    result["fg_rank_correct_rate"]=100*np.divide(correct,occupied,out=np.zeros(len(w)),where=occupied>0)
    for cohort in ["recoverable","correct","true_empty","occupied"]:
        for quantity,denom in [("amplitude","supported"),("support","n"),
                               ("margin","n"),("share","n"),("empty","n")]:
            numerator=np.array([o["cohorts"][cohort][quantity+"_sum"] for o in outcomes])
            denominator=w@np.array([o["cohorts"][cohort][denom] for o in outcomes])
            result[cohort+"_"+quantity]=np.divide(w@numerator,denominator,
                out=np.full(len(w),np.nan),where=denominator>0)
    result["gaussian_amplitude"]=(w@np.array([o["gaussian_amplitude_mean"] for o in outcomes]))/w.sum(1)
    return result


def effect(baselines,outcomes,seed=928):
    n=len(outcomes)
    rng=np.random.default_rng(seed)
    w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000)])
    before,after=statistics(baselines,w),statistics(outcomes,w)
    result={}
    for name,values in after.items():
        delta=values-before[name]
        result[name]=dict(before=float(before[name][0]),after=float(values[0]),
            delta=float(delta[0]),ci95=np.nanpercentile(delta[1:],[2.5,97.5]).tolist())
    return dict(scenes=n,effect=result,fn_harmed_scenes=sum(a["empty_fn"]>b["empty_fn"] for a,b in zip(outcomes,baselines)),
                correct_to_empty=sum(o["correct_to_empty"] for o in outcomes),
                rescued=sum(o["rescued"] for o in outcomes))


def local_summary():
    result={}
    for epoch in [1,3,5]:
        result[str(epoch)]={}
        for split in ["train","val"]:
            records=[read(p) for p in sorted((RUN/"local"/f"epoch{epoch}").glob(split+"_*.json"))]
            if not records:continue
            baselines=[r["baseline"] for r in records]
            rows=[]
            for radius in [1e-4,1e-3]:
                for condition in CONDITIONS:
                    selected=[next(v for v in r["interventions"] if
                              (v["radius"],v["condition"])==(radius,condition)) for r in records]
                    row=effect(baselines,[v["outcome"] for v in selected])
                    row.update(condition=condition,radius=radius,
                               actual_radius_mean=float(np.mean([v["raw_step_relative"] for v in selected])),
                               gradient_norm_mean=float(np.mean([v["gradient_norm"] for v in selected])))
                    rows.append(row)
            # Paired 2x2 interaction, no baseline needed since coefficients sum to zero.
            interactions=[]
            for radius in [1e-4,1e-3]:
                order=["combo_111","combo_011","B2_no_empty_ce","B3_no_set_no_empty_ce"]
                source=[[next(v["outcome"] for v in r["interventions"]
                        if (v["radius"],v["condition"])==(radius,c)) for r in records] for c in order]
                n=len(records);rng=np.random.default_rng(928)
                w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000)])
                stats=[statistics(o,w) for o in source];ints={}
                for name in stats[0]:
                    v=stats[0][name]-stats[1][name]-stats[2][name]+stats[3][name]
                    ints[name]=dict(value=float(v[0]),ci95=np.nanpercentile(v[1:],[2.5,97.5]).tolist())
                interactions.append(dict(radius=radius,statistics=ints))
            result[str(epoch)][split]=dict(scenes=len(records),conditions=rows,interactions=interactions)
    return result


def spatial_summary():
    result={}
    for epoch in [1,3,5]:
        directory=RUN/"local"/f"epoch{epoch}"
        def load(split):
            files=sorted(directory.glob(split+"_*.npz"))
            return [dict(np.load(f)) for f in files]
        train=load("train");val=load("val")
        if not train or not val:continue
        boundaries={}
        for name in ["amplitude","size","support"]:
            merged=np.concatenate([a[name][a["valid"]] for a in train])
            boundaries[name]=np.quantile(merged,[.25,.5,.75]).tolist()
        entry=dict(train_boundaries=boundaries)
        for split,arrays in [("train",train),("val",val)]:
            groups=[]
            gaussian_total=sum(int(a["valid"].sum()) for a in arrays)
            mixed_total=sum(int((a["valid"]&(a["h"]>1e-5)&(a["h"]<1-1e-5)).sum()) for a in arrays)
            for k in range(4):
                totals=np.zeros(6)
                for a in arrays:
                    select=a["valid"]&(a["h"]>=k/4)&(a["h"]<((k+1)/4 if k<3 else 1.00001))
                    n=int(select.sum())
                    vals=[n,float(a["h"][select].sum()),float(a["amplitude"][select].sum())]
                    vals += [float((a["amplitude_derivative_"+source][select]/
                               np.maximum(a["amplitude"][select],1e-30)).sum())
                             for source in ["ce_empty","set_negative","ce_occupied"]]
                    totals+=vals
                groups.append(dict(h_bin=[k/4,(k+1)/4],gaussians=int(totals[0]),
                    mean_h=float(totals[1]/max(totals[0],1)),
                    mean_amplitude=float(totals[2]/max(totals[0],1)),
                    relative_amplitude_derivative_ce_empty=float(totals[3]/max(totals[0],1)),
                    relative_amplitude_derivative_set_negative=float(totals[4]/max(totals[0],1)),
                    relative_amplitude_derivative_ce_occupied=float(totals[5]/max(totals[0],1))))
            # Within-frame target-class x amplitude x size x support strata:
            # contrast high empty coverage >=.75 with low <=.25.
            scene_contrasts=[]
            for a in arrays:
                code=a["target"].astype(np.int64).copy()
                for name in ["amplitude","size","support"]:
                    code=code*4+np.searchsorted(boundaries[name],a[name],side="right")
                weights=[];deltas=[]
                for group in np.unique(code[a["valid"]]):
                    lo=a["valid"]&(code==group)&(a["h"]<=.25)
                    hi=a["valid"]&(code==group)&(a["h"]>=.75)
                    nl,nh=int(lo.sum()),int(hi.sum())
                    if min(nl,nh)<2:continue
                    v=a["amplitude_derivative_ce_empty"]/np.maximum(a["amplitude"],1e-30)
                    weights.append(nl*nh/(nl+nh))
                    deltas.append(float(v[hi].mean()-v[lo].mean()))
                if weights:
                    scene_contrasts.append([float(np.dot(weights,deltas)),float(np.sum(weights)),len(weights)])
                else:
                    scene_contrasts.append([0.,0.,0])
            data=np.asarray(scene_contrasts)
            n=len(data);rng=np.random.default_rng(928)
            w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000)])
            den=w@data[:,1]
            values=np.divide(w@data[:,0],den,out=np.full(len(w),np.nan),where=den>0)
            entry[split]=dict(groups=groups,
                shared_coverage=dict(valid_gaussians=gaussian_total,mixed_gaussians=mixed_total,
                    mixed_fraction=mixed_total/max(gaussian_total,1),tolerance=1e-5,
                    interpretation="supported Gaussian has contributions to both true-occupied and true-empty voxels"),
                matched_high_minus_low_relative_ce_empty_derivative=dict(value=float(values[0]),
                    ci95=np.nanpercentile(values[1:],[2.5,97.5]).tolist(),
                    scenes_with_matches=int((data[:,1]>0).sum()),strata=int(data[:,2].sum())))
        result[str(epoch)]=entry
    return result


def adamw_summary():
    result={}
    for epoch in [1,3,5]:
        result[str(epoch)]={}
        for mode in ["classifier","full"]:
            records=[read(p) for p in sorted((RUN/"adamw"/f"epoch{epoch}"/mode).glob("pair_*.json"))]
            if not records:continue
            result[str(epoch)][mode]={}
            for role in ["source","heldout"]:
                bases=[r["baselines"][role] for r in records]
                rows=[]
                for condition in records[0]["interventions"]:
                    name=condition["condition"]
                    after=[next(x for x in r["interventions"] if x["condition"]==name)["outcomes"][role] for r in records]
                    row=effect(bases,after);row["condition"]=name;rows.append(row)
                result[str(epoch)][mode][role]=rows
    return result


def short_summary():
    result={}
    for epoch in [1,3]:
        rows=[]
        for folder in sorted((RUN/"short"/f"epoch{epoch}").glob("*")):
            if not (folder/"baseline.json").exists():continue
            before=read(folder/"baseline.json")["records"]
            for step in [1,16,64]:
                path=folder/f"step_{step:02d}.json"
                if not path.exists():continue
                after=read(path)["records"]
                assert [r["frame"] for r in before]==[r["frame"] for r in after]
                row=effect([r["outcome"] for r in before],[r["outcome"] for r in after])
                row.update(condition=folder.name,step=step,
                           empty_bias_before=float(np.mean([r["empty_bias"] for r in before])),
                           empty_bias_after=float(np.mean([r["empty_bias"] for r in after])),
                           empty_scalar_before=float(np.mean([r["empty_scalar"] for r in before])),
                           empty_scalar_after=float(np.mean([r["empty_scalar"] for r in after])))
                rows.append(row)
        result[str(epoch)]=rows
    return result


def short_contrasts():
    result={}
    for epoch in [1,3]:
        base=RUN/"short"/f"epoch{epoch}"/"B0"/"step_64.json"
        if not base.exists():continue
        before=read(base)["records"]
        rows=[]
        for folder in sorted((RUN/"short"/f"epoch{epoch}").glob("*")):
            path=folder/"step_64.json"
            if not path.exists():continue
            after=read(path)["records"]
            row=effect([r["outcome"] for r in before],[r["outcome"] for r in after])
            row["condition"]=folder.name
            rows.append(row)
        result[str(epoch)]=rows
    return result


def short_strength_contrasts():
    result={}
    for epoch in [1,3]:
        folder=RUN/"short"/f"epoch{epoch}"
        path=folder/"project_set"/"step_64.json"
        if not path.exists():continue
        after=read(path)["records"];rows=[]
        for reference in ["half_set","norm_set","B0_reference"]:
            path=folder/reference/"step_64.json"
            if not path.exists():continue
            before=read(path)["records"]
            row=effect([v["outcome"] for v in before],[v["outcome"] for v in after])
            row["reference"]=reference;rows.append(row)
        result[str(epoch)]=rows
    return result


def adamw_contrasts():
    result={}
    for epoch in [1,3,5]:
        for mode in ["classifier","full"]:
            records=[read(p) for p in sorted((RUN/"adamw"/f"epoch{epoch}"/mode).glob("pair_*.json"))]
            if not records:continue
            before=[next(v for v in r["interventions"] if v["condition"]=="B0")["outcomes"]["heldout"] for r in records]
            rows=[]
            for condition in records[0]["interventions"]:
                name=condition["condition"]
                after=[next(v for v in r["interventions"] if v["condition"]==name)["outcomes"]["heldout"] for r in records]
                row=effect(before,after);row["condition"]=name;rows.append(row)
            result[f"{epoch}_{mode}"]=rows
    return result


def equal_bias_summary():
    directory=RUN/"equal_bias"
    models={};contrasts={}
    for epoch in [1,3]:
        values={}
        for condition in ["B0","project_set"]:
            folder=directory/f"epoch{epoch}_{condition}"
            if not (folder/"complete.json").exists():continue
            values[condition]=read(folder/"complete.json")
            values[condition]["thresholds"]=read(folder/"selection.json")["thresholds"]
        models[str(epoch)]=values
        if len(values)!=2:continue
        before=directory/f"epoch{epoch}_B0"
        after=directory/f"epoch{epoch}_project_set"
        frames=read(before/"validation_frames.json")
        assert frames==read(after/"validation_frames.json")
        scenes=sorted(set(v["scene"] for v in frames))
        index={v:i for i,v in enumerate(scenes)}
        n=len(scenes);rng=np.random.default_rng(928)
        w=np.vstack([np.ones(n),rng.multinomial(n,np.full(n,1/n),size=1000)])
        row={}
        a=np.load(before/"validation.npz");b=np.load(after/"validation.npz")
        for state in ["native","calibrated"]:
            source=np.zeros((n,18,18),np.float64);target=source.copy()
            for k,meta in enumerate(frames):
                source[index[meta["scene"]]]+=a[state][k]
                target[index[meta["scene"]]]+=b[state][k]
            bs=metrics((w@source.reshape(n,-1)).reshape(-1,18,18))
            ts=metrics((w@target.reshape(n,-1)).reshape(-1,18,18))
            entry={}
            for key,metric in [("miou","mIoU"),("iou","IoU"),("fnr","occupied_FNR"),("fpr","empty_FPR")]:
                delta=ts[metric]-bs[metric]
                entry[key]=dict(before=float(bs[metric][0]),after=float(ts[metric][0]),
                    delta=float(delta[0]),ci95=np.percentile(delta[1:],[2.5,97.5]).tolist())
            row[state]=dict(scenes=n,frames=len(frames),effect=entry)
        contrasts[str(epoch)]=row
    return dict(models=models,contrasts=contrasts,exploratory=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--interim",action="store_true")
    a=parser.parse_args()
    summary=dict(local=local_summary(),spatial=spatial_summary(),
                 adamw=adamw_summary(),adamw_contrasts=adamw_contrasts(),
                 short=short_summary(),short_contrasts=short_contrasts(),
                 short_strength_contrasts=short_strength_contrasts(),
                 equal_bias=equal_bias_summary(),
                 status="interim" if a.interim else "aggregated")
    dest=RUN/("interim_summary.json" if a.interim else "summary.json")
    write_json(dest,summary)
    print(dest)
    for e in ["1","3","5"]:
        if "val" not in summary["local"].get(e,{}):continue
        print("epoch",e,"local val")
        for r in summary["local"][e]["val"]["conditions"]:
            if r["radius"]==.001:
                print(r["condition"],*[round(r["effect"][k]["delta"],5) for k in ["fnr","fpr","miou"]])
    for e,rows in summary["short"].items():
        for r in rows:
            if r["step"]==64:
                print("short64",e,r["condition"],*[round(r["effect"][k]["delta"],5) for k in ["fnr","fpr","miou"]])


if __name__=="__main__":
    main()
