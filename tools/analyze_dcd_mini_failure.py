"""Read-only post-hoc analysis of completed DCD logs and full-val aggregates."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'work_dirs/dcd_mini_1ep'
OUT=RUN/'failure_analysis'
FIG=ROOT/'analysis/figures/dcd_mini_failure'


def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    summary=json.loads((RUN/'summary.json').read_text())
    assert summary['status']=='complete' and summary['val_frames']==6019
    trace=[json.loads(s) for s in (RUN/'train/trace.jsonl').read_text().splitlines()]
    assert len(trace)==4096 and len({r['index'] for r in trace})==4096
    keys=('loss_pts','loss_cls','pred_to_gt_distance_m','gt_to_pred_distance_m',
          'pred_exponential_mean','gt_exponential_mean','queried_prediction_fraction',
          'queried_gt_fraction','valid_gt_points')
    windows=[]
    for i in range(4):
        rows=trace[i*1024:(i+1)*1024]
        windows.append(dict(window=i+1,first_frame=i*1024+1,last_frame=(i+1)*1024,
            **{k:float(np.mean([r['losses']['OPUSSetDCDLoss/'+k] for r in rows])) for k in keys},
            occupancy_loss=float(np.mean([r['losses']['OccupancyLoss'] for r in rows]))))
    csv_write(OUT/'training_windows.csv',windows)
    # The names are frozen in the baseline coverage protocol; no model is built.
    fields=json.loads((ROOT/'work_dirs/gaussian_voxel_coverage_full/protocol.json').read_text())['fields']
    arrays={}
    hist_rows=[]
    for name in ('before','after'):
        with np.load(RUN/f'aggregate/{name}_val.npz') as z:
            arrays[name]={k:z[k] for k in ('count_hist','field_sum','field_n','class_range_voxels','class_range_sum')}
        a=arrays[name];hist=a['count_hist'][:,1].sum(0);den=int(hist.sum())
        assert den==int(summary['models'][name]['GTfg'])
        counts=[hist[0],hist[1],hist[2:4].sum(),hist[4:8].sum(),hist[8:].sum()]
        for label,n in zip(('0','1','2–3','4–7','≥8'),counts):
            hist_rows.append(dict(model=name,core_count_bin=label,voxels=int(n),fraction=float(n/den)))
    csv_write(OUT/'core_count_distribution.csv',hist_rows)
    original=list(csv.DictReader((RUN/'class_metrics.csv').open(encoding='utf-8-sig')))
    byname={name:{int(r['class_id']):r for r in original if r['model']==name} for name in ('before','after')}
    class_rows=[]
    for c in range(17):
        b,a=byname['before'][c],byname['after'][c]
        core={name:float(arrays[name]['class_range_sum'][:,c,:,fields.index('within_1sigma')].sum()/
            arrays[name]['class_range_voxels'][:,c].sum()) for name in ('before','after')}
        class_rows.append(dict(class_id=c,class_name=b['class_name'],
            before_core_mean=core['before'],after_core_mean=core['after'],
            delta_IoU=float(a['IoU_percent'])-float(b['IoU_percent']),
            delta_FN=int(float(a['FN_to_empty']))-int(float(b['FN_to_empty'])),
            delta_FP=int(float(a['FP_from_empty']))-int(float(b['FP_from_empty']))))
    csv_write(OUT/'class_changes.csv',class_rows)
    empty={};buffers={}
    checkpoint_hashes={}
    for name,path in (('before',ROOT/'ckpts/opus_gaussian_8f_tiny.pth'),('after',RUN/'train/epoch_1.pth')):
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
        checkpoint_hashes[name]=digest.hexdigest()
        z=torch.load(path,map_location='cpu',weights_only=False)
        state=z['state_dict']
        empty[name]={k:float(state['head.'+k]) for k in ('empty_scalar','empty_bias')}
        buffers[name]={k:v.clone() for k,v in state.items() if k.startswith('head.empty_') and k not in ('head.empty_scalar','head.empty_bias')}
        del z,state
    assert checkpoint_hashes['before']==json.loads((RUN/'baseline_reuse.json').read_text())['checkpoint_sha256']
    assert checkpoint_hashes['after']==json.loads((RUN/'train/complete.json').read_text())['checkpoint_sha256']
    assert buffers['before'].keys()==buffers['after'].keys()
    assert all(torch.equal(v,buffers['after'][k]) for k,v in buffers['before'].items())
    gradnorm=np.asarray([r['grad_norm'] for r in trace if 'grad_norm' in r])
    means={name:{k:float(arrays[name]['field_sum'][:,fields.index(k)].sum()/arrays[name]['field_n'][:,fields.index(k)].sum())
        for k in ('max_foreground','empty_score','foreground_empty_ratio')} for name in ('before','after')}
    covered={name:summary['models'][name]['n_1sigma']/(1-summary['models'][name]['no_1sigma']/100) for name in ('before','after')}
    derivative=[dict(distance_m=r,exponential=float(np.exp(-(r/.4)**2)),
        radial_derivative_unit_weight=float(2*r/.4**2*np.exp(-(r/.4)**2))) for r in (.2,.4,.8,1.2,1.6)]
    result=dict(status='complete',basis='Existing 4096-frame trace and complete 6019-frame paired validation; no new training or inference',
        validation=summary,training_windows=windows,core_distribution=hist_rows,class_changes=class_rows,
        conditional_core_count=covered,empty_parameters=empty,empty_buffers_unchanged=True,
        checkpoint_sha256=checkpoint_hashes,
        foreground_score_means=means,radial_derivative=derivative,
        optimizer_updates=len(gradnorm),grad_norm_max=float(gradnorm.max()),
        clipped_updates=int((gradnorm>35).sum()),
        extra_zero_core_voxels=int(round((summary['models']['after']['no_1sigma']-summary['models']['before']['no_1sigma'])/100*summary['models']['before']['GTfg'])),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),RUN/'train/trace.jsonl',RUN/'summary.json',ROOT/'loss/opus_set_dcd_loss.py')})
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    x=np.linspace(0,1.6,500);e=np.exp(-(x/.4)**2)
    axes[0].plot(x,e,label='Exponential distance term')
    axes[0].set(xlabel='Distance (m)',ylabel='exp(-(r / 0.4)^2)',ylim=(0,1.05))
    axes[1].plot(x,np.maximum(2*x/.4**2*e,1e-8),label='DCD radial gradient, weight=1')
    # Axis-aligned displacement for the original summed-coordinate Smooth-L1.
    axes[1].plot(x,np.minimum(x/.2,1),ls='--',label='Smooth-L1, axis-aligned, beta=0.2')
    axes[1].set(xlabel='Distance (m)',ylabel='Single-pair derivative magnitude',yscale='log',ylim=(1e-6,4))
    axes[1].legend(fontsize=8)
    fig.savefig(FIG/'distance_and_gradient.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    x=np.arange(5)
    for i,(name,label) in enumerate((('before','Original tiny'),('after','DCD fine-tuned'))):
        y=[r['fraction']*100 for r in hist_rows if r['model']==name]
        ax.bar(x+(i-.5)*.36,y,width=.36,label=label)
    ax.set(xticks=x,xticklabels=['0','1','2–3','4–7','8+'],xlabel='Number of 1-sigma Gaussian cores',ylabel='GT foreground voxels (%)')
    ax.legend();fig.savefig(FIG/'core_count_distribution.png',dpi=180);plt.close(fig)
    print('DCD post-hoc analysis completed from existing artifacts.',flush=True)


if __name__=='__main__':
    main()
