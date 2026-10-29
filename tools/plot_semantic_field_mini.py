"""Standalone paper-style figure from completed counts; no training dependencies."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=ROOT/'work_dirs/semantic_field_mini_v1'

def main():
    result=json.loads((RUN/'summary.json').read_text());assert result['status']=='complete'
    m=result['metrics'];b=m['baseline'];names=('geometry','semantic');colors=('#2f78a4','#c27535')
    labels=('Geometry field','Local semantic field')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12,6.7),constrained_layout=True)
    panels=[('mIoU','mIoU (higher is better)','mIoU_delta_95CI'),
        ('geo1','Geometric 1-sigma coverage','geo1_delta_95CI'),('same1','Predicted same-class core coverage',None),
        ('FN_rate','Misses / GT foreground (lower is better)',None),
        ('FP_over_GTfg','False positives / GT foreground (lower is better)',None),
        ('high_core','Coverage of initial high-load GT subset',None)]
    for ax,(key,title,ci) in zip(axes.flat,panels):
        delta=np.array([m[n][key]-b[key] for n in names]);x=np.arange(2)
        ax.axhline(0,color='.5',lw=.8,zorder=0);ax.bar(x,delta,color=colors,width=.55)
        if ci:
            intervals=np.array([result['paired_vs_baseline'][n][ci] for n in names])
            ax.vlines(x,intervals[:,0],intervals[:,1],color='black',lw=1.2)
            ax.scatter(np.repeat(x,2),intervals.ravel(),marker='_',s=60,color='black')
        ax.set_xticks(x,labels,rotation=12,ha='right');ax.set_title(title,fontsize=10)
        ax.set_ylabel('Change vs matched baseline (pp)');ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(delta):
            anchor=(max(v,intervals[i,1]) if v>=0 else min(v,intervals[i,0])) if ci else v
            ax.annotate(f'{v:+.3f}',(i,anchor),xytext=(0,5 if v>=0 else -5),textcoords='offset points',
                ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
        lo,hi=ax.get_ylim();margin=max((hi-lo)*.2,.03);ax.set_ylim(lo-margin,hi+margin)
    fig.suptitle('Local field supervision: full Occ3D validation after matched 4096-frame fine-tuning',fontsize=12)
    folder=ROOT/'analysis/figures';folder.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(folder/f'semantic_field_mini_comparison.{suffix}',dpi=180)
    plt.close(fig)
    # Additional unweighted-class and non-road checks prevent volume dominance.
    checks={}
    for n in ('initial','baseline',*names):
        a=np.load(RUN/'aggregate'/f'{n}.npz');cc=a['class_coverage'];valid=cc[:,0]>0
        core=cc[:,1]/np.maximum(cc[:,0],1);same=cc[:,4]/np.maximum(cc[:,0],1)
        other=np.arange(17)!=11
        checks[n]=dict(macro_core1=100*float(core[valid].mean()),macro_same_core1=100*float(same[valid].mean()),
            nonroad_core1=100*float(cc[other,1].sum()/cc[other,0].sum()),
            nonroad_same_core1=100*float(cc[other,4].sum()/cc[other,0].sum()))
    (RUN/'coverage_class_balance_audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
