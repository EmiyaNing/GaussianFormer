"""Standalone plots for the paired geometry/evidence experiments."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tools.gaussian_geometry_causal import RUN
from tools.report_gaussian_geometry_causal import COMPARISONS

DEST=ROOT/'analysis/figures/gaussian_geometry_causal'
LABELS=dict(no_amplitude='Block amplitude',no_geometry='Block geometry',
            no_both='Block both',no_empty_center_boost='Block selected boosts')


def save(fig,name):
    DEST.mkdir(parents=True,exist_ok=True)
    fig.savefig(DEST/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(DEST/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    s=json.loads((RUN/'summary.json').read_text())
    ci=json.loads((RUN/'scene_bootstrap.json').read_text())
    names=COMPARISONS;y=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,key,title in zip(axes,('mIoU','n_2sigma','CE_weighted'),
            ('Validation mIoU change (pp)','Validation 2-sigma count change','Training weighted CE change')):
        split='train' if key=='CE_weighted' else 'val'
        delta=np.array([s[n][split][key]-s['standard'][split][key] for n in names])
        ax.barh(y,delta,color=['#4477aa','#ee7733','#aa3377','#228833'])
        if key!='CE_weighted':
            lo=np.array([ci[n][split][key][0] for n in names])
            hi=np.array([ci[n][split][key][1] for n in names])
            ax.errorbar((lo+hi)/2,y,xerr=(hi-lo)/2,fmt='none',ecolor='black',capsize=3,lw=1)
        ax.axvline(0,color='black',lw=.8);ax.set_title(title)
        ax.set_yticks(y,[LABELS[n] for n in names] if ax is axes[0] else [])
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Paired short-training effects relative to standard; one training seed')
    fig.tight_layout();save(fig,'paired_effects')
    rows=[json.loads(x) for x in (RUN/'common/standard_s928/trace.jsonl').read_text().splitlines()]
    rows=[r for r in rows if 'gaussian' in r]
    x=np.array([(r['microbatch']+1)/8 for r in rows])
    A=np.array([r['gaussian']['amplitude'] for r in rows])
    alpha=np.array([r['gaussian']['opacity'] for r in rows])
    scale=np.array([r['gaussian']['scale_mean'] for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    axes[0].plot(x,A);axes[0].set_title('Semantic total amplitude A')
    axes[1].plot(x,alpha,color='#ee7733');axes[1].set_title('Opacity alpha');axes[1].set_ylim(0,1.03)
    axes[2].plot(x,scale,color='#228833');axes[2].axhline(.4,color='black',ls='--',lw=.8)
    axes[2].set_title('Mean scale (m)');axes[2].set_ylim(.08,.41)
    for ax in axes:ax.set_xlabel('AdamW update');ax.grid(alpha=.2)
    fig.suptitle('Common warmup trajectory; each point uses its actual training frame')
    fig.tight_layout();save(fig,'warmup_attributes')
    if (RUN/'mature_summary.json').exists():
        mature_plot(json.loads((RUN/'mature_summary.json').read_text()))


def mature_plot(result):
    """Show the mature pair separately from the badly underfit early branches."""
    base='mature_standard';control='mature_no_empty_center_boost'
    s=result['trained'];ref=result['reference']
    fig,axes=plt.subplots(2,3,figsize=(12,6.4))
    x=np.arange(2);width=.24
    for ax,key,title in zip(axes[0,:2],('mIoU','IoU'),
            ('Native semantic mIoU (%)','Native binary occupancy IoU (%)')):
        for j,(values,label,color) in enumerate(((ref,'Original checkpoint','#bbbbbb'),
                (s[base],'Standard continuation','#4477aa'),
                (s[control],'Selected-boost control','#228833'))):
            values=[values[split][key] for split in ('train','val')]
            bars=ax.bar(x+(j-1)*width,values,width,color=color,label=label)
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
        ax.set_xticks(x,['Train (700 scenes)','Val (150 scenes)'])
        maximum=max(v[sp][key] for v in (ref,s[base],s[control]) for sp in ('train','val'))
        ax.set_title(title);ax.set_ylim(0,min(100,maximum*1.18));ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(loc='upper right',fontsize=7,frameon=False)
    ax=axes[0,2]
    for j,(name,label,color) in enumerate(((base,'Standard','#4477aa'),
                                           (control,'Control','#228833'))):
        values=[s[name]['val'][key] for key in ('FN_rate','FPR')]
        bars=ax.bar(x+(j-.5)*.32,values,.32,label=label,color=color)
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
    ax.set_xticks(x,['FN / GT foreground','FP / GT empty'])
    ax.set_title('Validation errors (%)');ax.grid(axis='y',alpha=.2)
    ax.set_ylim(0,max(s[base]['val']['FN_rate'],s[control]['val']['FN_rate'])*1.25)
    ax=axes[1,0];keys=('mIoU','IoU','FN_rate','FPR');y=np.arange(4)
    delta=np.array([s[control]['val'][k]-s[base]['val'][k] for k in keys])
    interval=result['intervals']['val']
    lo=np.array([interval[k][0] for k in keys]);hi=np.array([interval[k][1] for k in keys])
    ax.barh(y,delta,color='#228833')
    ax.errorbar((lo+hi)/2,y,xerr=(hi-lo)/2,fmt='none',ecolor='black',capsize=3,lw=1)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(y,['mIoU','Binary IoU','FN rate','FP rate'])
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title('Control minus standard (pp)\nPaired scene bootstrap 95% intervals')
    for ax,key,title in zip(axes[1,1:],('n_2sigma','no_1sigma'),
            ('Mean 2-sigma GS count on GT foreground','GT foreground without 1-sigma support (%)')):
        for j,(name,label,color) in enumerate(((base,'Standard','#4477aa'),
                                               (control,'Control','#228833'))):
            bars=ax.bar(x+(j-.5)*.32,[s[name][sp][key] for sp in ('train','val')],
                        .32,color=color,label=label)
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
        ax.set_xticks(x,['Train','Val']);ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2)
        maximum=max(s[n][sp][key] for n in (base,control) for sp in ('train','val'))
        ax.set_ylim(0,maximum*1.2)
    dose=100*result['dose']['mean_blocked_fraction']
    fig.suptitle(f'Mature-checkpoint paired continuation: 128 updates per arm; one seed\n'
                 f'Control blocks boosts for {dose:.4f}% of predicted GS per training frame')
    fig.tight_layout();save(fig,'mature_effects')


if __name__=='__main__':main()
