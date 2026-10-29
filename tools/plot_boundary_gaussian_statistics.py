"""Export scene distributions and error proportions as standalone PNG/PDF figures."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/boundary_gaussian_full'
DEST=ROOT/'analysis/figures/boundary_gaussian_full'


def main():
    summary=json.loads((RUN/'summary.json').read_text())
    assert summary['status']=='complete','Never plot partial results as a full census'
    with (RUN/'scene_gaussian_details.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==850*4
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':200})
    colors={'train':'#2878B5','val':'#D85838'}
    names={'train':'Train (700 scenes)','val':'Validation (150 scenes)'}
    for metric,figure_title,filename in (
            ('boundary_any_fraction','Boundary Gaussian prevalence across all 850 scenes','scene_boundary_distribution'),
            ('boundary_foreground_fraction','Mixing at least two foreground GT classes (empty does not count)','scene_foreground_boundary_distribution')):
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True,constrained_layout=True)
        for ax,domain,title in zip(axes,('all','camera'),('Full GT grid','Camera mask')):
            for split in ('train','val'):
                for support,style,label in (('native','-','native support'),('core3sigma','--','3-sigma core')):
                    x=np.sort([float(r[metric])*100 for r in rows if r['split']==split and r['scope']==support+'_'+domain])
                    y=np.arange(1,len(x)+1)/len(x)
                    ax.step(np.r_[0,x,100],np.r_[0,y,1],where='post',color=colors[split],linestyle=style,
                            linewidth=1.8,label=f'{names[split]}, {label}')
            ax.set(xlabel='Boundary Gaussian fraction in a scene (%)',title=title,xlim=(0,100),ylim=(0,1))
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
        axes[0].set_ylabel('Cumulative fraction of scenes')
        axes[1].legend(loc='lower right' if metric=='boundary_foreground_fraction' else 'upper left',
                       fontsize=8,frameon=False)
        fig.suptitle(figure_title,fontsize=13)
        for suffix in ('png','pdf'):fig.savefig(DEST/f'{filename}.{suffix}')
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    metrics=('wrong_class_fraction','miss_fraction','false_positive_fraction')
    for ax,scope,title in zip(axes,('native_all','native_camera'),('Full GT grid','Camera mask')):
        x=np.arange(3);width=.34
        for i,split in enumerate(('train','val')):
            errors=summary['statistics'][split]['errors'][scope]['union_any']
            values=np.array([errors[m] for m in metrics])*100
            bars=ax.bar(x+(i-.5)*width,values,width,color=colors[split],label=names[split])
            ax.bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=3,fontsize=9)
        ax.set_xticks(x,['Wrong class','Miss','False positive'])
        ax.set(title=title,ylabel='Fraction of boundary-covered voxels (%)')
        ax.set_ylim(0,max(1,max(p.get_height() for p in ax.patches)*1.25))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Original model predictions within deduplicated boundary support',fontsize=13)
    for suffix in ('png','pdf'):fig.savefig(DEST/f'boundary_union_errors.{suffix}')
    plt.close(fig)
    print(DEST)


if __name__=='__main__':main()
