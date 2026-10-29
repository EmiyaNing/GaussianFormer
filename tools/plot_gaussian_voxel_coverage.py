"""Export standalone coverage-count and Gaussian-position figures."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/gaussian_voxel_coverage_full'
FIG=ROOT/'analysis/figures/gaussian_voxel_coverage_full'


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(FIG/f'{name}.{ext}',dpi=200)
    plt.close(fig)


def main():
    s=json.loads((RUN/'summary.json').read_text())
    assert s['status']=='complete'
    z=np.load(RUN/'global_arrays.npz')
    groups=('miss','detected_occupied')
    group_labels=('Missed foreground','Detected occupancy')
    colors=('#cc5a4a','#347db1')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for row,split in enumerate(('train','val')):
        hist=z[split+'__count_hist']
        for ax,j,label in zip(axes[row],(0,2),('Native positive support','Within 2 sigma')):
            for i,g in enumerate(groups):
                cdf=np.cumsum(hist[i,j])/hist[i,j].sum()
                x=np.arange(len(cdf))
                ax.step(x,cdf,where='post',color=colors[i],label=group_labels[i],linewidth=2)
            q95=max(s['statistics'][split][g]['counts'][('native','within_1sigma','within_2sigma')[j]]['q95'] or 1024 for g in groups)
            xmax=max(8,int(q95*1.3))
            ax.set(xlabel='Number of Gaussians',ylabel='Fraction of foreground voxels',
                   title=f'{split.capitalize()}: {label}',xlim=(0,xmax),ylim=(0,1.01))
            ax.set_xscale('symlog',linthresh=1)
            ticks=[x for x in (0,1,2,4,8,16,32,64,128,256,512,1024) if x<=xmax]
            ax.set_xticks(ticks,[str(x) for x in ticks])
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(frameon=False)
    save(fig,'count_cdf')

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    bands=('share_0to1','share_1to2','share_2to3','share_over3')
    band_names=('0-1 sigma','1-2 sigma','2-3 sigma','Beyond 3 sigma')
    palette=('#3c8d76','#7dbb9b','#e5bd66','#cb6d4e')
    for ax,split in zip(axes,('train','val')):
        bottom=np.zeros(2)
        for key,name,color in zip(bands,band_names,palette):
            values=np.array([100*s['statistics'][split][g]['means'][key] for g in groups])
            ax.bar(np.arange(2),values,bottom=bottom,label=name,color=color,width=.6)
            for i,value in enumerate(values):
                if value>=5:
                    ax.text(i,bottom[i]+value/2,f'{value:.1f}%',ha='center',va='center',fontsize=10)
            bottom+=values
        ax.set(xticks=(0,1),xticklabels=('Missed FG','Detected FG'),ylabel='Mean evidence share per voxel (%)',
               title=split.capitalize(),ylim=(0,100))
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,loc='upper left',bbox_to_anchor=(1,1))
    save(fig,'radial_evidence')

    with (RUN/'conditional_miss_rates.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,split in zip(axes,('train','val')):
        selected=[r for r in rows if r['split']==split]
        values=[100*float(r['miss_fraction']) if r['miss_fraction'] else np.nan for r in selected]
        ax.bar(np.arange(len(values)),values,color='#527fa4')
        ax.set(xticks=np.arange(len(values)),xticklabels=[r['within_2sigma_count'] for r in selected],
               xlabel='Number of Gaussians within 2 sigma',ylabel='Miss rate among GT foreground (%)',
               title=split.capitalize(),ylim=(0,105))
        for i,value in enumerate(values):
            if np.isfinite(value):
                label='<0.1' if 0<value<.1 else f'{value:.1f}'
                ax.text(i,value+1,label,ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    save(fig,'conditional_miss')


if __name__=='__main__':
    main()
