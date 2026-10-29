"""Standalone publication-exportable figures for the full diagnostic census."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'work_dirs/gaussian_scale_supervision_full'
CONDITIONS=('baseline','scale_cap30','opacity_mass_matched','scale_cap30_mass_restored')


def main():
    summary=json.loads((RUN/'summary.json').read_text());assert summary['status']=='complete'
    directory=ROOT/'analysis/figures/gaussian_scale_supervision_full';directory.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(directory/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    groups=['pure_foreground','pure_empty_no_fp_owner','pure_empty_fp_owner']
    labels=['Pure FG','Pure empty: no FP ownership','Pure empty: FP owner']
    colors=['#3579b5','#e68e35'];fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,key,title in zip(axes,['all_axes_ge_039','all_scale_sigmoid_gradients_zero'],
        ['All three scales at least 0.39 m','All three FP32 scale-sigmoid derivatives exactly zero']):
        for i,split in enumerate(('train','val')):
            values=[100*summary['statistics'][split]['groups'][g]['flag_fractions'][key] for g in groups]
            ax.bar(np.arange(3)+(i-.5)*.36,values,.36,color=colors[i],label=['Train','Validation'][i])
        ax.set(xticks=np.arange(3),xticklabels=labels,ylabel='Gaussians in group (%)',title=title,ylim=(0,105))
        ax.tick_params(axis='x',labelrotation=15);ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False);save(fig,'saturation')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    subgroup=['pure_empty_fp_owner_visible_empty_center','pure_empty_fp_owner_outside_camera_fg_center',
              'pure_empty_fp_owner_outside_camera_empty_center']
    names=['Visible empty center','Outside-mask FG center','Outside-mask empty center']
    left=np.zeros(2)
    for key,label,color in zip(subgroup,names,['#3579b5','#e68e35','#99b5c9']):
        values=np.array([100*summary['statistics'][s]['groups'][key]['gaussians']/
            summary['statistics'][s]['groups']['pure_empty_fp_owner']['gaussians'] for s in ('train','val')])
        axes[0].barh(np.arange(2),values,left=left,label=label,color=color);left+=values
    axes[0].set(yticks=np.arange(2),yticklabels=['Train','Validation'],xlim=(0,100),
                xlabel='Within camera-pure-empty FP-owner Gaussians (%)',title='Where is the Gaussian center?')
    axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=9)
    keys=['pure_empty_fp_owner',subgroup[0],'pure_empty_fp_owner_all_support_empty']
    labels=['Camera support\npure empty','Also visible\nempty center','Full-grid support\nalso pure empty']
    for i,split in enumerate(('train','val')):
        values=[100*summary['statistics'][split]['groups'][g]['flag_fractions']['all_scale_sigmoid_gradients_zero'] for g in keys]
        axes[1].bar(np.arange(3)+(i-.5)*.36,values,.36,color=colors[i],label=['Train','Validation'][i])
    axes[1].set(xticks=np.arange(3),xticklabels=labels,ylim=(0,100),ylabel='Within each FP-owner group (%)',
                title='All three scale derivatives exactly zero')
    axes[1].legend(frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    save(fig,'center_support')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    for ax,key,title in zip(axes,['mIoU','miss_rate','false_positive_rate'],['Semantic mIoU','Foreground miss rate','Empty false-positive rate']):
        for i,split in enumerate(('train','val')):
            values=[100*summary['statistics'][split]['conditions'][c]['native'][key] for c in CONDITIONS]
            ax.bar(np.arange(4)+(i-.5)*.36,values,.36,color=colors[i],label=['Train','Validation'][i])
        ax.set(xticks=np.arange(4),xticklabels=['B0','S: scale cap','O: mass match','R: mass restore'],ylabel='Percent',title=title)
        ax.tick_params(axis='x',labelrotation=20);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(frameon=False);save(fig,'interventions')
    with (RUN/'decision_curves.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    palette=['#222222','#3579b5','#e68e35','#6b9e4a']
    for ax,split in zip(axes,['train','val']):
        for name,label,color in zip(CONDITIONS,['B0','S','O','R'],palette):
            selected=[r for r in rows if r['split']==split and r['condition']==name]
            x=np.array([float(r['false_positive_rate'])*100 for r in selected])
            y=np.array([float(r['miss_rate'])*100 for r in selected])
            ax.plot(x,y,label=label,color=color)
            native=summary['statistics'][split]['conditions'][name]['native']
            ax.scatter([native['false_positive_rate']*100],[native['miss_rate']*100],color=color,s=25)
        budget=summary['statistics'][split]['conditions']['baseline']['native']['false_positive_rate']*100
        ax.axvline(budget,color='gray',linestyle=':',linewidth=1)
        ax.set(xlabel='False-positive rate (%)',ylabel='Miss rate (%)',title=split.capitalize(),xlim=(0,max(12,budget*2)))
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
    axes[0].legend(frameon=False);save(fig,'decision_curves')


if __name__=='__main__':main()
