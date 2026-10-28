"""Export the fixed semantic-routing diagnostic using the saved paired counts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.pmg_world_native_adapt.gate_optimization_report import CONTROLS
from tools.pmg_world_native_adapt.report import regional

ROOT=Path('analysis/pmg_world_semantic_route_20261026/S1')


def main():
    sources=(*CONTROLS,('S1',ROOT,'epoch_01'))
    metrics={name:json.loads((folder/f'{prefix}_metrics.json').read_text()) for name,folder,prefix in sources}
    stats=json.loads((ROOT/'comparison_statistics.json').read_text())['comparisons']['S1_minus_T-A0']
    route=json.loads((ROOT/'route_statistics.json').read_text())['rows']
    colors={'T-A0':'#666666','N-native':'#df8a24','N-half':'#287ba5','G1':'#aa3964','S1':'#218765'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    t=np.arange(1,4)
    for name,metric in metrics.items():
        axes[0,0].plot(t,regional(metric,'fast','occupied_semantic_accuracy'),'o-',color=colors[name],label=name)
    axes[0,0].set(title='Fast-target semantic accuracy',ylabel='Accuracy (%)',xticks=t)
    axes[0,0].legend(frameon=False,ncol=2)
    for axis,region,key,title in ((axes[0,1],'F_only','F_only','F-only accuracy change vs T-A0'),
                                 (axes[1,0],'static_classes','static_classes_accuracy','Static-class GT accuracy change vs T-A0')):
        base=regional(metrics['T-A0'],region,'occupied_semantic_accuracy')
        for name in ('N-native','N-half','G1','S1'):
            delta=regional(metrics[name],region,'occupied_semantic_accuracy')-base
            if name=='S1':
                ci=np.asarray(stats[key]['per_horizon_delta_95ci_pp'])
                axis.errorbar(t,delta,yerr=np.stack((delta-ci[:,0],ci[:,1]-delta)),fmt='o-',capsize=3,color=colors[name])
            else:
                axis.plot(t,delta,'o:',color=colors[name])
        axis.axhline(0,color='#777777',linewidth=.8)
        axis.set(title=title,ylabel='Change (percentage points)',xticks=t)
    axes[1,1].plot([r['seconds'] for r in route],[r['fraction_selected']*100 for r in route],'o-',color=colors['S1'])
    axes[1,1].set(title='Children routed to G1',ylabel='Selected children (%)')
    for axis in axes.flat:
        axis.set_xlabel('Forecast horizon (s)')
        axis.grid(alpha=.15)
    fig.suptitle('S1: route movable semantic classes to G1, retain other T-A0 Gaussians',fontsize=13)
    fig.supxlabel('1024 samples / 150 scenes. Error bars: S1 paired scene 95% intervals. Predicted classes are not GT motion; F-only is not static.',fontsize=8)
    fig.savefig(ROOT/'comparison_overview.png',dpi=180)
    fig.savefig(ROOT/'comparison_overview.pdf')
    plt.close(fig)
    print(ROOT/'comparison_overview.png')


if __name__=='__main__':
    main()
