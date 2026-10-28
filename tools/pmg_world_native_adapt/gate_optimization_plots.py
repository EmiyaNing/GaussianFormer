"""Export the completed G1 diagnostic and G2 probe as standalone figures."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.pmg_world_native_adapt.gate_optimization_report import CONTROLS
from tools.pmg_world_native_adapt.report import regional

ROOT = Path('analysis/pmg_world_gate_optimization_20261026')


def main():
    sources = (*CONTROLS, ('G1-unit', ROOT/'G1_unit', 'epoch_01'), ('G2', ROOT/'G2', 'epoch_01'))
    metrics = {name: json.loads((folder/f'{prefix}_metrics.json').read_text()) for name, folder, prefix in sources}
    stats = json.loads((ROOT/'G2/comparison_statistics.json').read_text())
    comparison = stats['comparisons']['G2_minus_T-A0']
    adapter = json.loads((ROOT/'G2/adapter_statistics.json').read_text())['rows']
    colors = {'T-A0':'#666666', 'N-native':'#df8a24', 'N-half':'#287ba5',
              'G1':'#aa3964', 'G1-unit':'#a3a3a3', 'G2':'#218765'}
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 3, figsize=(14,8), constrained_layout=True)
    horizons = np.arange(1,4)
    for name, metric in metrics.items():
        style = 'o:' if name=='G1-unit' else 'o-'
        axes[0,0].plot(np.arange(7)*.5, [h['miou'] for h in metric['horizons']], style, color=colors[name], label=name)
        for axis, field, title in ((axes[0,1], 'occupied_semantic_accuracy','Fast-target semantic accuracy'),
                                   (axes[0,2], 'occupied_recall','Fast-target occupancy recall')):
            axis.plot(horizons,regional(metric,'fast',field),style,color=colors[name])
            axis.set(title=title, ylabel='Percent', xticks=horizons)
    axes[0,0].set(title='Global mIoU',ylabel='mIoU (%)')
    axes[0,0].legend(frameon=False,ncol=2)
    for axis,region,key,title in ((axes[1,0],'F_only','F_only','F-only accuracy change vs T-A0'),
                                  (axes[1,1],'static_classes','static_classes_accuracy','Static-class GT accuracy change')):
        base=regional(metrics['T-A0'],region,'occupied_semantic_accuracy')
        for name in ('N-native','N-half','G1','G1-unit','G2'):
            delta=regional(metrics[name],region,'occupied_semantic_accuracy')-base
            if name=='G2':
                ci=np.array(comparison[key]['per_horizon_delta_95ci_pp'])
                axis.errorbar(horizons,delta,yerr=np.stack((delta-ci[:,0],ci[:,1]-delta)),fmt='o-',capsize=3,color=colors[name])
            else:
                axis.plot(horizons,delta,'o:',color=colors[name])
        axis.axhline(0,color='#666666',linewidth=.7)
        axis.set(title=title,ylabel='Change (percentage points)',xticks=horizons)
    axes[1,2].plot([r['seconds'] for r in adapter], [r['ratio_mean']*100 for r in adapter], 'o-',label='Mean',color=colors['G2'])
    axes[1,2].plot([r['seconds'] for r in adapter], [r['ratio_max']*100 for r in adapter], 'o--',label='Maximum',color=colors['G2'])
    axes[1,2].axhline(10,color='#777777',linewidth=.7,label='Fixed bound')
    axes[1,2].set(title='G2 correction / native-increment norm',ylabel='Percent',ylim=(0,10.5))
    axes[1,2].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel('Forecast horizon (s)')
        axis.grid(alpha=.15)
    fig.suptitle('G1 failure diagnostic and G2 bounded state adapter',fontsize=14)
    fig.supxlabel('1024 development samples / 150 scenes. Error bars: G2 paired scene 95% intervals; F-only is not a static-region definition.',fontsize=9)
    fig.savefig(ROOT/'comparison_overview.png',dpi=180)
    fig.savefig(ROOT/'comparison_overview.pdf')
    plt.close(fig)
    print(ROOT/'comparison_overview.png')


if __name__=='__main__':
    main()
