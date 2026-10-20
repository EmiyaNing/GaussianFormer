"""Freeze training-scene policies, including a stronger classwise local control."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.empty_decision_optimization import RUN, BUDGETS, read, write_json, file_sha256
from tools.eval_empty_decision_optimization import (
    select_points, EDGES, BINS, histogram_curve_counts, histogram_counts,
    metrics, region_metrics, chosen_index)


def main():
    if (RUN/'operating_points.json').exists():
        cached = read(RUN/'operating_points.json')
        assert cached['histogram_sha256'] == file_sha256(RUN/'tuning/histograms.npz')
        for name,meta in read(RUN/'tuning/summary.json')['models'].items():
            assert file_sha256(ROOT/meta['path']) == cached['model_sha256'][name]
        print('Reusing immutable training-selected operating points',flush=True)
        return
    result = select_points()
    with np.load(RUN/'tuning/histograms.npz') as z:
        h,fixed,native = z['B3_seed928'],z['fixed'],z['native']
    allowed = np.arange(1,BINS+2)
    ccs = histogram_curve_counts(h,fixed,allowed)
    diag = np.diagonal(ccs[:,0],axis1=-2,axis2=-1)[:,:17]
    unions = ccs[:,0].sum(-2)[:,:17]+ccs[:,0].sum(-1)[:,:17]-diag
    ious = np.divide(diag,unions,out=np.zeros_like(diag,dtype=float),where=unions>0)
    out_fp = ccs[:,1,17,:17]
    n_empty = native[1,17].sum()
    proposals = []
    for lam in np.r_[0.,np.logspace(-5,4,181)]:
        proposals.append(allowed[(ious/17-lam*out_fp/max(n_empty,1)).argmax(0)])
    unique = np.unique(proposals,axis=0)
    candidate_counts = np.stack([histogram_counts(h,fixed,c)[0] for c in unique])
    camera = metrics(candidate_counts[:,0])['mIoU']
    risk = metrics(candidate_counts[:,1])['empty_FPR']/100
    for budget in BUDGETS:
        i,feasible = chosen_index(camera,risk,budget,result['reference_out_fpr'])
        tag = 'unconstrained' if budget is None else f'eps{budget:g}'
        result['operating_points'].append(dict(id=f'B3_class_{tag}_seed928',family='B3_class',seed=928,
            budget=budget,source='B3_seed928',cuts=unique[i].tolist(),threshold=EDGES[unique[i]-1].tolist(),
            feasible=feasible,tuning=region_metrics(candidate_counts[i])))
    result['additional_control'] = dict(name='B3_class',proposals=len(unique),
        definition='ordinary binary local NLL network plus classwise score thresholds; same supervision and risk budgets',
        source_sha256=file_sha256(__file__))
    write_json(RUN/'operating_points.json',result)
    print('Final frozen operating points',len(result['operating_points']),flush=True)


if __name__ == '__main__':
    main()
