"""Composition matching must not turn a mixture shift into a score-factor shift."""
import numpy as np
from tools.report_gaussian_support_regions import compare_factors,FIELDS


def test_class_distance_matching_removes_pure_mixture_change():
    shape=(2,5,5,18,4)
    totals={}
    for split,counts in [('train',[90,10]),('val',[10,90])]:
        n=np.zeros(shape,np.int64)
        fields=np.zeros((*shape,len(FIELDS)),np.float64)
        for c,count in enumerate(counts):
            n[0,2,0,c,0]=count
            # log geometry + log opacity + log amplitude + log concentration - log empty
            means=np.zeros(len(FIELDS));means[0]=float(c);means[5]=float(c);means[6]=float(c)
            fields[0,2,0,c,0]=means*count
        totals[split]=dict(field_counts=n,strata_counts=n,field_sums=fields)
    result=compare_factors(totals,0,[2],[0,1,2])
    assert result['val']['means']['log_geometry']>result['train']['means']['log_geometry']
    assert result['matched']['val_minus_train_log_factors']['log_geometry']==0
    assert result['matched']['identity_residual']==0
    assert result['matched']['train']['covered_foreground_fraction']==1
