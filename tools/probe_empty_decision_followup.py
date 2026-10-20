"""Separate support-information probes from deployable decision-head gains."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,log_loss
from threadpoolctl import threadpool_limits
from tools.empty_decision_followup import RUN,read,write_json,file_sha256,samples
from tools.fit_empty_decision_followup import targets


def features(data,cols):
    return np.c_[data['X'][:,:cols],np.eye(17,dtype=np.float32)[data['fg']]]


def fit():
    dest=RUN/'probes';dest.mkdir(exist_ok=True);data=samples('fit');tune=samples('tune');records={}
    for scope in ['camera','all']:
        keep=data['region']==0 if scope=='camera' else np.ones(len(data['gt']),bool)
        for group,cols in [('base',8),('S',9),('support',12)]:
            name=scope+'_'+group;path=dest/(name+'.joblib')
            if path.exists():records[name]=read(path.with_suffix('.json'));continue
            network=HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,min_samples_leaf=200,
                learning_rate=.06,l2_regularization=1.,early_stopping=False,random_state=928)
            weight=data['weight'][keep];weight=weight/weight.mean()
            with threadpool_limits(8):network.fit(features(data,cols)[keep],targets(data)[keep],sample_weight=weight)
            joblib.dump(network,path)
            result=dict(scope=scope,group=group,cols=cols,fit_rows=int(keep.sum()),role='diagnostic only',
                        source_sha256=file_sha256(__file__))
            rows={}
            with threadpool_limits(8):p=network.predict_proba(features(tune,cols))
            for r,region in [(0,'camera'),(1,'complement')]:
                m=tune['region']==r;w=tune['weight'][m];y=targets(tune)[m]
                rows[region]=dict(q_AP=float(average_precision_score(y==0,p[m,0],sample_weight=w)),
                    NLL=float(log_loss(y,p[m],sample_weight=w,labels=[0,1,2])))
            result['tuning']=rows;write_json(path.with_suffix('.json'),result);records[name]=result
            print('support probe',name,rows,flush=True)
    write_json(RUN/'support_probes_training.json',records)


def evaluate():
    with np.load(RUN/'validation_samples.npz') as z:data={k:z[k] for k in z.files}
    results={}
    for name,record in read(RUN/'support_probes_training.json').items():
        network=joblib.load(RUN/'probes'/(name+'.joblib'))
        with threadpool_limits(8):p=network.predict_proba(features(data,record['cols']))
        rows={};y=targets(data)
        for r,region in [(0,'camera'),(1,'complement')]:
            mask=data['region']==r;w=data['weight'][mask]
            rows[region]=dict(q_AP=float(average_precision_score(y[mask]==0,p[mask,0],sample_weight=w)),
                q_prevalence=float(np.average(y[mask]==0,weights=w)),
                NLL=float(log_loss(y[mask],p[mask],sample_weight=w,labels=[0,1,2])))
        results[name]={**record,'validation':rows}
    write_json(RUN/'support_probes_validation.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','eval']);a=p.parse_args()
    fit() if a.stage=='fit' else evaluate()
