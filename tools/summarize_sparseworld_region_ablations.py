#!/usr/bin/env python3
"""Scene paired bootstrap, figures, and machine-readable results for A1-A4."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.sparseworld_region_utils import ROOT,write_json,metric_from_hist,percentile_interval
from tools.sparseworld_region_ablation_common import OUT,REPORT,GROUPS,records

REPEATS=2000

def weights(n,seed=20261008):
    return np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=REPEATS).astype(np.float64)

def ratio(c):
    return np.divide(c[...,1],c[...,0],out=np.full(c.shape[:-1],np.nan),where=c[...,0]>0)*100

def ci_rate(counts,w):
    c=counts.sum(0);boot=np.tensordot(w,counts,axes=(1,0))
    return dict(value=ratio(c),ci=percentile_interval(ratio(boot)),n=c[...,0],scenes=(counts[...,0]>0).sum(0))

def diff_rate(a,b,w):
    return dict(value=ratio(a.sum(0))-ratio(b.sum(0)),ci=percentile_interval(ratio(np.tensordot(w,a,axes=(1,0)))-ratio(np.tensordot(w,b,axes=(1,0)))))

def state_rate(c,w):
    total=c.sum(-1);fc=c[...,2]+c[...,3];pc=c[...,1]+c[...,3]
    f=np.stack([total,fc],-1);p=np.stack([total,pc],-1)
    return dict(F=ci_rate(f,w),P=ci_rate(p,w),P_minus_F=diff_rate(p,f,w))

def model_summary(hist,counts,base,w):
    metrics=metric_from_hist(hist.sum(0));bm=metric_from_hist(np.tensordot(w,hist,axes=(1,0)))
    return dict(metrics={k:dict(value=metrics[k],ci=percentile_interval(bm[k])) for k in ('mIoU','OccIoU')},
                groups={name:ci_rate(counts[:,:,g,:2],w) for g,name in enumerate(GROUPS)},
                delta_vs_P={name:diff_rate(counts[:,:,g,:2],base[:,:,g,:2],w) for g,name in enumerate(GROUPS)})

def main():
    recs=records();assert len(recs)==150
    split=json.loads((ROOT/'scene_split.json').read_text());hold=np.array([r['scene'] in split['holdout'] for r in recs])
    data={};details=[]
    for r in recs:
        with np.load(OUT/'val'/(r['token']+'.npz')) as d:
            for k,name in enumerate(d['names']):
                if '_static_' in name:continue  # A4 uses branch-effective ROI counts from JSON.
                data.setdefault(str(name),[[],[]]);data[str(name)][0].append(d['hist'][k]);data[str(name)][1].append(d['counts'][k])
        details.append(json.loads((OUT/'val'/(r['token']+'.json')).read_text()))
    data={k:tuple(np.asarray(x) for x in v) for k,v in data.items()}
    assert all(len(v[0])==150 for v in data.values())
    result=dict(samples=150,scenes=150,holdout_scenes=int(hold.sum()),bootstrap=REPEATS,groups=GROUPS,models={},holdout={})
    for domain,sel in [('models',np.ones(150,bool)),('holdout',hold)]:
        w=weights(int(sel.sum()))
        for name,(hist,counts) in data.items():result[domain][name]=model_summary(hist[sel],counts[sel],data['P'][1][sel],w)
    # Explicit same-renderer F-P contrasts, rather than mixing a modified F with native P.
    paired={}
    for mode in ('native','equal_count','equal_spatial','no_padding','no_group_filter','threshold_low','threshold_high'):
        f='F' if mode=='native' else 'F_'+mode;p='P' if mode in ('native','equal_count') else 'P_'+mode
        paired[mode]={g:diff_rate(data[p][1][:,:,k,:2],data[f][1][:,:,k,:2],weights(150)) for k,g in enumerate(GROUPS)}
    result['paired_renderer_advantage']=paired
    cov={}
    for a in ('F','P'):
        c=np.array([[d['diagnostics'][f'{a}_coverage_{h}'] for h in (2,4,6)] for d in details])
        cov[a]={g:ci_rate(c[:,:,k],weights(150)) for k,g in enumerate(GROUPS)}
    result['raw_point_coverage']=cov
    result['diagnostics']={key:dict(mean=float(np.mean([d['diagnostics'][key] for d in details])),
             min=float(np.min([d['diagnostics'][key] for d in details])),max=float(np.max([d['diagnostics'][key] for d in details])))
             for key in details[0]['diagnostics'] if '_coverage_' not in key}
    intervention={}
    for roi in ('future_only_static','current_near_static'):
        intervention[roi]={}
        for branch in ('F','P'):
            for domain,dom in [('all',np.ones(150,bool)),('holdout',hold)]:
                selected=[d['interventions'][roi] for d,keep in zip(details,dom) if keep and d['interventions'][roi]['available'] and d['interventions'][roi][branch+'_mask_pixels']>0]
                n=len(selected);w=weights(n)
                counts={mode:np.array([s['effects'][branch+'_'+mode][::-1] for s in selected]) for mode in ('baseline','targeted','random1','random2')}
                random=(counts['random1']+counts['random2'])/2
                intervention[roi][branch+'_'+domain]=dict(scenes=n,available_scenes=sum(d['interventions'][roi]['available'] for d,keep in zip(details,dom) if keep),
                    mask_fraction_mean=np.mean([s[branch+'_mask_fraction'] for s in selected]),
                    baseline=ci_rate(counts['baseline'],w),targeted=ci_rate(counts['targeted'],w),random=ci_rate(random,w),
                    targeted_minus_baseline=diff_rate(counts['targeted'],counts['baseline'],w),
                    random_minus_baseline=diff_rate(random,counts['baseline'],w),targeted_minus_random=diff_rate(counts['targeted'],random,w))
    result['interventions']=intervention
    adaptation={}
    for p in sorted((OUT/'adaptation').glob('*.npz')):
        with np.load(p) as d:
            assert d['tokens'].tolist()==[r['token'] for r in recs]
            adaptation[p.stem]=model_summary(d['hist'],d['counts'],data['P'][1],weights(150))
    result['adaptation']=adaptation
    budget_files=list((OUT/'budget_followup').glob('*.npz'))
    if len(budget_files)==150:
        budget={}
        for r in recs:
            with np.load(OUT/'budget_followup'/(r['token']+'.npz')) as d:
                for k,name in enumerate(d['names']):
                    budget.setdefault(str(name),[]).append(d['counts'][k])
        budget={k:np.array(v) for k,v in budget.items()}
        result['budget_followup']={}
        for name in budget:
            if not name.startswith('F_'):continue
            p='P_'+name[2:]
            result['budget_followup'][name[2:]]={g:dict(F=ci_rate(budget[name][:,k,:2],weights(150)),
                P=ci_rate(budget[p][:,k,:2],weights(150)),P_minus_F=diff_rate(budget[p][:,k,:2],budget[name][:,k,:2],weights(150))) for k,g in enumerate(GROUPS)}
    # 512 samples are aggregated into scene clusters before bootstrap.
    original=json.loads((ROOT/'manifest.json').read_text())['samples'];scenes=sorted({r['scene'] for r in original});si={s:i for i,s in enumerate(scenes)}
    sums={}
    for r in original:
        with np.load(OUT/'observation'/(r['token']+'.npz')) as d:
            for k in d.files:
                if k=='version':continue
                if k not in sums:sums[k]=np.zeros((150,*d[k].shape),np.int64)
                sums[k][si[r['scene']]]+=d[k]
    w=weights(150);obs={}
    for k in ('distance','frustum','cross'):
        obs[k]=state_rate(sums[k],w)
    dc=sums['distance']
    obs['distance_regions']={
        'current_near10':state_rate(dc[:,:,:,0].sum(3),w),
        'target_near10':state_rate(dc[:,:,:,:,0].sum(3),w),
        'current_near_target_far20':state_rate(dc[:,:,:,0,2:].sum(3),w)}
    q=sums['quality'];obs['quality']=state_rate(q.sum((5,6)),w)
    # Class x target-distance standardized difference in P-F between more vs zero depth-consistent views.
    # Common support and fixed pooled weights, re-estimated ratios within each scene bootstrap draw.
    standard=[]
    for a in range(2):
        for j in range(3):
            v=q[:,j,a,1]  # scenes,bins,class,distance,state
            low=v[:,0];high=v[:,1:].sum(1)
            lc=low.sum(0);hc=high.sum(0);support=(lc.sum(-1)>=100)&(hc.sum(-1)>=100)
            ww=(lc.sum(-1)+hc.sum(-1))*support;ww=ww/ww.sum()
            def adv(c):return np.divide(c[...,1]-c[...,2],c.sum(-1),out=np.zeros(c.shape[:-1],float),where=c.sum(-1)>0)*100
            point=((adv(hc)-adv(lc))*ww).sum()
            boot=((adv(np.tensordot(w,high,axes=(1,0)))-adv(np.tensordot(w,low,axes=(1,0))))*ww).sum((1,2))
            standard.append(dict(branch='FP'[a],horizon=j+1,increment=float(point),ci=np.percentile(boot,[2.5,97.5]),strata=int(support.sum())))
    obs['depth_supported_vs_zero_standardized']=standard
    result['observation']=obs
    np.savez_compressed(OUT/'scene_observation_counts.npz',**sums,scenes=np.array(scenes))
    write_json(OUT/'summary.json',result)
    plot(result)
    print('Saved summary and figures',flush=True)

def plot(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    out=REPORT/'figures';out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    arms=['F','H','P','M','HP_full','HP_equal_points']
    for ax,key,title in zip(axs,['mIoU','static_future_only'],['整体 mIoU（3 秒）','未来视锥独有静态区域：准确率（3 秒）']):
        vals=[s['models'][a]['metrics']['mIoU']['value'][2] if key=='mIoU' else s['models'][a]['groups'][key]['value'][2] for a in arms]
        bars=ax.bar(range(len(arms)),vals,color=['#687d96','#88aace','#d37761','#9e84b7','#69a694','#9bb889']);ax.set_title(title);ax.set_ylabel('%');ax.set_xticks(range(len(arms)),['预测 F','历史 H','未来 P','混合 M','融合\n1440 query','融合\n720 query']);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9);ax.set_ylim(0,max(vals)*1.18)
    save(fig,'a1_windows')
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    matrix=np.array(s['observation']['distance']['P_minus_F']['value'],dtype=float)[2].copy()
    support=np.array(s['observation']['distance']['F']['scenes'])[2]
    matrix[support<20]=np.nan
    for ax,m,title in zip(axs,matrix,['全部占据体素','静态占据体素']):
        im=ax.imshow(m,cmap='RdBu_r',vmin=-20,vmax=20,origin='lower');ax.set_title(title+'｜P−F，3 秒')
        ax.set_xticks(range(4),['0-10','10-20','20-30','30+']);ax.set_yticks(range(4),['0-10','10-20','20-30','30+'])
        ax.set_xlabel('到目标时刻车辆的距离（m）');ax.set_ylabel('到当前车辆的距离（m）')
        for i in range(4):
            for j in range(4):
                if np.isfinite(m[i,j]):ax.text(j,i,f'{m[i,j]:+.1f}',ha='center',va='center',fontsize=10,color='white' if abs(m[i,j])>12 else '#24374a')
                else:ax.text(j,i,'不足 20 场景',ha='center',va='center',fontsize=8,color='gray')
    fig.colorbar(im,ax=axs,label='P−F（百分点）：蓝色 F 更准，红色 P 更准');save(fig,'a2_distance')
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');names=list(s['paired_renderer_advantage'])
    vals=[s['paired_renderer_advantage'][a]['static_future_only']['value'][2] for a in names]
    intervals=np.array([s['paired_renderer_advantage'][a]['static_future_only']['ci'][2] for a in names])
    ax.errorbar(vals,np.arange(len(names)),xerr=np.stack([np.array(vals)-intervals[:,0],intervals[:,1]-np.array(vals)]),fmt='o',color='#467b9c')
    labels={'native':'原版设置','equal_count':'等 query 数（720）','equal_spatial':'等空间配额（4 m 块）','no_padding':'关闭空间填补','no_group_filter':'关闭组几何过滤','threshold_low':'统一低阈值（0.2）','threshold_high':'统一高阈值（0.5）'}
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(len(names)),[labels[a] for a in names]);ax.invert_yaxis();ax.set_xlabel('← F 更准         P−F（百分点）         P 更准 →');ax.set_title('未来视锥独有静态区域｜3 秒差值与 95% 区间');ax.grid(axis='x',alpha=.15)
    save(fig,'a3_controls')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');names=[];values=[];cis=[]
    for roi in s['interventions']:
        for b in ('F','P'):
            d=s['interventions'][roi][b+'_all']['targeted_minus_random'];names.append({'future_only_static':'未来视锥独有静态','current_near_static':'当前 10 m 内静态'}[roi]+' / '+b);values.append(d['value']);cis.append(d['ci'])
    values=np.array(values);cis=np.array(cis);ax.errorbar(values,range(4),xerr=np.stack([values-cis[:,0],cis[:,1]-values]),fmt='o',color='#9a654a')
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(4),names);ax.invert_yaxis();ax.set_xlabel('定向 − 同面积随机遮挡（百分点）；越负，指定内容损失越大');ax.set_title('局部输入干预｜各分支有效场景，95% 区间');ax.grid(axis='x',alpha=.15)
    save(fig,'a4_interventions')

if __name__=='__main__':main()
