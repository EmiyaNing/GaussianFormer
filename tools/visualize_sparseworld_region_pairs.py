#!/usr/bin/env python3
"""Publication-style paired-region figures and reproducible, non-extreme cases."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
from PIL import Image

from tools.sparseworld_region_utils import ROOT, CLASS_NAMES, STATE_NAMES, write_json, states

COLORS=['#8555a4','#de8b35','#367eab','#b9c1c7']
STATE_ZH=['双方错误','仅感知器正确','仅预测器正确','双方正确']


def save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white')
    if path.stem!='case':
        fig.savefig(path.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--report-dir',type=Path,default=Path('analysis/hypotheses/prediction_perception_regions'))
    args=parser.parse_args()
    root=args.root
    figroot=args.report_dir/'figures'
    figroot.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
                          'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    summary=json.loads((root/'summary.json').read_text())['splits']['all']
    sem=np.asarray(summary['states']['proportions']['value'])
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for domain,ax in zip([1,2,0],axes):
        values=sem[:,0,domain,0]
        left=np.zeros(3)
        for state in [3,2,1,0]:
            ax.barh(np.arange(3),values[:,state],left=left,color=COLORS[state],label=STATE_ZH[state])
            for h in range(3):
                if values[h,state]>5:
                    ax.text(left[h]+values[h,state]/2,h,f'{values[h,state]:.1f}',ha='center',va='center',fontsize=9)
            left+=values[:,state]
        ax.set_title(['全部合法体素','GT 占据体素','GT 空闲体素'][domain])
        ax.set_xlabel('精确分类状态比例（%）')
        ax.set_yticks(np.arange(3),['1 秒','2 秒','3 秒'])
        ax.set_xlim(0,100)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,-.4),ncol=4,frameon=False)
    fig.suptitle('同一位置的配对比较：大量正确空闲不代表物体已被识别',fontsize=14,y=1.03)
    save(fig,figroot/'01_four_states.png')
    region=pd.read_parquet(root/'region_stats.parquet')
    records=json.loads((root/'manifest.json').read_text())['samples']
    lookup=np.asarray([r['scene'] for r in records])
    fig,axes=plt.subplots(3,3,figsize=(12,12),sharex=True,sharey=True)
    for h in range(1,4):
        rows=region[region.horizon==h]
        sums=rows.groupby('block_id')[['occupied','both_wrong','only_perception','only_forecast']].sum().reindex(range(1600),fill_value=0)
        supported=rows[rows.occupied>0].copy()
        supported['scene']=lookup[supported.sample_position.to_numpy()]
        ns=supported.groupby('block_id').scene.nunique().reindex(range(1600),fill_value=0).to_numpy()
        denominator=sums.occupied.to_numpy()
        advantage=np.divide((sums.only_perception-sums.only_forecast).to_numpy()*100,denominator,
                   out=np.full(1600,np.nan),where=denominator>0)
        wrong=np.divide(sums.both_wrong.to_numpy()*100,denominator,out=np.full(1600,np.nan),where=denominator>0)
        advantage[ns<20]=np.nan
        wrong[ns<20]=np.nan
        for col,values,title,cmap,lo,hi in [(0,advantage,'P − F 准确率（百分点）','RdBu_r',-20,20),
                         (1,wrong,'双方错误（%）','Purples',0,100),
                         (2,ns.astype(float),'有 GT 占据的场景数','viridis',0,150)]:
            image=axes[h-1,col].imshow(values.reshape(40,40),origin='lower',extent=[-40,40,-40,40],
                                       cmap=cmap,vmin=lo,vmax=hi)
            axes[h-1,col].set_title(f'{h} 秒 · {title}')
            axes[h-1,col].set_ylabel('目标 ego x（m）')
            axes[h-1,col].set_xlabel('目标 ego y（m）')
            axes[h-1,col].plot(0,0,'k+',ms=8)
            fig.colorbar(image,ax=axes[h-1,col],shrink=.8)
    fig.suptitle('2 m 空间块 · 优势、共同困难与支持量（不足 20 场景留白）',y=1.01,fontsize=14)
    save(fig,figroot/'02_spatial_maps.png')
    specs=[('distance',['0–10','10–20','20–30','≥30'],'距离（m）'),
           ('speed',['<0.5','0.5–2','≥2','未知'],'实际速度（m/s，仅可运动类别）'),
           ('boundary',['内部','边界带'],'GT 语义边界（0.4 m）'),
           ('camera',['mask=0','mask=1'],'目标时刻 camera 标注 mask'),
           ('frustum',['都不在','仅未来','仅历史','都在','未知'],'实际窗口视锥覆盖'),
           ('semantic_group',['可运动类','静态类','others'],'语义分组')]
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    palette=['#205c83','#af7034','#795287']
    for ax,(attr,labels,title) in zip(axes.flat,specs):
        info=summary['attributes'][attr]['advantage']
        values=np.asarray(info['value'])
        ci=np.asarray(info['ci95'])
        for h in range(3):
            x=np.arange(len(labels))+(h-1)*.16
            ax.errorbar(x,values[h],yerr=np.maximum(0,np.stack([values[h]-ci[h,:,0],ci[h,:,1]-values[h]])),
                        fmt='o',capsize=3,color=palette[h],label=f'{h+1} 秒')
        ax.axhline(0,color='#999999',lw=1)
        ax.set_xticks(np.arange(len(labels)),labels,rotation=12)
        ax.set_title(title)
        ax.set_ylabel('P − F 精确分类率（百分点）')
        ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(frameon=False)
    fig.suptitle('属性与区域优势 · 场景级 95% 区间（相关性，未作因果声明）',fontsize=14,y=1.02)
    fig.tight_layout()
    save(fig,figroot/'03_attribute_advantage.png')
    data=dict(np.load(root/'region_counts.npz'))
    cov=data['covariates'][:,:,:17].sum(axis=(0,4,5,6,7))
    total=cov.sum(-1)
    values=np.divide((cov[...,1]-cov[...,2])*100,total,out=np.full_like(total,np.nan,dtype=float),where=total>0)
    support=(data['covariates'][:,:,:17].sum(axis=(4,5,6,7,8))>0).sum(0)
    values[support<20]=np.nan
    fig,axes=plt.subplots(1,3,figsize=(11,9),sharey=True)
    for h,ax in enumerate(axes):
        im=ax.imshow(values[h],aspect='auto',cmap='RdBu_r',vmin=-30,vmax=30)
        ax.set_xticks(range(4),['0–10','10–20','20–30','≥30'])
        ax.set_yticks(range(17),CLASS_NAMES[:17],fontsize=9)
        ax.set_xlabel('距离（m）')
        ax.set_title(f'{h+1} 秒')
    fig.colorbar(im,ax=axes.tolist(),shrink=.7,label='P − F 准确率（百分点）')
    fig.suptitle('同类别、同距离再比较：蓝色偏向 F，红色偏向 P',fontsize=14)
    save(fig,figroot/'04_class_distance.png')
    cases=[]
    used=set()
    for state,name in enumerate(STATE_NAMES):
        for h in range(1,4):
            candidates=region[(region.horizon==h)&(region.occupied>=20)].copy()
            candidates['fraction']=candidates[name]/candidates.occupied
            if state==2:
                candidates=candidates[candidates.only_forecast>candidates.only_perception]
            elif state==1:
                candidates=candidates[candidates.only_perception>candidates.only_forecast]
            else:
                candidates=candidates[candidates.fraction>=.5]
            candidates=candidates[candidates[name]>0]
            median=candidates.fraction.median()
            candidates['distance_to_median']=abs(candidates.fraction-median)
            candidates['hash']=[hashlib.sha256(f'20261008:{i}:{h}:{b}'.encode()).hexdigest()
                                for i,b in zip(candidates.sample_position,candidates.block_id)]
            candidates=candidates.sort_values(['distance_to_median','hash'])
            chosen=None
            for row in candidates.itertuples():
                rec=records[int(row.sample_position)]
                if rec['scene'] not in used:
                    chosen=row
                    used.add(rec['scene'])
                    break
            if chosen is None:
                continue
            record=records[int(chosen.sample_position)]
            case=dict(state=name,state_zh=STATE_ZH[state],sample_position=int(chosen.sample_position),
                      scene=record['scene'],token=record['token'],horizon=h,block_id=int(chosen.block_id),
                      x=float(chosen.x),y=float(chosen.y),fraction=float(chosen.fraction),
                      median_fraction=float(median),occupied=int(chosen.occupied),
                      selection='Eligible group median, hash tie break, distinct scenes; F/P cases require net advantage')
            case_dir=figroot/'cases'
            case_dir.mkdir(exist_ok=True)
            filename=f'{name}_{h}s.png'
            case['figure']='figures/cases/'+filename
            with np.load(root/'pairs'/(record['token']+'.npz')) as pair:
                g,f,p=pair['GT'][h-1],pair['F'][h-1],pair['P'][h-1]
                cx,cy=int((chosen.x+40)/.4),int((chosen.y+40)/.4)
                sx=slice(max(0,cx-10),min(200,cx+11))
                sy=slice(max(0,cy-10),min(200,cy+11))
                # Display identical GT-derived upper surface height for both models;
                # empty XY columns are shown at fixed z=0.4 m.
                occupied=g[sx,sy]<17
                height=15-np.argmax(occupied[...,::-1],axis=-1)
                height[~occupied.any(-1)]=3
                arrays=[np.take_along_axis(a[sx,sy],height[...,None],axis=-1)[...,0] for a in [g,f,p]]
                state_image=states(*arrays)
                fig=plt.figure(figsize=(13,8))
                fig.subplots_adjust(bottom=.20)
                grid=fig.add_gridspec(2,4,height_ratios=[1,1.1],hspace=.45)
                block_x=int(chosen.block_id)//40*5
                block_y=int(chosen.block_id)%40*5
                for k,(a,title) in enumerate(zip(arrays+[state_image],['GT 上表面','预测器 F（同高度）','感知器 P（同高度）','精确分类状态'])):
                    ax=fig.add_subplot(grid[0,k])
                    cmap=ListedColormap(COLORS) if k==3 else plt.get_cmap('tab20',18)
                    ax.imshow(a,origin='lower',cmap=cmap,vmin=0,vmax=3 if k==3 else 17)
                    ax.add_patch(Rectangle((block_y-sy.start-.5,block_x-sx.start-.5),5,5,
                                           fill=False,edgecolor='black',linewidth=1.4))
                    ax.set_title(title)
                    ax.set_xlabel('局部 y 网格')
                    if k==0:
                        ax.set_ylabel('局部 x 网格')
                target=record['targets'][h-1]
                point=np.asarray([chosen.x,chosen.y,0.4,1])
                q=pair['P_projection'][h-1,:6]@point
                wh=pair['P_wh'][h-1,:6]
                inview=(q[:,2]>.01)&(q[:,0]>=0)&(q[:,1]>=0)&(q[:,0]<q[:,2]*wh[:,0])&(q[:,1]<q[:,2]*wh[:,1])
                camera=int(np.flatnonzero(inview)[0]) if inview.any() else 0
                for k,(frame,title) in enumerate([(target['forecast_frames'][0],'F 当前帧'),
                                                (target['perception_frames'][0],'P 目标帧')]):
                    ax=fig.add_subplot(grid[1,k*2:(k+1)*2])
                    image=Image.open(frame['images'][camera]).convert('RGB')
                    image.thumbnail((900,500))
                    ax.imshow(image)
                    ax.set_title(title+' · '+record['camera_names'][camera])
                    ax.axis('off')
                fig.suptitle(f'{STATE_ZH[state]} · {h} 秒 · 块内占比 {chosen.fraction*100:.1f}% · '
                            f'中心 ({chosen.x:.0f}, {chosen.y:.0f}) m\n'
                            f'scene={record["scene"]}   sample={record["token"]}',fontsize=10)
                rates=np.asarray([chosen.both_wrong,chosen.only_perception,chosen.only_forecast,chosen.both_correct])/chosen.occupied*100
                classes=np.unique(np.concatenate([a.reshape(-1) for a in arrays]))
                fig.legend(handles=[Patch(color=plt.get_cmap('tab20',18)(int(c)),label=CLASS_NAMES[int(c)])
                                    for c in classes],loc='lower center',bbox_to_anchor=(.5,.105),
                           ncol=6,frameon=False,fontsize=8)
                fig.text(.5,.045,'黑框内全部高度：'+ ' / '.join(f'{label} {value:.1f}%' for label,value in zip(STATE_ZH,rates)),ha='center',fontsize=9)
                fig.text(.5,.01,'上图仅显示 GT 最上方占据表面；空闲列取固定高度。黑框为 2 m 统计块，周围图像提供场景背景。',ha='center',fontsize=9)
                fig.savefig(case_dir/filename,dpi=150,bbox_inches='tight',facecolor='white')
                plt.close(fig)
            cases.append(case)
    write_json(root/'selected_cases.json',dict(cases=cases,seed=20261008))
    print('[figures]',figroot,'cases',len(cases),flush=True)


if __name__=='__main__':
    main()
