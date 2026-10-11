#!/usr/bin/env python3
"""Reproducible examples illustrating full-census pose and query diagnostics."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from tools.sparseworld_full_regions_common import ROOT,REPORT
from tools.sparseworld_fast_motion_common import MotionIndex
from tools.sparseworld_query_geometry import fit_rigid
from tools.sparseworld_region_utils import voxel_centers,write_json
import tools.diagnose_sparseworld_query_lineage as query


def main():
    assert json.loads((ROOT/'motion_summary.json').read_text())['status']=='complete'
    q=pd.read_parquet(ROOT/'query_instance_diagnostics.parquet')
    o=pd.read_parquet(ROOT/'motion_instance_oracles.parquet')
    keys=['source_token','horizon_seconds','instance_token']
    q=q[(q.split=='val')&q.eligible&(q.horizon_seconds==3)&(q.label==4)&(q.gt_voxels>=100)].copy()
    q=q.merge(o[(o.split=='val')&(o.radius_m==2)][keys+['eligible','F_base_iou','F_rigid_iou','current_gt_voxels','current_border','future_border']],on=keys,suffixes=('','_occupancy'))
    q=q[(q.raw_squared>1e-6)&(q.current_gt_voxels>=20)&~q.current_border.fillna(True)&~q.future_border].copy();q['pose_share']=q.pose_removable_squared/q.raw_squared
    q['shape_harm']=q.template_gt_iou-q.aligned_actual_gt_iou
    pools=[('位姿占主导',q[(q.pose_share>=.7)&q.eligible_occupancy&(q.F_rigid_iou-q.F_base_iou>=.1)],'pose_share'),
           ('历史形状保持更好',q[(q.shape_harm>=.01)&(q.pair_distance_change_m>=.3)],'shape_harm'),
           ('位姿与剩余差异并存',q[(q.pose_share>=.35)&(q.pose_share<=.65)],'pose_share')]
    selected=[];used=set()
    for name,pool,metric in pools:
        pool=pool.copy();pool['selection_distance']=abs(pool[metric]-pool[metric].median())
        pool=pool.sort_values(['selection_distance','source_token','instance_token'])
        for _,r in pool.iterrows():
            key=tuple(r[k] for k in keys)
            if key not in used:
                selected.append((name,r,int(len(pool))));used.add(key);break
    manifest=json.loads((ROOT/'manifest.json').read_text());records={r['token']:r for r in manifest['samples']}
    motion=MotionIndex();query.MOTION=motion;query.RUN_ROOT=ROOT
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');font_manager.fontManager.addfont(font.get_file())
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'path'})
    xyz=voxel_centers().reshape(-1,3)
    text='''# 具体案例｜全量验证中的位置与表示形状

[全量结果](results.md) · [机制统计](motion_results.md)

本页图例来自完整验证集。为便于阅读，固定选取 3 秒、car、未来 GT 至少 100 体素、当前 GT 至少 20 体素、两时刻框均未截断且存在可靠历史 query 绑定的目标；在各固定图例条件中选取最接近该条件中位数的实例，按 token 打破并列。**这些图用于解释统计含义，不用于估计发生比例；总体排序以机制报告的完整分母为准。**

上排在目标时刻原生 ego 网格中展示 GT、原始预测器 F、未来感知器 P：蓝色是 car 占据，灰色是其他占据，虚线为目标 GT 框。下排在目标物体坐标系中展示绑定 query 的未过滤点：青色为当前表示按真实运动搬移的模板，橙色为实际未来表示，灰色为目标 GT。右下图展示已找到的刚体纠正后的 XZ 结构；点的顺序不作为对应关系。

每个 query 可以含目标以外的点；形状图描述与目标绑定的表示，不等于完整车身表面。下排不使用未来类别和置信度筛点，坐标轴范围包含全部原始点。强烈的局部漂移也可能触发原 renderer 的组中心过滤，详见机制报告。
'''
    audit=[]
    for number,(name,row,pool_n) in enumerate(selected,1):
        rec=records[row.source_token];j=int(row.horizon_seconds)-1;target=rec['targets'][j]
        meta=json.loads((ROOT/'attributes'/(rec['token']+'.json')).read_text())['instances'][j]
        inst=next(a for a in meta if a['token']==row.instance_token)
        pose=np.asarray(target['pose']);cp=np.asarray(rec['current_pose']);rot=Quaternion(inst['rotation']).rotation_matrix;center=np.asarray(inst['translation'])
        with np.load(ROOT/'pairs'/(rec['token']+'.npz')) as pair,np.load(ROOT/'query_lineage'/(rec['token']+'.npz')) as raw,np.load(ROOT/'attributes'/(rec['token']+'.npz')) as attrs:
            seed=query.bindings(rec,pair,raw)[row.instance_token];parents=seed['queries'];a=seed['annotation']
            assert len(parents)==int(row.parent_queries)
            current=raw['source_points'][parents].reshape(-1,3).astype(float)@cp[:3,:3].T+cp[:3,3]
            expected=(current-np.asarray(a['translation']))@Quaternion(a['rotation']).rotation_matrix@rot.T+center
            actual=raw['future_points_'+str(j)][parents].reshape(-1,3).astype(float)@pose[:3,:3].T+pose[:3,3]
            fitted=fit_rigid(expected,actual);np.testing.assert_allclose(fitted['raw_squared'],row.raw_squared,rtol=1e-6)
            ids=np.flatnonzero(attrs['instance'][j].reshape(-1)==inst['id']);truth=xyz[ids]@pose[:3,:3].T+pose[:3,3]
            native_center=(center-pose[:3,3])@pose[:3,:3];span=max(6.,max(inst['size'][:2])+3)
            crop=(abs(xyz[:,0]-native_center[0])<=span)&(abs(xyz[:,1]-native_center[1])<=span)
            fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
            for ax,key,title in zip(axes[0],('GT','F','P'),('目标 GT','原始预测器 F','未来感知器 P')):
                labels=pair[key][j].reshape(-1);occupied=crop&(labels<17)
                for cls,color,size in [(False,'#c4c8cb',2),(True,'#36799a',4)]:
                    points=xyz[occupied&((labels==4)==cls)]
                    ax.scatter(points[:,1],points[:,0],s=size,c=color,rasterized=True)
                local=np.asarray([[-1,-1,0],[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])*np.asarray(inst['size'])[[1,0,2]]/2
                outline=(local@rot.T+center-pose[:3,3])@pose[:3,:3]
                ax.plot(outline[:,1],outline[:,0],'k--',lw=1)
                ax.set_xlim(native_center[1]-span,native_center[1]+span);ax.set_ylim(native_center[0]-span,native_center[0]+span)
                ax.set_title(title);ax.set_xlabel('ego y（左，m）');ax.set_ylabel('ego x（前，m）');ax.set_aspect('equal');ax.grid(alpha=.1)
            body={k:(p-center)@rot for k,p in [('GT',truth),('模板',expected),('实际',actual),('纠正',fitted['aligned'])]}
            for ax,k,dims,title in [(axes[1,0],'实际',(0,1),'原始未来形状：XY'),(axes[1,1],'纠正',(0,1),'纠正位姿后：XY'),(axes[1,2],'纠正',(0,2),'纠正位姿后：XZ')]:
                for key,color,label in [('GT','#9ba2a8','GT'),('模板','#2799a2','保持历史形状'),(k,'#c96d49','实际未来表示')]:
                    p=body[key];ax.scatter(p[:,dims[0]],p[:,dims[1]],s=3,alpha=.35,c=color,label=label,rasterized=True)
                ax.set_title(title);ax.set_xlabel('物体 x（m）');ax.set_ylabel('物体 '+('y' if dims[1]==1 else 'z')+'（m）');ax.set_aspect('equal');ax.grid(alpha=.1)
            axes[1,0].legend(frameon=False,fontsize=8)
            fig.suptitle(f'{name}｜3 秒 car｜GT {int(row.gt_voxels)} 体素｜{len(parents)} 个历史 query')
            filename=f'case_{number}'
            for ext in ('png','svg'):fig.savefig(REPORT/'figures'/(filename+'.'+ext),dpi=170)
            plt.close(fig)
        text+=f'\n## {number}. {name}\n\n![{name}](figures/{filename}.png)\n\n候选池 {pool_n:,} 个目标；场景 `{row.scene}`，源帧 `{row.source_token}`，实例 `{row.instance_token}`。GT 速度 {row.speed:.2f} m/s，刚体可恢复原始点集差异 {row.pose_share*100:.2f}%，内部成对距离变化 {row.pair_distance_change_m:.2f} m；历史模板相对纠正后实际表示的未来 GT IoU 差为 {row.shape_harm*100:.2f} 个百分点。\n'
        audit.append(dict(name=name,source_token=row.source_token,instance_token=row.instance_token,horizon=3,candidate_instances=pool_n,figure=filename))
    if not selected:text+='\n本轮没有目标满足预先指定的图例条件，未强行选择支持预设方向的实例。\n'
    (REPORT/'cases.md').write_text(text)
    write_json(ROOT/'case_selection.json',dict(rule='Val, car, 3s, future GT>=100/current GT>=20, neither box truncated, historical binding; nearest conditional median with token tie break',selected=audit))


if __name__=='__main__':main()
