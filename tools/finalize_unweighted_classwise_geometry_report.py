"""Add final interpretation and refresh report provenance after the corrected run."""
from pathlib import Path
import csv,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.unweighted_classwise_geometry_experiment import RUN,assert_protocol,sha,write
MARKER='<!-- uniform-geometry-original-classification-interpretation-v1 -->'

def main():
    assert_protocol()
    assert json.loads((RUN/'queue_progress.json').read_text())['status']=='complete'
    summary=json.loads((RUN/'summary.json').read_text());m=summary['metrics']
    f,c,b=[m[n] for n in ('unweighted_geometry','classwise','baseline')]
    with (RUN/'aggregate/per_class.csv').open() as stream:rows=list(csv.DictReader(stream))
    by={(r['arm'],int(r['class_id'])):r for r in rows}
    improved=sum(float(by['unweighted_geometry',i]['IoU'])>float(by['classwise',i]['IoU']) for i in range(17))
    path=ROOT/'analysis/unweighted_classwise_geometry_mini_results.md'
    text=path.read_text().split(MARKER)[0].rstrip()
    verdict=(f"**保留原分类后，去掉几何类别加权带来小幅恢复，但当前版本仍未超过原始目标。** "
             f"F 比 C 高 {f['mIoU']-c['mIoU']:.3f} 个百分点，比 B 低 {b['mIoU']-f['mIoU']:.3f} 个百分点。")
    if verdict not in text:
        pos=text.index('\n\n',text.index('**实验已完成。**'))
        text=text[:pos]+'\n\n'+verdict+text[pos:]
    lines=[MARKER,'','## 5. 结果怎样解读','',
        f"**类别权重解释了 C 的一部分退化。** 唯一改动是几何项由类别加权改为有效类等权，mIoU 提高 {f['mIoU']-c['mIoU']:.3f} 个百分点，17类中 {improved} 类提高；terrain 轻微下降。这个比较没有改动原分类目标或分类权重。",'',
        f"相对 C，漏检减少 {int(c['FN']-f['FN']):,} 个体素，二值 FP 增加 {int(f['FP']-c['FP']):,} 个体素；二值IoU提高 {f['IoU']-c['IoU']:.3f} 个百分点。同类1σ覆盖提高 {f['same1']-c['same1']:.3f} 个百分点，固定初始高负载区域的1σ覆盖提高 {f['high_core']-c['high_core']:.3f}、二值召回提高 {f['high_recall']-c['high_recall']:.3f} 个百分点。",'',
        f"Query 最近归属也只部分恢复：Top20份额由 {c['top20_nearest_share']:.3f}% 降至 {f['top20_nearest_share']:.3f}%，仍高于 B 的 {b['top20_nearest_share']:.3f}%。F 的整体1σ覆盖、同类覆盖、漏检与任务指标也仍弱于 B，因此不能认为已经解决覆盖或query负载问题。",'',
        '这组结果支持继续区分“每类如何汇总”与“几何匹配是否同类”两个因素。去掉外层类别权重后，类别内部仍各自求均值，稀有类和大类别仍获得相同的类间系数；这并未恢复原始全局样本归一化。权重变化同时影响优化强度，本轮也没有单独控制两者。', '',
        '此前36.513%的结果来自“逐类几何＋逐类分类”等权分支，是我对用户要求的误读。本报告39.871%的 F 才是本次要求的“逐类几何＋原分类、几何类间等权”结果。', '',
        '- [本轮权重](../work_dirs/unweighted_classwise_geometry_mini_v1/train/unweighted_geometry/epoch_1.pth)',
        '- [产物核验](../work_dirs/unweighted_classwise_geometry_mini_v1/final_integrity_audit.json)','']
    path.write_text(text+'\n\n'+'\n'.join(lines))
    interpretation=RUN/'interpretation_audit.json'
    write(interpretation,dict(status='complete',IoU_improved_classes=improved,
        delta_vs_classwise={k:f[k]-c[k] for k in ('mIoU','IoU','FN_rate','FP_over_GTfg','geo1','same1','top20_nearest_share')},
        delta_vs_baseline={k:f[k]-b[k] for k in ('mIoU','IoU','geo1')},
        original_classification_preserved=True,only_geometry_class_weights_changed=True))
    subprocess.run(['node','.chk.mjs',str(path),str(ROOT/'analysis/unweighted_classwise_geometry_mini_design.md')],cwd=ROOT,check=True,capture_output=True,text=True)
    audit_path=RUN/'final_integrity_audit.json';audit=json.loads(audit_path.read_text())
    for artifact in (path,interpretation,Path(__file__).resolve()):audit['artifacts'][str(artifact.relative_to(ROOT))]=sha(artifact)
    for name,digest in audit['artifacts'].items():assert sha(ROOT/name)==digest,name
    audit['finalized']=time.strftime('%Y-%m-%d %H:%M:%S %z')
    audit['documentation_check']='passed: KaTeX formulas, links and image targets'
    write(audit_path,audit)
    print(json.dumps({'status':'complete','mIoU':f['mIoU'],'IoU_improved_classes':improved,'report':str(path)}))

if __name__=='__main__':main()
