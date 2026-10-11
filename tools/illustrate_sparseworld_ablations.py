#!/usr/bin/env python3
"""Reproducible conceptual diagrams; illustrations are not experimental samples."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib import font_manager
import numpy as np

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path'})
OUT=Path('analysis/hypotheses/prediction_perception_regions/ablations/figures')
BLUE='#286b9e';RED='#bf5c4c';TEAL='#268579';INK='#24374a';GRAY='#647486';PALE='#f3f6fa'

def canvas(title,subtitle,height=5):
    fig,ax=plt.subplots(figsize=(12,height));fig.subplots_adjust(left=.025,right=.975,bottom=.045,top=.83)
    ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off');fig.suptitle(title,x=.04,y=.965,ha='left',fontsize=20,color=INK,weight='bold')
    fig.text(.04,.89,subtitle,color=GRAY,fontsize=11)
    return fig,ax

def box(ax,x,y,w,h,title,body='',color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=.12',facecolor=PALE,edgecolor=color,lw=1.4))
    ax.text(x+.17,y+h-.22,title,va='top',color=color,weight='bold',fontsize=13)
    if body:ax.text(x+.17,y+h-.66,body,va='top',color=INK,fontsize=11,linespacing=1.65)

def arrow(ax,a,b,color=GRAY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.6,color=color))
def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
    plt.close(fig)

fig,ax=canvas('四组实验：沿着“图像 → 三维输出”寻找原因','阅读路线示意｜A1、A3、A4 改变模型条件；A2 统计不同条件下的区域差异。')
for x,title,body,c in [(0.15,'输入图像','历史窗口 / 未来窗口\n看到了哪些时刻？',BLUE),(4.2,'稀疏三维点','点的位置、数量与类别\n在哪里形成空间覆盖？',TEAL),(8.25,'体素输出','过滤、阈值与空间填补\n最终哪些体素被判对？',RED)]:box(ax,x,3.1,3.55,1.8,title,body,c)
arrow(ax,(3.75,4),(4.13,4));arrow(ax,(7.8,4),(8.18,4))
box(ax,.15,.35,3.55,1.9,'A1 + A4｜输入侧','A1：换窗口，比较信息组合\nA4：遮挡局部，检验输入依赖',BLUE)
box(ax,4.2,.35,3.55,1.9,'A3｜表示与后处理','控制 query 数与空间分布\n改变过滤、阈值和填补',TEAL)
box(ax,8.25,.35,3.55,1.9,'A2｜位置与观测条件','同一位置距两时刻车辆多远？\n实际输入有多少几何支持？',RED)
for x in (1.9,6,10):arrow(ax,(x,3.05),(x,2.35))
save(fig,'guide_experiment_map')

fig,ax=canvas('A1｜“未来窗口”替换了历史窗口，并非只多看几帧','以预测 3 秒后为例；每个圆点代表一个时刻的六相机图像。',5.5)
times=np.arange(-2,3.01,.5);xs=np.linspace(2.5,11.5,len(times))
for x,t in zip(xs,times):
    ax.axvline(x,ymin=.16,ymax=.91,color='#e8edf2',lw=1)
    ax.text(x,5.7,f'{t:g}',ha='center',color=GRAY,fontsize=11)
ax.text(.1,5.7,'相对当前时间 / 秒',color=GRAY,fontsize=11)
rows=[('H：历史五帧',4.65,[-2,-1.5,-1,-.5,0],BLUE),('P：未来五帧',3.4,[1,1.5,2,2.5,3],RED),('M：混合五帧',2.15,[-2,-1,1,2,3],TEAL)]
for label,y,ts,c in rows:
    ax.text(.1,y,label,va='center',color=c,weight='bold')
    selected=[xs[np.argmin(np.abs(times-t))] for t in ts]
    ax.plot(selected,[y]*5,color=c,lw=2)
    ax.scatter(xs,[y]*len(xs),s=40,facecolor='white',edgecolor='#c7d1dd',zorder=3)
    ax.scatter(selected,[y]*5,s=160,facecolor=c,edgecolor='white',lw=1.5,zorder=4)
ax.text(6.1,.65,'当前 t = 0',ha='center',color=BLUE,weight='bold');ax.text(11.5,.65,'目标 t = 3 s',ha='right',color=RED,weight='bold')
ax.text(.1,.05,'F 也只读历史五帧，但经过未来预测模块；H 则直接用感知头在目标坐标中输出。',color=INK,fontsize=11)
save(fig,'guide_a1_timeline')

fig,ax=canvas('A2｜结构没动，测量距离的车辆位置变了','几何关系示意，不对应某个真实样本；两种距离都指向同一个静态结构。')
ax.plot([.4,11.5],[2,2],lw=15,color='#e4eaf0',solid_capstyle='round',zorder=0)
for x,name,c in [(4.5,'当前车辆',BLUE),(10,'3 秒后的车辆',RED)]:
    ax.add_patch(FancyBboxPatch((x-.45,1.65),.9,.7,boxstyle='round,pad=.04',facecolor=c,edgecolor='white'))
    for xx in (x-.25,x+.25):ax.add_patch(Circle((xx,1.6),.09,color=INK))
    ax.text(x,1.15,name,ha='center',color=c,weight='bold')
ax.add_patch(Rectangle((.8,2.5),.8,1.6,facecolor='#d9e9e3',edgecolor=TEAL,lw=2))
ax.text(1.2,4.45,'同一静态结构',ha='center',color=TEAL,weight='bold')
arrow(ax,(1.2,3.4),(4.5,2.5),BLUE);ax.text(3.25,3.5,'距当前车 15 m',ha='center',color=BLUE)
arrow(ax,(1.6,4.05),(10,2.55),RED);ax.text(7,4.15,'距目标时刻车 40 m',ha='center',color=RED)
arrow(ax,(5.5,1.95),(9,1.95),INK)
ax.text(7.25,1.35,'自车向前运动',ha='center',color=GRAY)
box(ax,.35,.05,11.25,.8,'对应交叉表：当前 10–20 m × 目标 ≥30 m',color=TEAL)
save(fig,'guide_a2_coordinates')

fig,ax=canvas('A3｜一个正确体素，要经过哪些步骤？','query 可理解为一组可学习的三维探测单元；本模型每个 query 最终输出 48 个点。',5.5)
items=[(.1,'① query','3 秒 F：1040 个\nP：720 个',BLUE),(3.15,'② 三维点','预测位置 + 类别分数\n检查覆盖与最近点语义',TEAL),(6.2,'③ 过滤与落格','几何过滤、分数阈值\n将点放入体素网格',TEAL),(9.25,'④ 空间填补','padding：闭运算\n再得到最终占据标签',RED)]
for x,title,body,c in items:box(ax,x,3.65,2.6,1.9,title,body,c)
for x in (2.75,5.8,8.85):arrow(ax,(x,4.55),(x+.33,4.55))
box(ax,.1,.7,5.25,2,'控制点预算','等数量：F 随机保留 720 个完整 query\n等空间配额：每个 4 m 块内令双方数量相同',BLUE)
box(ax,6.1,.7,5.75,2,'控制后处理','双方一起关闭 padding，或一起调整阈值\n观察原有的准确率差是否仍然保留',RED)
ax.text(.15,.08,'注意：删 query 也会改变保留下来的位置与内容；“优势变化”不能直接等同于“点数是唯一原因”。',color=GRAY,fontsize=11)
save(fig,'guide_a3_pipeline')

fig,ax=canvas('A4｜比较“遮哪里”，同时控制“遮多少”','在同一分支、同一个 ROI 中比较；灰色遮挡的像素数在每个相机视角内完全相同。',5.5)
box(ax,.2,4.85,11.45,.8,'选取 ROI：选 GT 最多的 2 m 块，不使用 F / P 正确性',color=TEAL)
box(ax,.3,3.05,5.3,1.3,'F：遮历史邻近上下文','投影 ROI 周围 8 m 的静态结构',BLUE)
box(ax,6.3,3.05,5.3,1.3,'P：遮未来 ROI 自身','投影 ROI 内的静态体素',RED)
arrow(ax,(2.95,4.8),(2.95,4.4),BLUE);arrow(ax,(8.95,4.8),(8.95,4.4),RED)
ax.text(6,2.55,'每个分支各自进行下列三组评估',ha='center',color=INK,weight='bold')
for x,label,body in [(.3,'原始图像','保留输入，得到原始正确率'),(4.25,'定向遮挡','遮住指定投影区域'),(8.2,'等面积随机遮挡','平移同一 mask；运行两个种子')]:
    box(ax,x,.95,3.45,1.25,label,body,GRAY)
ax.text(6,.35,'核心比较：定向遮挡后的准确率 − 随机遮挡后的准确率',ha='center',color=INK,weight='bold')
ax.text(6,-.12,'越负 → 移除指定内容造成的损失更大。F 和 P 的遮挡面积不同，不能跨分支比较大小。',ha='center',color=GRAY,fontsize=11)
save(fig,'guide_a4_controls')
print('Wrote 5 conceptual diagrams as PNG + SVG')
