#!/usr/bin/env python3
"""Standalone teaching diagram; every arrow describes actual experiment flow."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'svg.fonttype':'path'})
fig,ax=plt.subplots(figsize=(12,6));fig.subplots_adjust(left=.025,right=.975,top=.84,bottom=.04)
ax.set_xlim(0,12);ax.set_ylim(0,6);ax.axis('off')
fig.suptitle('快速区域蒸馏：未来标签指导历史预测',x=.04,y=.96,ha='left',fontsize=20,weight='bold',color='#24374a')
fig.text(.04,.885,'Teacher 只在训练期使用；两种 Student 的推理输入都只有历史五帧。',fontsize=12,color='#647486')
def box(x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025,rounding_size=.12',facecolor='#f3f6fa',edgecolor=color,lw=1.4))
    ax.text(x+.16,y+h-.2,title,va='top',fontsize=14,weight='bold',color=color)
    ax.text(x+.16,y+h-.7,body,va='top',fontsize=11,color='#24374a',linespacing=1.7)
def arrow(a,b,color='#647486'):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.5,color=color))
box(.15,3.35,3.45,1.7,'冻结 Teacher','真实未来窗口 → 感知器\n相同原架构、相同初始权重','#bf5c4c')
box(4.15,3.35,3.45,1.7,'未来 occupancy 标签','原版体素化后的语义标签\n保留错误与 free，不用 GT 修正','#bf5c4c')
box(.15,.8,3.45,1.7,'历史 Student','固定感知器 / 全量微调\n原版未来预测模块，六步 rollout','#286b9e')
box(4.15,.8,3.45,1.7,'预测体素类别分数','与 Teacher 同时距、同坐标\n训练时使用可反传的类别分数','#286b9e')
box(8.15,2,3.6,2,'只在快速区域计算 KD','速度 ≥2 m/s 的 GT 占据体素\n1 / 2 / 3 秒 occupancy BCE\n总损失 = task + 0.1 × KD','#268579')
arrow((3.65,4.2),(4.05,4.2));arrow((3.65,1.65),(4.05,1.65))
arrow((7.65,4.15),(8.1,3.45));arrow((7.65,1.7),(8.1,2.55))
ax.text(6,5.65,'对照：每种 Student 各有一组 task 基线和一组快速区域 KD',ha='center',fontsize=12,color='#24374a')
ax.text(6,.15,'Mask 来自标注的实例速度；学生推理只读历史，区域 Mask 仅用于离线评分。',ha='center',fontsize=11,color='#647486')
out=Path('analysis/hypotheses/prediction_perception_regions/fast_motion_distillation/figures');out.mkdir(exist_ok=True)
for ext in ('png','svg'):fig.savefig(out/f'architecture.{ext}',dpi=180,facecolor='white')
plt.close(fig)
print('Saved architecture PNG + SVG')
