"""绘制Q4材料坐标基线与残留网格项的诊断对照图。

图中“残留网格项”仅表示对照计算中保留的网格输运项，避免把该
诊断口径表述为ALE方法本身的结论。两组数据均使用真实首次达到
含水率阈值的 event 字段作终止时刻标记。
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
FONT = '/System/Library/Fonts/Supplemental/Songti.ttc'
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    FONT_FAMILY = 'Songti SC'
else:
    FONT_FAMILY = 'STSong'
plt.rcParams.update({'font.family':FONT_FAMILY,'axes.unicode_minus':False,
                     'font.size':8,'axes.linewidth':.8,
                     'xtick.direction':'in','ytick.direction':'in'})
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'q4_ale_compare'
base=np.load(OUT/'q4_材料坐标基线.npz')
grid=np.load(OUT/'q4_残留ALE对流.npz')

def scalar(data, key):
    """读取零维 npz 字段并转为普通浮点数。"""
    return float(np.asarray(data[key]).reshape(()))

# 两组均保存到 184020.0000 s；仍按时间戳取交集，避免依赖数组长度。
common_end=min(float(base['t'][-1]), float(grid['t'][-1]), 184020.0)
ib=np.where(base['t'] <= common_end + 1e-9)[0]
ig=np.where(grid['t'] <= common_end + 1e-9)[0]
tb=base['t'][ib] / 3600.0
tg=grid['t'][ig] / 3600.0
cb=base['C'][ib, 0]
cg=grid['C'][ig, 0]
sb=base['C'][ib, -1]
sg=grid['C'][ig, -1]
event_b=scalar(base, 'event')
event_g=scalar(grid, 'event')

# 两组时间步相同；差值仍用时间戳插值，保证脚本对未来冻结结果稳健。
cg_on_b=np.interp(base['t'][ib], grid['t'][ig], cg)
center_delta=cg_on_b-cb
event_b_h=event_b/3600.0
event_g_h=event_g/3600.0

fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
# (a) 中心含水率与阈值
a=axs[0,0]
a.plot(tb,cb,color='#1f4e79',lw=1.6,label='材料坐标基线')
a.plot(tg,cg,color='#c43c39',lw=1.6,label='残留网格项')
a.axhline(.15,color='0.35',ls='--',lw=.8,label='终止阈值')
a.axvline(event_b_h,color='#1f4e79',ls='--',lw=.8)
a.axvline(event_g_h,color='#c43c39',ls=':',lw=1.0)
a.plot(event_b_h,.15,'o',ms=3.5,color='#1f4e79')
a.plot(event_g_h,.15,'o',ms=3.5,color='#c43c39')
a.set_xlabel('时间 / h'); a.set_ylabel('中心含水率')
a.set_title('(a) 中心含水率演化与终止阈值',loc='left',pad=6)
a.legend(frameon=False,fontsize=7,loc='upper right')
# (b) 中心差值
a=axs[0,1]
a.plot(tb,center_delta,color='#7a3e9d',lw=1.6)
a.axhline(0,color='0.35',lw=.8)
a.axvline(event_b_h,color='#1f4e79',ls='--',lw=.8)
a.axvline(event_g_h,color='#c43c39',ls=':',lw=1.0)
a.set_xlabel('时间 / h'); a.set_ylabel('中心含水率差')
a.set_title('(b) 残留网格项引起的中心差值',loc='left',pad=6)
# (c) 表面含水率
a=axs[1,0]
a.plot(tb,sb,color='#1f4e79',lw=1.6,label='材料坐标基线')
a.plot(tg,sg,color='#c43c39',lw=1.6,label='残留网格项')
a.axvline(event_b_h,color='#1f4e79',ls='--',lw=.8)
a.axvline(event_g_h,color='#c43c39',ls=':',lw=1.0)
a.set_xlabel('时间 / h'); a.set_ylabel('表面含水率')
a.set_title('(c) 表面含水率演化',loc='left',pad=6)
a.legend(frameon=False,fontsize=7,loc='upper right')
# (d) 终止时间对照
a=axs[1,1]
labels=['材料坐标\n基线','残留\n网格项']
vals=[event_b_h,event_g_h]
bars=a.bar(labels,vals,width=.52,color=['#1f4e79','#c43c39'],alpha=.9)
for bar,val in zip(bars,vals):
    a.text(bar.get_x()+bar.get_width()/2,val+0.35,f'{val:.4f}',ha='center',va='bottom',fontsize=8)
a.set_ylabel('终止时间 / h')
a.set_title('(d) 两种口径的终止时间对照',loc='left',pad=6)
a.set_ylim(0,max(vals)*1.14)
for a in axs.flat:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
    a.grid(axis='y',color='#d9d9d9',lw=.5,alpha=.55)
    # 柱状图横轴为中文分类标签，不套用数值坐标格式。
    if a is not axs[1,1]:
        a.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    a.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for ext in ('png','svg','pdf'): fig.savefig(OUT/f'Q4_ALE对照.{ext}',dpi=400,bbox_inches='tight')
plt.close(fig)
