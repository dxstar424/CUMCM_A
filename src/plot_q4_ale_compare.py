"""绘制Q4材料坐标基线与残留ALE网格对流诊断对照。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Songti.ttc')
plt.rcParams.update({'font.family':'Songti SC','axes.unicode_minus':False,'font.size':8,'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'q4_ale_compare'
base=np.load(ROOT/'results'/'q4.npz'); ale=np.load(OUT/'q4_残留ALE对流.npz')
t=base['t']/3600.0; ta=ale['t']/3600.0
# 统一到残留ALE终止时刻，突出相同时间窗的场差。
end=min(base['t'][-1],ale['t'][-1]); idx=np.where(base['t']<=end+1e-9)[0];
tc=base['t'][idx]/3600.0; cb=base['C'][idx,0]; ca=ale['C'][:len(idx),0]; sb=base['C'][idx,-1]; sa=ale['C'][:len(idx),-1]
R0=0.02
fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
# 中心含水率
a=axs[0,0]; a.plot(t,base['C'][:,0],color='#1f4e79',lw=1.6,label='材料坐标基线'); a.plot(ta,ale['C'][:,0],color='#c43c39',lw=1.6,label='残留ALE项'); a.axhline(.15,color='0.35',ls='--',lw=.8); a.set_xlabel('时间 / h'); a.set_ylabel('中心含水率'); a.set_title('(a) 中心含水率演化',loc='left',pad=6); a.legend(frameon=False,fontsize=7)
# 中心差值
a=axs[0,1]; a.plot(tc,ca-cb,color='#7a3e9d',lw=1.6); a.axhline(0,color='0.35',lw=.8); a.set_xlabel('时间 / h'); a.set_ylabel('中心含水率差'); a.set_title('(b) 残留ALE项引起的中心差值',loc='left',pad=6)
# 表面含水率
a=axs[1,0]; a.plot(t,base['C'][:,-1],color='#1f4e79',lw=1.6,label='材料坐标基线'); a.plot(ta,ale['C'][:,-1],color='#c43c39',lw=1.6,label='残留ALE项'); a.set_xlabel('时间 / h'); a.set_ylabel('表面含水率'); a.set_title('(c) 表面含水率演化',loc='left',pad=6); a.legend(frameon=False,fontsize=7)
# 半径曲线和事件位置
raw=np.load(ROOT/'results'/'inputs.npz'); rad=raw['rad']; a=axs[1,1]; a.plot(rad[:,0]/3600,rad[:,1],color='#2f7f6f',lw=1.6); a.axvline(float(base['event'])/3600,color='#1f4e79',ls='--',lw=.9,label='基线终止'); a.axvline(float(ale['event'])/3600,color='#c43c39',ls=':',lw=1.2,label='残留ALE终止'); a.set_xlabel('时间 / h'); a.set_ylabel('半径 / cm'); a.set_title('(d) 半径收缩与终止时刻',loc='left',pad=6); a.legend(frameon=False,fontsize=7)
for a in axs.flat:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y',color='#d9d9d9',lw=.5,alpha=.55); a.xaxis.set_major_formatter(FormatStrFormatter('%.4f')); a.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for ext in ('png','svg','pdf'): fig.savefig(OUT/f'Q4_ALE对照.{ext}',dpi=400,bbox_inches='tight')
plt.close(fig)
