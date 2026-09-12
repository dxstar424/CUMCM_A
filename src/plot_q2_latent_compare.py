from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Songti.ttc')
from matplotlib.ticker import FormatStrFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'q2_latent_compare'
base=np.load(ROOT/'results'/'q2.npz')
lat=np.load(OUT/'q2_含潜热.npz')
plt.rcParams.update({'font.family':'Songti SC','axes.unicode_minus':False,'font.size':8,'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in'})
fig,axs=plt.subplots(2,2,figsize=(7.0,5.0),sharex=True)
t=lat['t']/3600
colors={'off':'#1f4e79','on':'#c43c39'}
# 中心温度
ax=axs[0,0]; ax.plot(t,base['T'][:,0],color=colors['off'],lw=1.5,label='不含蒸发潜热'); ax.plot(t,lat['T'][:,0],color=colors['on'],lw=1.5,label='含蒸发潜热'); ax.set_ylabel('中心温度 / ℃'); ax.legend(frameon=False,fontsize=7,loc='best'); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# 中心含水率
ax=axs[0,1]; ax.plot(t,base['C'][:,0],color=colors['off'],lw=1.5); ax.plot(t,lat['C'][:,0],color=colors['on'],lw=1.5); ax.set_ylabel('中心含水率'); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# 表面温度
ax=axs[1,0]; ax.plot(t,base['T'][:,-1],color=colors['off'],lw=1.5); ax.plot(t,lat['T'][:,-1],color=colors['on'],lw=1.5); ax.set_xlabel('时间 / h'); ax.set_ylabel('表面温度 / ℃'); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# 通量
ax=axs[1,1]; ax.plot(t,lat['q_conv'],color='#1f4e79',lw=1.5,label='对流热通量'); ax.plot(t,lat['q_lat'],color='#c43c39',lw=1.5,label='蒸发潜热通量'); ax.axhline(0,color='0.5',lw=.6); ax.set_xlabel('时间 / h'); ax.set_ylabel('表面热通量 / W·m$^{-2}$'); ax.legend(frameon=False,fontsize=7,loc='best'); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for ax in axs.flat:
    ax.grid(axis='y',color='#d9d9d9',lw=.5,alpha=.55); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
fig.subplots_adjust(left=.10,right=.98,bottom=.11,top=.98,wspace=.28,hspace=.28)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'Q2_蒸发潜热对照.{ext}',dpi=400,bbox_inches='tight')
plt.close(fig)
