"""Q4残留ALE项三维时空曲面对照图：水平排布，子图名置底。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Songti.ttc')
plt.rcParams.update({'font.family':'Songti SC','axes.unicode_minus':False,'font.size':8,
                     'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in'})
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'q4_ale_compare'
base=np.load(OUT/'q4_材料坐标基线.npz'); ale=np.load(OUT/'q4_残留ALE对流.npz')
step_t=10; step_x=4
n=min(len(base['t']),len(ale['t'])); idx=np.arange(0,n,step_t)
if idx[-1] != n-1: idx=np.append(idx,n-1)
idx_x=np.arange(0,201,step_x)
if idx_x[-1] != 200: idx_x=np.append(idx_x,200)
t=base['t'][idx]/3600.0; xi=idx_x/200.0
TT,XX=np.meshgrid(t,xi); Cb=base['C'][idx][:,idx_x].T; Ca=ale['C'][idx][:,idx_x].T; D=Ca-Cb
fig=plt.figure(figsize=(15.0,5.8))
axes=[fig.add_subplot(1,3,k,projection='3d') for k in (1,2,3)]
for ax in axes:
    ax.view_init(elev=25,azim=-126); ax.set_box_aspect((1.48,1.0,.74)); ax.grid(True,alpha=.20)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.set_xlabel('时间 / h',labelpad=4); ax.set_ylabel('材料坐标 ξ',labelpad=4); ax.set_zlabel('含水率',labelpad=4)
zmin=float(min(Cb.min(),Ca.min())); zmax=float(max(Cb.max(),Ca.max()))
for ax,Z,cmap in [(axes[0],Cb,'YlGnBu'),(axes[1],Ca,'YlOrRd')]:
    ax.plot_surface(TT,XX,Z,cmap=cmap,rstride=2,cstride=2,linewidth=0,antialiased=True,alpha=.96)
    ax.set_zlim(zmin,zmax); ax.set_zticks(np.linspace(0,2.5,6))
lim=max(abs(float(D.min())),abs(float(D.max())))
ax=axes[2]; surf=ax.plot_surface(TT,XX,D,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),rstride=2,cstride=2,linewidth=0,antialiased=True,alpha=.98)
ax.set_zlabel('含水率差',labelpad=5); ax.set_zlim(-lim,lim); ax.set_zticks(np.linspace(-.4,0,5))
cb=fig.colorbar(surf,ax=ax,shrink=.62,pad=.03,aspect=18); cb.set_label('含水率差'); cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ev0=float(base['event'])/3600; ev1=float(ale['event'])/3600
for ax in axes:
    ax.plot([ev0,ev0],[0,1],[0,0],color='#1f4e79',lw=1.0,ls='--',alpha=.85)
    ax.plot([ev1,ev1],[0,1],[0,0],color='#c43c39',lw=1.0,ls=':',alpha=.85)
# 使用图级文字设置统一水平基线，避免三维坐标框造成标题高低不齐。
labels=['（a）材料坐标基线','（b）保留残留网格项','（c）残留网格项引起的含水率差值']
for xpos,label in zip((.17,.50,.83),labels):
    fig.text(xpos,.070,label,ha='center',va='center',fontsize=10)
fig.text(.50,.018,'蓝色虚线：基线终止；红色点线：残留网格项终止',ha='center',va='center',fontsize=8)
fig.subplots_adjust(left=.015,right=.985,bottom=.145,top=.98,wspace=.015)
for ext in ('png','svg','pdf'): fig.savefig(OUT/f'Q4_ALE三维时空曲面对照.{ext}',dpi=400,bbox_inches='tight')
plt.close(fig)
