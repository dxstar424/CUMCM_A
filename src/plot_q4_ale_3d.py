"""Q4残留ALE项三维时空曲面对照图。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Songti.ttc')
plt.rcParams.update({'font.family':'Songti SC','axes.unicode_minus':False,'font.size':8,
                     'axes.linewidth':.7,'xtick.direction':'in','ytick.direction':'in'})
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'q4_ale_compare'
base=np.load(OUT/'q4_材料坐标基线.npz'); ale=np.load(OUT/'q4_残留ALE对流.npz')
# 以统一时间网格和材料坐标下采样，保留全时段趋势与径向梯度。
step_t=10; step_x=4
idx=np.arange(0,min(len(base['t']),len(ale['t'])),step_t)
if idx[-1] != min(len(base['t']),len(ale['t']))-1: idx=np.append(idx,min(len(base['t']),len(ale['t']))-1)
idx_x=np.arange(0,201,step_x)
if idx_x[-1]!=200: idx_x=np.append(idx_x,200)
t=base['t'][idx]/3600.0; xi=idx_x/200.0
TT,XX=np.meshgrid(t,xi)
Cb=base['C'][idx][:,idx_x].T
Ca=ale['C'][idx][:,idx_x].T
D=Ca-Cb
# 物理半径仅用于辅助解释，材料坐标是三维曲面的横向空间轴。
fig=plt.figure(figsize=(10.0,7.2))
axes=[fig.add_subplot(2,2,1,projection='3d'),fig.add_subplot(2,2,2,projection='3d'),fig.add_subplot(2,2,(3,4),projection='3d')]
view=(26,-126)
for ax in axes: ax.view_init(elev=view[0],azim=view[1]); ax.set_box_aspect((1.55,1.0,.72)); ax.grid(True,alpha=.20); ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.zaxis.set_major_formatter(FormatStrFormatter('%.4f')); ax.set_xlabel('时间 / h',labelpad=5); ax.set_ylabel('材料坐标 ξ',labelpad=5)
# 统一高度范围，确保两组曲面可直接比较。
zmin=float(min(Cb.min(),Ca.min())); zmax=float(max(Cb.max(),Ca.max()))
for ax,Z,title,cmap in [(axes[0],Cb,'（a）材料坐标基线','YlGnBu'),(axes[1],Ca,'（b）保留残留网格项','YlOrRd')]:
    surf=ax.plot_surface(TT,XX,Z,cmap=cmap,rstride=2,cstride=2,linewidth=0,antialiased=True,alpha=.96)
    ax.set_title(title,pad=8,loc='left'); ax.set_zlabel('含水率',labelpad=5); ax.set_zlim(zmin,zmax)
    # 只在第一行面板给出稀疏刻度，避免信息过载。
    ax.set_zticks(np.linspace(0,2.5,6))
# 差值曲面：负值表示残留项组含水率低于基线。
ax=axes[2]; lim=max(abs(float(D.min())),abs(float(D.max())))
surf=ax.plot_surface(TT,XX,D,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),rstride=2,cstride=2,linewidth=0,antialiased=True,alpha=.98)
ax.set_title('（c）残留网格项引起的含水率差值',pad=8,loc='left'); ax.set_zlabel('含水率差',labelpad=5); ax.set_zlim(-lim,lim)
ax.set_zticks(np.linspace(-.3,0,4))
cb=fig.colorbar(surf,ax=ax,shrink=.62,pad=.08,aspect=18); cb.set_label('差值'); cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# 终止时间投影线：蓝色为基线，红色为残留网格项。
ev0=float(base['event'])/3600; ev1=float(ale['event'])/3600
for ax in axes:
    ax.plot([ev0,ev0],[0,1],[0,0],color='#1f4e79',lw=1.0,ls='--',alpha=.85)
    ax.plot([ev1,ev1],[0,1],[0,0],color='#c43c39',lw=1.0,ls=':',alpha=.85)
fig.text(.12,.015,'蓝色虚线：基线终止；红色点线：残留网格项终止',fontsize=8)
fig.subplots_adjust(left=.02,right=.96,bottom=.08,top=.96,wspace=.02,hspace=.08)
for ext in ('png','svg','pdf'): fig.savefig(OUT/f'Q4_ALE三维时空曲面对照.{ext}',dpi=400,bbox_inches='tight')
plt.close(fig)
