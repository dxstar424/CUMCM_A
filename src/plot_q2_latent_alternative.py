"""生成Q2蒸发潜热对照的时空差值与终点剖面图。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import FormatStrFormatter

font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family': 'Songti SC', 'axes.unicode_minus': False,
                     'font.size': 8, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in'})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'q2_latent_compare'
base = np.load(ROOT / 'results' / 'q2.npz')
latent = np.load(OUT / 'q2_含潜热.npz')
t = latent['t'] / 3600.0
r_cm = np.linspace(0.0, 2.0, latent['T'].shape[1])
# 每分钟取一点绘制差值图，保留全部径向节点。
idx = np.arange(0, len(t), 60)
if idx[-1] != len(t) - 1:
    idx = np.append(idx, len(t) - 1)

dT = latent['T'][idx] - base['T'][idx]
dC = latent['C'][idx] - base['C'][idx]
TT, RR = np.meshgrid(t[idx], r_cm)

fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
# 温度差时空图
ax = axs[0, 0]
norm = Normalize(vmin=float(dT.min()), vmax=0.0)
im = ax.pcolormesh(TT, RR, dT.T, cmap='Blues_r', norm=norm, shading='auto')
ax.set_xlabel('时间 / h'); ax.set_ylabel('半径 / cm')
cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=24); cb.set_label('温度差 / ℃'); cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# 含水率差时空图
ax = axs[0, 1]
vlim = max(abs(float(dC.min())), abs(float(dC.max())), 1e-12)
norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
im = ax.pcolormesh(TT, RR, dC.T, cmap='RdBu_r', norm=norm, shading='auto')
ax.set_xlabel('时间 / h'); ax.set_ylabel('半径 / cm')
cb = fig.colorbar(im, ax=ax, pad=0.02, aspect=24); cb.set_label('含水率差'); cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# 终点温度径向剖面
ax = axs[1, 0]
ax.plot(r_cm, base['T'][-1], color='#1f4e79', lw=1.7, label='不含蒸发潜热')
ax.plot(r_cm, latent['T'][-1], color='#c43c39', lw=1.7, label='含蒸发潜热')
ax.set_xlabel('半径 / cm'); ax.set_ylabel('温度 / ℃'); ax.legend(frameon=False, fontsize=7, loc='best')
# 终点含水率径向剖面
ax = axs[1, 1]
ax.plot(r_cm, base['C'][-1], color='#1f4e79', lw=1.7, label='不含蒸发潜热')
ax.plot(r_cm, latent['C'][-1], color='#c43c39', lw=1.7, label='含蒸发潜热')
ax.set_xlabel('半径 / cm'); ax.set_ylabel('含水率'); ax.legend(frameon=False, fontsize=7, loc='best')
for ax in axs.flat:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#d9d9d9', lw=.5, alpha=.55)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
for ext in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'Q2_蒸发潜热对照_时空差值版.{ext}', dpi=400, bbox_inches='tight')
plt.close(fig)
