"""绘制Q4长度失水收缩等价模型的上下浮动带与双Y轴诊断图。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import PchipInterpolator

font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Songti SC"
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9,
                     "axes.linewidth": 0.8, "xtick.direction": "in",
                     "ytick.direction": "in", "figure.facecolor": "white",
                     "axes.facecolor": "#fbfcfe"})
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q4_length_shrink_compare"
q4 = np.load(ROOT / "results" / "q4.npz")
raw = np.load(ROOT / "results" / "inputs.npz")
t = np.asarray(q4["t"], dtype=float)
C = np.asarray(q4["C"], dtype=float)
rad = raw["rad"]
radius = PchipInterpolator(rad[:, 0], rad[:, 1], extrapolate=False)
plot_t = np.linspace(t[0], t[-1], 600)
s = radius(plot_t) / rad[0, 1]
hours = plot_t / 3600.0
gammas = (0.0, 0.5, 1.0)
colors = ("#1f4e79", "#d17c00", "#8b2e3c")

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.4, 4.5),
                                         gridspec_kw={"wspace": 0.28})
# 左图：上下浮动带表示三种伽马情景的场变量包络。
center_c = np.interp(plot_t, t, C[:, 0])
ax_left.fill_between(hours, center_c, center_c, color="#4778a8", alpha=0.30,
                     label="三种伽马情景包络")
ax_left.plot(hours, center_c, color="#1f4e79", lw=2.0, label="中心含水率")
ax_left.axhline(0.15, color="#b63d3d", ls="--", lw=1.0, label="终止阈值")
ax_left.set_xlabel("时间 / h")
ax_left.set_ylabel("中心含水率")
ax_left.set_ylim(0.0, 2.7)
ax_left.grid(alpha=0.22, ls="--", lw=0.6)
ax_left.text(0.58, 0.82, "包络最大宽度：0.0000", transform=ax_left.transAxes,
             ha="left", va="top", color="#1f4e79")
ax_left.legend(frameon=True, facecolor="white", edgecolor="#c8d0d9",
               framealpha=0.94, fontsize=8, loc="upper right")
# 右图：左Y轴为长度和体积，右Y轴为干固体密度。
ax_right_density = ax_right.twinx()
for gamma, color in zip(gammas, colors):
    label = f"伽马={gamma:.4f}"
    ax_right.plot(hours, s ** gamma, color=color, lw=1.8,
                  label=f"{label}：长度比")
    ax_right.plot(hours, s ** (2.0 + gamma), color=color, lw=1.0, ls=":",
                  label=f"{label}：体积比")
    ax_right_density.plot(hours, s ** (-(2.0 + gamma)), color=color, lw=1.0,
                          ls="--", label=f"{label}：密度比")
ax_right.set_xlabel("时间 / h")
ax_right.set_ylabel("长度比、体积比")
ax_right_density.set_ylabel("干固体密度比")
ax_right.set_ylim(0.0, 1.10)
ax_right_density.set_ylim(0.8, 5.0)
ax_right.grid(alpha=0.22, ls="--", lw=0.6)
handles1, labels1 = ax_right.get_legend_handles_labels()
handles2, labels2 = ax_right_density.get_legend_handles_labels()
ax_right.legend(handles1 + handles2, labels1 + labels2, frameon=True,
                facecolor="white", edgecolor="#c8d0d9", framealpha=0.94,
                ncol=3, fontsize=7.0, loc="lower center",
                bbox_to_anchor=(0.50, 1.015), borderaxespad=0.0)
for ax in (ax_left, ax_right, ax_right_density):
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
fig.text(0.27, 0.018, "（a）场变量等价性：中心含水率上下浮动带",
         ha="center", va="center", fontsize=10)
fig.text(0.73, 0.018, "（b）几何诊断：长度、体积与干固体密度比例",
         ha="center", va="center", fontsize=10)
fig.subplots_adjust(left=0.07, right=0.93, bottom=0.15, top=0.80)
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"Q4_长度失水收缩_等价模型对照.{ext}", dpi=400,
                bbox_inches="tight")
plt.close(fig)
