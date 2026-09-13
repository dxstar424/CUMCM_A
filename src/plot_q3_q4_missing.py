"""生成论文初稿中缺失的 Q3、Q4 基线分析图。

图 1：附件 2 半径实测点与 PCHIP 插值，辅助 Y 轴严格映射 R/R0；
图 2：Q3、Q4 两问基线的中心、表面含水率对照；
图 3：Q4 材料坐标下含水率时空热力图，叠加 0.1500 终止等值线。

图中不设置顶部标题，采用底部小图标注，便于直接排版到论文。
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q3_q4_missing_figures"
OUT.mkdir(parents=True, exist_ok=True)

font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Songti SC"
plt.rcParams.update({
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcfe",
    "savefig.facecolor": "white",
})

raw = np.load(ROOT / "results" / "inputs.npz")
rad = np.asarray(raw["rad"], dtype=float)
q3 = np.load(ROOT / "results" / "q3.npz")
q4 = np.load(ROOT / "results" / "q4.npz")

RAD0_CM = float(rad[0, 1])
Q3_EVENT_H = float(q3["event"]) / 3600.0
Q4_EVENT_H = float(q4["event"]) / 3600.0


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def set_four_decimals(ax) -> None:
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))


def through_event(run) -> tuple[np.ndarray, np.ndarray]:
    """仅展示真实终止时刻以前的数据，末状态由夹住事件的两步线性插值。"""
    time = np.asarray(run["t"], dtype=float)
    field = np.asarray(run["C"], dtype=float)
    event = float(run["event"])
    upper = int(np.searchsorted(time, event, side="left"))
    if not 0 < upper < len(time):
        raise ValueError("事件时刻未被已保存数据夹住")
    weight = (event - time[upper - 1]) / (time[upper] - time[upper - 1])
    endpoint = field[upper - 1] + weight * (field[upper] - field[upper - 1])
    return np.r_[time[:upper], event] / 3600.0, np.vstack([field[:upper], endpoint])


def plot_radius_pchip() -> None:
    """附件 2 半径测点、PCHIP 插值及无量纲半径双 Y 轴图。"""
    t_h = rad[:, 0] / 3600.0
    r_cm = rad[:, 1]
    pchip = PchipInterpolator(t_h, r_cm, extrapolate=False)
    dense_h = np.linspace(float(t_h[0]), float(t_h[-1]), 1200)
    dense_r = pchip(dense_h)

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    # R/R0 只是 R 的线性换算；辅助轴共享同一条曲线，避免独立双轴虚假偏离。
    ax2 = ax.secondary_yaxis("right", functions=(lambda value: value / RAD0_CM,
                                                 lambda value: value * RAD0_CM))
    ax.plot(t_h, r_cm, "o", ms=3.0, mfc="white", mec="#1f4e79", mew=0.8,
            label="附件2实测半径")
    ax.plot(dense_h, dense_r, color="#b63d3d", lw=1.8, label="PCHIP插值曲线")
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("半径 / cm")
    ax2.set_ylabel("半径比 $R/R_0$")
    ax.set_xlim(0.0, 72.0)
    ax.set_ylim(1.10, 2.08)
    ax.grid(alpha=0.22, ls="--", lw=0.6)
    h1, l1 = ax.get_legend_handles_labels()
    fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2,
               frameon=True, facecolor="white", edgecolor="#c8d0d9",
               framealpha=0.95, fontsize=8)
    set_four_decimals(ax)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.18, top=0.86)
    fig.text(0.50, 0.035, "附件2半径实测点与PCHIP插值及半径比", ha="center", va="center", fontsize=10)
    save(fig, "图_Q3Q4_半径实测点_PCHIP插值_半径比")


def plot_q3_q4_curves() -> None:
    """两问基线对照；物性关系也不同，时间差不能全部归因于半径收缩。"""
    t3, c3 = through_event(q3)
    t4, c4 = through_event(q4)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True,
                             gridspec_kw={"wspace": 0.23})
    colors = {"q3": "#1f4e79", "q4": "#b63d3d"}
    for ax, idx, ylab in ((axes[0], 0, "中心含水率"), (axes[1], -1, "表面含水率")):
        ax.plot(t3, c3[:, idx], color=colors["q3"], lw=1.9, label="问题三基线")
        ax.plot(t4, c4[:, idx], color=colors["q4"], lw=1.9, label="问题四基线")
        ax.axhline(0.1500, color="#555555", lw=1.0, ls=(0, (4, 2)), label="终止阈值 0.1500")
        ax.axvline(Q3_EVENT_H, color=colors["q3"], lw=0.9, ls=":")
        ax.axvline(Q4_EVENT_H, color=colors["q4"], lw=0.9, ls=":")
        ax.set_xlabel("时间 / h")
        ax.set_ylabel(ylab)
        ax.set_xlim(0.0, 60.0)
        ax.set_ylim(0.0, 2.70)
        ax.grid(alpha=0.22, ls="--", lw=0.6)
        set_four_decimals(ax)
    # 事件标签置于图外上边缘，避免覆盖曲线。
    # 两个终止时刻相距较近，分成两行标注，避免文字相互覆盖。
    for ax in axes:
        ax.text(Q3_EVENT_H, 1.085, f"问题三：{Q3_EVENT_H:.4f} h",
                transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=7.5, color=colors["q3"])
        ax.text(Q4_EVENT_H, 1.025, f"问题四：{Q4_EVENT_H:.4f} h",
                transform=ax.get_xaxis_transform(), ha="right", va="bottom",
                fontsize=7.5, color=colors["q4"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
               frameon=True, facecolor="white", edgecolor="#c8d0d9",
               framealpha=0.95, fontsize=8)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.25, top=0.79)
    fig.text(0.28, 0.095, "（a）两问基线的中心含水率对照", ha="center", va="center", fontsize=10)
    fig.text(0.72, 0.095, "（b）两问基线的表面含水率对照", ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.025, "两问物性关系与几何设定均有差异，终止时间差不单独代表收缩效应。",
             ha="center", va="center", fontsize=8, color="#444444")
    save(fig, "图_Q3Q4_两问基线_中心表面含水率对照")


def plot_q4_heatmap() -> None:
    """Q4 材料坐标下的含水率时空热力图及终止等值线。"""
    t_h, field = through_event(q4)
    xi = np.linspace(0.0, 1.0, q4["C"].shape[1])
    c = field.T
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    mesh = ax.pcolormesh(t_h, xi, c, shading="auto", cmap="YlOrRd",
                         norm=Normalize(vmin=0.0, vmax=2.60), rasterized=True)
    ax.contour(t_h, xi, c, levels=[0.1500], colors="#1f4e79",
               linewidths=1.2, linestyles="--")
    ax.axvline(Q4_EVENT_H, color="#1f4e79", lw=1.0, ls=":")
    ax.text(Q4_EVENT_H, 1.015, f"终止：{Q4_EVENT_H:.4f} h",
            transform=ax.get_xaxis_transform(), ha="right", va="bottom",
            fontsize=8, color="#1f4e79")
    fig.legend([Line2D([0], [0], color="#1f4e79", lw=1.2, ls="--")],
               ["含水率 0.1500 等值线"], loc="upper left", bbox_to_anchor=(0.12, 0.995),
               frameon=False, fontsize=8)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("材料坐标 $\\xi=r/R(t)$")
    ax.set_xlim(0.0, Q4_EVENT_H)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(alpha=0.16, ls=":", lw=0.5, color="white")
    cb = fig.colorbar(mesh, ax=ax, pad=0.025, fraction=0.046)
    cb.set_label("含水率")
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    cb.ax.yaxis.set_major_locator(MaxNLocator(6))
    fig.subplots_adjust(left=0.105, right=0.90, bottom=0.18, top=0.86)
    fig.text(0.50, 0.035, "问题四基线的材料坐标含水率时空演化", ha="center", va="center", fontsize=10)
    save(fig, "图_Q4_收缩半径_材料坐标含水率时空热力图")


def main() -> None:
    plot_radius_pchip()
    plot_q3_q4_curves()
    plot_q4_heatmap()
    t3, c3 = through_event(q3)
    t4, c4 = through_event(q4)
    checks = {
        "问题三终止时间_h": f"{t3[-1]:.4f}",
        "问题四终止时间_h": f"{t4[-1]:.4f}",
        "问题三事件插值中心含水率": f"{c3[-1, 0]:.4f}",
        "问题四事件插值中心含水率": f"{c4[-1, 0]:.4f}",
        "事件以后数据均已去除": bool(t3[-1] == Q3_EVENT_H and t4[-1] == Q4_EVENT_H),
        "辅助轴映射": "右轴半径比=左轴半径/初始半径；不重复绘制比值曲线",
        "对照口径": "两问基线对照；物性关系与几何设定均有差异，不将时间差全部归因于收缩",
    }
    (OUT / "Q3Q4补充图_绘图验证.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Q3终止时间：{Q3_EVENT_H:.4f} h；Q4终止时间：{Q4_EVENT_H:.4f} h")
    print(f"输出目录：{OUT}")


if __name__ == "__main__":
    main()
