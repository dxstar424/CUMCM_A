"""生成 Q3 基线缺失的四张分析图。

数据只读取已经保存的 q3.npz、inputs.npz、event_summary.json 和
sensitivity.json；不重跑模型，也不延长含水率时间序列。图中文字为中文，数值
统一保留四位小数，图例置于坐标区外，便于直接排版。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, FixedLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q3_baseline_figures"
OUT.mkdir(parents=True, exist_ok=True)

# macOS 自带简体中文字体，避免中文缺字和乱码。
FONT = "/System/Library/Fonts/Supplemental/Songti.ttc"
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    # 直接取字体文件登记的 family 名，避免 macOS 字体别名导致回退到 DejaVu。
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({
    "font.size": 9,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcfe",
    "savefig.facecolor": "white",
    "mathtext.fontset": "stix",
})

q3 = np.load(ROOT / "results" / "q3.npz")
inputs = np.load(ROOT / "results" / "inputs.npz")
events = json.loads((ROOT / "results" / "event_summary.json").read_text())
sensitivity = json.loads((ROOT / "results" / "sensitivity.json").read_text())
t_s = np.asarray(q3["t"], float)
t_h = t_s / 3600.0
C = np.asarray(q3["C"], float)
event_s = float(q3["event"])
event_h = event_s / 3600.0
threshold = float(events.get("threshold", 0.15))


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def four(ax, which=("x", "y")):
    if "x" in which:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    if "y" in which:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))


def figure_criteria() -> None:
    """全时段、尾部和事件跨步放大，支撑 57.4629 h 判据。"""
    cmax = np.max(C, axis=1)
    c0, cs = C[:, 0], C[:, -1]
    # 数据中最后一个样本为严格阈值后的 57.4667 h，不生成任何新样本。
    tail_start = 36.0
    mask = t_h >= tail_start
    prev = max(0, len(t_h) - 3)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.25),
                           gridspec_kw={"width_ratios": (1.35, 1.15, .90), "wspace": .30})
    colors = {"中心": "#1f4e79", "表面": "#b54a4a", "最大": "#2e7d5b"}
    for a, x, y0, ym, ys in ((ax[0], t_h, c0, cmax, cs), (ax[1], t_h[mask], c0[mask], cmax[mask], cs[mask])):
        a.plot(x, y0, color=colors["中心"], lw=1.75, label="中心含水率")
        a.plot(x, ym, color=colors["最大"], lw=0.9, ls="--", marker="o", markevery=max(1, len(x)//13), ms=2.4,
               mfc="white", label="全域最大值")
        a.plot(x, ys, color=colors["表面"], lw=1.55, label="表面含水率")
        a.axhline(threshold, color="#555555", ls=(0, (4, 2)), lw=1.0, label="终止阈值 0.1500")
        a.axvline(event_h, color="#1f4e79", ls=":", lw=1.0)
        a.set_xlabel("时间 / h"); a.set_ylabel("含水率")
        a.grid(alpha=.22, ls="--", lw=.55); a.set_ylim(0.0, 2.70); four(a)
    ax[0].set_xlim(0.0, t_h[-1])
    ax[1].set_xlim(tail_start, t_h[-1])
    # 最后两个步长与线性插值事件，纵轴用四位小数的 10^-4 量级，避免阈值附近被四舍五入抹平。
    tx = t_h[prev:]; yy = (c0[prev:] - threshold) * 1e4
    ax[2].plot(tx, yy, color="#1f4e79", lw=1.8, marker="o", ms=4, label="中心值减阈值")
    ax[2].axhline(0, color="#555555", ls=(0, (4, 2)), lw=1.0)
    ax[2].axvline(event_h, color="#b54a4a", ls=":", lw=1.1)
    ax[2].scatter([event_h], [0], color="#b54a4a", s=22, zorder=4, label="线性插值事件")
    ax[2].set_xlabel("时间 / h"); ax[2].set_ylabel(r"中心含水率差 / $10^{-4}$")
    ax[2].set_xlim(tx[0] - .002, tx[-1] + .002); ax[2].grid(alpha=.22, ls="--", lw=.55); four(ax[2])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(.5, .995), ncol=4,
               frameon=True, facecolor="white", edgecolor="#c8d0d9", fontsize=8)
    for x, txt in zip((.19, .50, .81), ("（a）全时段判据", "（b）终止前长时尾部", "（c）终止跨步放大")):
        fig.text(x, .035, txt, ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=.06, right=.985, bottom=.17, top=.79)
    save(fig, "图_Q3_长时含水率_终止判据")


def figure_heatmap() -> None:
    """Q3 固定半径含水率热力图，叠加 0.1500 等值线。"""
    radius_cm = np.linspace(0.0, 2.0, C.shape[1])
    fig, ax = plt.subplots(figsize=(8.5, 4.55))
    mesh = ax.pcolormesh(t_h, radius_cm, C.T, shading="auto", cmap="YlOrRd",
                         norm=Normalize(vmin=0.0, vmax=2.60), rasterized=True)
    contour = ax.contour(t_h, radius_cm, C.T, levels=[threshold], colors="#1f4e79",
                         linewidths=1.25, linestyles="--")
    if contour.allsegs[0]:
        ax.clabel(contour, fmt={threshold: "0.1500阈值"}, inline=True, fontsize=8)
    ax.axvline(event_h, color="#1f4e79", ls=":", lw=1.0)
    ax.text(event_h, 1.015, f"终止：{event_h:.4f} h", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", color="#1f4e79", fontsize=8)
    ax.set_xlabel("时间 / h"); ax.set_ylabel("半径 / cm")
    ax.set_xlim(0.0, t_h[-1]); ax.set_ylim(0.0, 2.0); ax.grid(alpha=.14, ls=":", lw=.5, color="white")
    four(ax)
    cb = fig.colorbar(mesh, ax=ax, pad=.025, fraction=.045)
    cb.set_label("含水率"); cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    cb.set_ticks([0.0, .5000, 1.0000, 1.5000, 2.0000, 2.6000])
    fig.subplots_adjust(left=.10, right=.90, bottom=.18, top=.95)
    fig.text(.50, .035, "（a）Q3固定半径含水率时空热力图及终止等值线", ha="center", va="center", fontsize=10)
    save(fig, "图_Q3_含水率时空热力图_终止等值线")


def figure_environment() -> None:
    """原始环境数据与 7200 s 后恒定平台的延拓。"""
    air = np.asarray(inputs["air"], float)
    at, temp, cb = air[:, 0] / 3600.0, air[:, 1], air[:, 2]
    x = np.array([0.0, 2.0, event_h]); tflat = np.array([50.0, 50.0, 50.0]); cflat = np.array([.05, .05, .05])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), gridspec_kw={"wspace": .27})
    # 前 4 h 保留附件 1 的全部输入，2 h 后实线为当前基线平台。
    ax = axes[0]; a2a = ax.twinx()
    m = at <= 4.0
    ax.plot(at[m], temp[m], color="#b54a4a", lw=1.4, marker="o", ms=2.1, mfc="white", label="附件1温度")
    a2a.plot(at[m], cb[m], color="#1f4e79", lw=1.4, marker="o", ms=2.1, mfc="white", label="附件1环境含水率")
    ax.axvline(2.0, color="#555555", ls="--", lw=.9); ax.axhline(50.0, color="#b54a4a", ls=":", lw=.8)
    a2a.axhline(.05, color="#1f4e79", ls=":", lw=.8)
    ax.set_xlim(0.0, 4.0); ax.set_xlabel("时间 / h"); ax.set_ylabel("环境温度 / ℃"); a2a.set_ylabel("环境含水率")
    a2a.set_ylim(.015, .055); a2a.set_yticks([.0200, .0300, .0400, .0500]); a2a.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(alpha=.2, ls="--", lw=.55); four(ax)
    # 长时平台不调用任何不存在的实验数据，仅显示模型实际使用的分段函数。
    ax = axes[1]; a2b = ax.twinx()
    ax.plot(x, tflat, color="#b54a4a", lw=2.0, label="基线温度平台 50.0000 ℃")
    a2b.plot(x, cflat, color="#1f4e79", lw=2.0, label="基线环境含水率平台 0.0500")
    ax.axvline(2.0, color="#555555", ls="--", lw=.9); ax.scatter([2.0], [50.0], color="#b54a4a", s=18, zorder=3)
    ax.set_xlim(2.0, event_h); ax.set_ylim(48.0, 52.0); a2b.set_ylim(.045, .055); a2b.set_yticks([.0460, .0480, .0500, .0520, .0540])
    ax.set_xlabel("时间 / h"); ax.set_ylabel("环境温度 / ℃"); a2b.set_ylabel("环境含水率")
    ax.grid(alpha=.2, ls="--", lw=.55); four(ax); a2b.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.text(event_h, 1.02, f"Q3终止 {event_h:.4f} h", transform=ax.get_xaxis_transform(), ha="right", color="#1f4e79", fontsize=8)
    # 只读取已创建的双 Y 轴对象；不要再次调用 twinx，否则会产生空的 0--1 轴。
    handles, labels = [], []
    for aa in (axes[0], a2a):
        hh, ll = aa.get_legend_handles_labels(); handles += hh; labels += ll
    for aa in (axes[1], a2b):
        hh, ll = aa.get_legend_handles_labels(); handles += hh; labels += ll
    # 去重且图例置于图外。
    uniq = {l: h for h, l in zip(handles, labels)}
    fig.legend(list(uniq.values()), list(uniq.keys()), loc="upper center", bbox_to_anchor=(.5, .995), ncol=3,
               frameon=True, facecolor="white", edgecolor="#c8d0d9", fontsize=8)
    fig.subplots_adjust(left=.075, right=.925, bottom=.18, top=.78)
    fig.text(.27, .035, "（a）附件1输入与平台起点", ha="center", va="center", fontsize=10)
    fig.text(.73, .035, "（b）Q3长时环境分段延拓", ha="center", va="center", fontsize=10)
    save(fig, "图_Q3_环境输入与长时平台延拓")


def figure_sensitivity() -> None:
    """读取当前 sensitivity.json，绘制终止时间相对基线的敏感性。"""
    baseline = event_h
    names = {"face_harm": "面通量：调和平均", "D_0.9": "扩散系数 ×0.9000", "D_1.1": "扩散系数 ×1.1000",
             "hm_0.9": "传质系数 ×0.9000", "hm_1.1": "传质系数 ×1.1000", "h_0.9": "传热系数 ×0.9000",
             "h_1.1": "传热系数 ×1.1000", "Cb_0.045": "平台含水率 0.0450", "Cb_0.055": "平台含水率 0.0550"}
    rows = [(names.get(r["label"], r["label"]), float(r["event_h"]), float(r["event_h"])-baseline) for r in sensitivity]
    rows.sort(key=lambda z: z[2])
    fig, ax = plt.subplots(figsize=(10.3, 5.35))
    y = np.arange(len(rows)); delta = np.array([r[2] for r in rows]); colors = np.where(delta < 0, "#2e7d5b", "#b54a4a")
    ax.barh(y, delta, color=colors, alpha=.88, height=.58, edgecolor="white", linewidth=.5)
    ax.axvline(0.0, color="#1f4e79", lw=1.1); ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("相对基线的终止时间变化 / h"); ax.set_xlim(min(-5.7, delta.min()-1), max(5.7, delta.max()+1))
    ax.grid(axis="x", alpha=.22, ls="--", lw=.55); four(ax, ("x",))
    for yy, (_, ev, d) in zip(y, rows):
        if d >= 0:
            xx, ha = d + .12, "left"
        else:
            # 负向条形的标注放在条形内部靠近零端，避免侵入纵轴标签。
            xx, ha = d + .12, "left"
        ax.text(xx, yy, f"{ev:.4f} h（{d:+.4f} h）", ha=ha, va="center", fontsize=7.4,
                color="#23313d", bbox=dict(facecolor="white", alpha=.78, edgecolor="none", pad=.25))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2e7d5b", label="终止时间缩短"), Patch(color="#b54a4a", label="终止时间延长")],
              loc="upper center", bbox_to_anchor=(.5, 1.08), ncol=2, frameon=True,
              facecolor="white", edgecolor="#c8d0d9", fontsize=8)
    ax.tick_params(axis="y", labelsize=8.0)
    fig.subplots_adjust(left=.38, right=.98, bottom=.16, top=.86)
    fig.text(.68, .035, f"（a）Q3当前基线敏感性（基线：{baseline:.4f} h）", ha="center", va="center", fontsize=10)
    save(fig, "图_Q3_终止时间_当前可复现敏感性")


def main() -> None:
    figure_criteria(); figure_heatmap(); figure_environment(); figure_sensitivity()
    summary = {"q3_event_h": f"{event_h:.4f}", "q3_event_s": f"{event_s:.4f}",
               "threshold": f"{threshold:.4f}",
               "q3_last_t_h": f"{float(t_h[-1]):.4f}", "q3_samples": int(len(t_h)),
               "center_max_abs": f"{float(np.max(np.abs(C[:, 0] - np.max(C, axis=1)))):.4f}",
               "outputs": sorted(p.name for p in OUT.iterdir() if p.suffix in {".png", ".svg", ".pdf"})}
    (OUT / "图_Q3_基线图清单.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
