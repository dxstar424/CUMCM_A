"""补算 Q3、Q4 基线之外的两类敏感性。

本脚本只在进程内修改模型参数，不回写生产基线文件：
1. Q3 将恒温恒湿平台开始时刻改为 3600、5400、7200、9000、10800 s；
2. Q4 将附件 2 半径曲线分别用 PCHIP 和分段线性插值。

所有输出数值按四位小数写入，结果目录为 results/q3_q4_additional_sensitivity。
"""
from __future__ import annotations

import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q3_q4_additional_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# model.py 在加载时读取 Excel。当前机器的 Homebrew Python 未安装 openpyxl，
# 这里仅为该次读取提供 npz 输入和最小模块元数据，不改变模型计算。
RAW = np.load(ROOT / "results" / "inputs.npz")
_read_excel = pd.read_excel


def _read_inputs(path, *args, **kwargs):
    key = "air" if "附件1" in str(path) else "rad"
    return pd.DataFrame(RAW[key])


pd.read_excel = _read_inputs
if "openpyxl" not in sys.modules:
    stub = types.ModuleType("openpyxl")
    stub.__version__ = "3.1.0"
    stub.Workbook = object
    sys.modules["openpyxl"] = stub
try:
    from src import model
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "src"))
    import model
pd.read_excel = _read_excel


def f4(value):
    """将结果按论文约定格式化为四位小数。"""
    return f"{float(value):.4f}"


def q3_run(plateau_start: float):
    old = model.PLATEAU_START
    try:
        model.PLATEAU_START = float(plateau_start)
        return model.run_fixed("q2", 1_000_000.0, 60.0, 60.0, True)
    finally:
        model.PLATEAU_START = old


def q4_run(method: str):
    original = model.RAD_PCHIP
    try:
        if method == "PCHIP":
            model.RAD_PCHIP = original
        elif method == "分段线性":
            model.RAD_PCHIP = interp1d(
                model.RAD_T, model.RAD_CM / 100.0,
                kind="linear", bounds_error=False,
                fill_value=(model.RAD_CM[0] / 100.0, model.RAD_CM[-1] / 100.0),
            )
        else:
            raise ValueError(method)
        return model.run_q4(end=300_000.0, dt=60.0)
    finally:
        model.RAD_PCHIP = original


def save_plot(q3_rows, q4_rows):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Songti SC"
    plt.rcParams.update({
        "font.size": 9, "axes.unicode_minus": False,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "figure.facecolor": "white", "axes.facecolor": "#fbfcfe",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=False,
                             gridspec_kw={"wspace": 0.27})
    blue, red, gray = "#1f4e79", "#b63d3d", "#67727e"
    # 左图：终止时间点与相对基线的竖直差值区间。
    x = np.arange(len(q3_rows), dtype=float)
    y = np.array([float(r["终止时间_h"]) for r in q3_rows])
    base = float(q3_rows[2]["终止时间_h"])
    axes[0].axhline(base, color=gray, lw=1.0, ls=(0, (4, 2)), zorder=1)
    axes[0].vlines(x, np.minimum(y, base), np.maximum(y, base), color=red, lw=1.3, alpha=.8, zorder=2)
    axes[0].scatter(x, y, s=35, color=blue, edgecolors="white", linewidths=.8, zorder=3)
    axes[0].set_xticks(x, [f"{float(r['平台开始_s'])/3600:.4f}" for r in q3_rows])
    axes[0].set_xlabel("平台开始时间 / h")
    axes[0].set_ylabel("终止时间 / h")
    axes[0].set_ylim(min(y.min(), base) - .3, max(y.max(), base) + .3)
    axes[0].grid(alpha=.22, ls="--", lw=.6)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0].text(0.99, 0.04, "点：终止时间；竖线：相对基线差值",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7.5, color="#4d5660")
    # 右图：两种半径插值终止时间及相对 PCHIP 插值差值。
    x2 = np.arange(len(q4_rows), dtype=float)
    y2 = np.array([float(r["终止时间_h"]) for r in q4_rows])
    base2 = y2[0]
    axes[1].axhline(base2, color=gray, lw=1.0, ls=(0, (4, 2)), zorder=1)
    axes[1].vlines(x2, np.minimum(y2, base2), np.maximum(y2, base2), color=red, lw=1.3, alpha=.8, zorder=2)
    axes[1].scatter(x2, y2, s=35, color=blue, edgecolors="white", linewidths=.8, zorder=3)
    axes[1].set_xticks(x2, ["PCHIP插值", "分段线性插值"])
    axes[1].set_xlabel("半径插值方法")
    axes[1].set_ylabel("终止时间 / h")
    axes[1].set_ylim(min(y2.min(), base2) - .05, max(y2.max(), base2) + .05)
    axes[1].grid(alpha=.22, ls="--", lw=.6)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[1].text(0.99, 0.04, "点：终止时间；竖线：相对 PCHIP 插值差值",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7.5, color="#4d5660")
    handles = [Line2D([0], [0], marker="o", color=blue, markerfacecolor=blue,
                      markeredgecolor="white", lw=0, label="重算终止时间"),
               Line2D([0], [0], color=red, lw=1.3, label="相对基线差值"),
               Line2D([0], [0], color=gray, lw=1.0, ls=(0, (4, 2)), label="基线终止时间")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, frameon=True, facecolor="white", edgecolor="#c8d0d9", fontsize=8)
    fig.subplots_adjust(left=.085, right=.97, bottom=.24, top=.82)
    # 两个小标题使用同一 figure 坐标基线，避免落在曲线和图例上。
    fig.text(.285, .095, "（a）恒温恒湿平台开始时刻敏感性", ha="center", va="center", fontsize=10)
    fig.text(.715, .095, "（b）半径插值方法敏感性", ha="center", va="center", fontsize=10)
    fig.text(.5, .035, "问题三、问题四新增基线敏感性；数值均保留四位小数",
             ha="center", va="center", fontsize=8, color="#444444")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"Q3Q4_新增敏感性_阶段切换与半径插值.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    start = time.time()
    stored_q3 = float(np.load(ROOT / "results" / "q3.npz")["event"])
    stored_q4 = float(np.load(ROOT / "results" / "q4.npz")["event"])
    plateau_values = [3600.0, 5400.0, 7200.0, 9000.0, 10800.0]
    q3_runs = [q3_run(v) for v in plateau_values]
    q3_base = q3_runs[2]
    q3_rows = []
    for start_s, run in zip(plateau_values, q3_runs):
        event_s = float(run["event"])
        field_diff = float(np.max(np.abs(run["C"][:len(q3_base["C"])] - q3_base["C"])))
        q3_rows.append({
            "平台开始_s": start_s, "平台开始_h": start_s / 3600.0,
            "终止时间_s": event_s, "终止时间_h": event_s / 3600.0,
            "相对7200秒基线终止时间差_h": (event_s - float(q3_base["event"])) / 3600.0,
            "相对7200秒基线终止时间差_percent": (event_s - float(q3_base["event"])) / float(q3_base["event"]) * 100.0,
            "相对7200秒基线最大含水率场差": field_diff,
            "时间步数": len(run["t"]),
        })
    q4_methods = ["PCHIP", "分段线性"]
    q4_runs = [q4_run(method) for method in q4_methods]
    q4_base = q4_runs[0]
    q4_rows = []
    for method, run in zip(q4_methods, q4_runs):
        event_s = float(run["event"])
        q4_rows.append({
            "插值方法": method, "终止时间_s": event_s, "终止时间_h": event_s / 3600.0,
            "相对PCHIP终止时间差_h": (event_s - float(q4_base["event"])) / 3600.0,
            "相对PCHIP终止时间差_percent": (event_s - float(q4_base["event"])) / float(q4_base["event"]) * 100.0,
            "相对PCHIP最大含水率场差": float(np.max(np.abs(run["C"][:len(q4_base["C"])] - q4_base["C"]))),
            "时间步数": len(run["t"]),
        })
    pd.DataFrame(q3_rows).to_csv(OUT / "Q3_平台开始时刻敏感性.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.DataFrame(q4_rows).to_csv(OUT / "Q4_半径插值敏感性.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    consistency = {
        "生产文件Q3终止时间_h": f4(stored_q3 / 3600.0),
        "Q3_7200秒重算终止时间_h": f4(float(q3_base["event"]) / 3600.0),
        "Q3_7200秒事件绝对差_s": f4(abs(float(q3_base["event"]) - stored_q3)),
        "生产文件Q4终止时间_h": f4(stored_q4 / 3600.0),
        "Q4_PCHIP重算终止时间_h": f4(float(q4_base["event"]) / 3600.0),
        "Q4_PCHIP事件绝对差_s": f4(abs(float(q4_base["event"]) - stored_q4)),
    }
    metrics = {
        "实验名称": "Q3恒温恒湿平台开始时刻与Q4半径插值方法新增敏感性",
        "Q3基线平台开始_s": f4(7200.0), "Q3基线终止时间_h": f4(float(q3_base["event"]) / 3600.0),
        "Q4基线插值方法": "PCHIP", "Q4基线终止时间_h": f4(float(q4_base["event"]) / 3600.0),
        "基线重算一致性": consistency,
        "Q3最大绝对终止时间差_h": f4(max(abs(float(r["相对7200秒基线终止时间差_h"]) ) for r in q3_rows)),
        "Q4线性插值终止时间差_h": f4(float(q4_rows[1]["相对PCHIP终止时间差_h"])),
        "数值格式": "所有表格和图形标注保留四位小数",
        "运行环境": {"Python": platform.python_version(), "NumPy": np.__version__, "SciPy": __import__("scipy").__version__},
        "运行耗时_s": f4(time.time() - start),
    }
    (OUT / "Q3Q4新增敏感性_指标.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "Q3Q4新增敏感性_运行信息.json").write_text(json.dumps({
        "脚本": "src/q3_q4_additional_sensitivity.py", "随机性": "无",
        "输入": ["results/inputs.npz", "results/q3.npz", "results/q4.npz"],
        "Q3平台开始_s": [f4(v) for v in plateau_values], "Q4插值方法": q4_methods,
        "基线事件重算一致性": consistency,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plot(q3_rows, q4_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
