"""问题四：长度失水收缩的等价模型与对照计算。

将圆柱看成三维轴对称控制体，在材料坐标 xi=r/R(t) 下令
    R(t)=R0*s(t),  L(t)=L0*s(t)**gamma,
其中 s(t) 来自题给半径曲线。每个控制体的体积、径向面面积和端面面积
同时乘以 2*pi*L(t)。因此在浓度和温度方程中该因子完全约去，gamma
只改变总体积/总失水量的几何换算，不改变局部场和达到中心阈值的时间。
脚本仍显式组装带 L(t) 的三维控制体矩阵，以数值检验这个等价性。
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q4_length_shrink_compare"
RAW = np.load(ROOT / "results" / "inputs.npz")
AIR, RAD = RAW["air"], RAW["rad"]
RP = PchipInterpolator(RAD[:, 0], RAD[:, 1] / 100.0, extrapolate=False)
R0, L0, T0, C0 = 0.02, 0.25, 28.0, 2.55
N, DT, TOL = 201, 60.0, 1.0e-10


def radius(t: float) -> float:
    if t <= RAD[0, 0]:
        return float(RAD[0, 1] / 100.0)
    if t >= RAD[-1, 0]:
        return float(RAD[-1, 1] / 100.0)
    return float(RP(t))


def environment(t: float) -> tuple[float, float]:
    return (50.0, 0.05) if t > 7200.0 else (
        float(np.interp(t, AIR[:, 0], AIR[:, 1])),
        float(np.interp(t, AIR[:, 0], AIR[:, 2])),
    )


def geometry(n=N):
    x = np.arange(n) / (n - 1)
    dx = 1.0 / (n - 1)
    xf = (np.arange(n - 1) + 0.5) * dx
    v = np.empty(n)
    v[0] = dx * dx / 8.0
    v[1:-1] = 0.5 * ((x[1:-1] + dx / 2) ** 2 - (x[1:-1] - dx / 2) ** 2)
    v[-1] = 0.5 * (1.0 - (1.0 - dx / 2) ** 2)
    return x, xf, v, dx


def coeffs(T, C):
    return (
        760.0 + 90.0 * C,
        1850.0 + 2150.0 * C / (C + 1.0),
        0.12 + 0.20 * C / (C + 1.0),
        4.2e-4 * np.exp(-0.30 / np.maximum(C, 1.0e-12) - 3850.0 / (T + 273.15)),
    )


def thomas(lo, di, up, rhs):
    a, b, c, d = lo.copy(), di.copy(), up.copy(), rhs.copy()
    for i in range(1, len(b)):
        q = a[i] / b[i - 1]
        b[i] -= q * c[i - 1]
        d[i] -= q * d[i - 1]
    z = np.empty(len(b)); z[-1] = d[-1] / b[-1]
    for i in range(len(b) - 2, -1, -1):
        z[i] = (d[i] - c[i] * z[i + 1]) / b[i]
    return z


def assemble(old, T, C, dt, t, variable, gamma, geo):
    x, xf, v, dx = geo
    R = radius(t)
    L = L0 * (R / R0) ** gamma
    rho, cp, k, D = coeffs(T, C)
    kap = k if variable == "T" else D
    cap = rho * cp if variable == "T" else np.ones_like(C)
    beta = 25.0 if variable == "T" else 8.0e-7
    bnd = environment(t)[0 if variable == "T" else 1]
    # Three-dimensional axisymmetric factors: volume and all radial/side areas.
    fac = 2.0 * np.pi * L
    storage = fac * v * R**2 * cap / dt
    lo = np.zeros(N); up = np.zeros(N); di = storage.copy(); rhs = storage * old
    face = 0.5 * (kap[:-1] + kap[1:])
    # In the material-coordinate equation the radial operator is already
    # multiplied by R^2; after the common geometric factors are retained its
    # conductance is fac*k*xf/dx (the R factor is absorbed by the coordinate
    # transformation).  This is exactly the production Q4 discretization.
    conduct = fac * face * xf / dx
    up[0] = -conduct[0]; di[0] += conduct[0]
    for j in range(1, N - 1):
        lo[j] = -conduct[j - 1]; up[j] = -conduct[j]
        di[j] += conduct[j - 1] + conduct[j]
    lo[-1] = -conduct[-1]
    di[-1] += conduct[-1] + fac * R * beta
    rhs[-1] += fac * R * beta * bnd
    return lo, di, up, rhs


def run(gamma, end=None):
    production = np.load(ROOT / "results" / "q4.npz")
    if end is None:
        end = float(production["t"][-1])
    geo = geometry()
    T = np.full(N, T0); C = np.full(N, C0)
    ts = [0.0]; Ts = [T.copy()]; Cs = [C.copy()]
    event = np.nan
    for t in np.arange(DT, end + 1.0e-9, DT):
        Told, Cold = T.copy(), C.copy()
        Tn, Cn = T.copy(), C.copy()
        for _ in range(500):
            Tnext = thomas(*assemble(Told, Tn, Cn, DT, float(t), "T", gamma, geo))
            Cnext = thomas(*assemble(Cold, Tnext, Cn, DT, float(t), "C", gamma, geo))
            err = max(float(np.max(abs(Tnext - Tn))), float(np.max(abs(Cnext - Cn))))
            Tn, Cn = Tnext, Cnext
            if err < TOL:
                break
        else:
            raise RuntimeError(f"Picard未收敛: {t}")
        T, C = Tn, Cn
        prev, now = float(np.max(Cold)), float(np.max(C))
        if np.isnan(event) and prev >= 0.15 and now < 0.15:
            event = float(t - DT + DT * (prev - 0.15) / (prev - now))
        ts.append(float(t)); Ts.append(T.copy()); Cs.append(C.copy())
        if not np.isnan(event):
            break
    return {"t": np.asarray(ts), "T": np.asarray(Ts), "C": np.asarray(Cs), "event": event, "gamma": gamma}


def fmt(x):
    if isinstance(x, dict): return {k: fmt(v) for k, v in x.items()}
    if isinstance(x, list): return [fmt(v) for v in x]
    if isinstance(x, (float, np.floating)): return f"{0.0 if abs(float(x)) < 0.00005 else float(x):.4f}"
    if isinstance(x, (int, np.integer)): return int(x)
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prod = np.load(ROOT / "results" / "q4.npz")
    end = float(prod["t"][-1])
    runs = [run(g, end=end) for g in (0.0, 0.5, 1.0)]
    base = runs[0]
    rows = []
    for d in runs:
        g = d["gamma"]; ev = float(d["event"])
        # evaluate the measured radius at the common production endpoint
        s_end = radius(end) / R0
        L_end = L0 * s_end ** g
        V_end = np.pi * radius(end) ** 2 * L_end
        V0 = np.pi * R0**2 * L0
        # total water proxy is volume times radial mean C; no density closure invented
        # 以截面平均含水率乘当前体积构造总含水量代理；不引入题目未给的
        # 干密度，因此该量用于情景间相对比较而非绝对质量标定。
        mean_c_end = float(2.0 * np.dot(geometry()[2], d["C"][-1]))
        mean_c0 = C0
        water_proxy0 = np.pi * R0**2 * L0 * mean_c0
        water_proxy_end = V_end * mean_c_end
        rows.append({"伽马": g, "终止时间_s": ev, "终止时间_h": ev / 3600.0,
                     "末时刻半径_m": radius(end), "末时刻长度_m": L_end,
                     "末时刻总体积_m3": V_end, "长度比": L_end / L0,
                     "体积比": V_end / V0, "干固体密度比": V0 / V_end,
                     "终止时含水量代理": water_proxy_end,
                     "含水量代理比": water_proxy_end / water_proxy0,
                     "含水量代理损失比": 1.0 - water_proxy_end / water_proxy0,
                     "相对基线场最大差": float(np.max(abs(d["C"][:len(base["C"])] - base["C"]))),
                     "相对基线终止时间差_h": (ev - float(base["event"])) / 3600.0})
        np.savez_compressed(OUT / f"Q4_长度收缩_伽马_{g:.4f}.npz", t=d["t"], T=d["T"], C=d["C"], event=ev, gamma=g)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "Q4_长度失水收缩_对照结果.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    # Same-time center values demonstrate field equivalence directly.
    times = [18000.0, 43200.0, 86400.0, 129600.0, 172800.0, float(base["event"])]
    out = []
    for t in times:
        j = min(int(round(t / DT)), len(base["t"]) - 1)
        row = {"时间_s": float(base["t"][j]), "时间_h": float(base["t"][j] / 3600.0)}
        for d in runs:
            g = d["gamma"]; row[f"中心含水率_伽马_{g:.4f}"] = float(d["C"][j, 0])
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / "Q4_长度失水收缩_关键时刻.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    metrics = {
        "实验名称": "Q4长度失水收缩等价模型",
        "模型定义": "R(t)=R0*s(t)，L(t)=L0*s(t)^gamma；三维轴对称控制体的体积、径向面和侧边界面积均显式乘2*pi*L(t)",
        "伽马情景": [0.0, 0.5, 1.0], "节点数": N, "时间步_s": DT,
        "基线终止时间_h": float(base["event"] / 3600.0),
        "最大场差": max(r["相对基线场最大差"] for r in rows),
        "最大终止时间差_h": max(abs(r["相对基线终止时间差_h"]) for r in rows),
        "长度收缩仅改变": "总长度、总体积和体积含水量代理的几何换算；在一维径向局部方程中因子约去",
        "结论": "伽马=0、0.5、1的局部温度/含水率轨迹和干燥终止时间数值等价；长度效应不能从当前题给径向数据独立反演",
    }
    (OUT / "Q4_长度失水收缩_指标.json").write_text(json.dumps(fmt(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "Q4_长度失水收缩_说明.md").write_text(
        "# 问题四长度失水收缩等价模型\n\n"
        "取 $R(t)=R_0s(t)$，并以参数 $\\gamma$ 表示长度收缩 $L(t)=L_0s(t)^\\gamma$。三维轴对称控制体中，体积、径向传质面和侧壁传质面分别为 $2\\pi L R^2V_i$、$2\\pi LR A_i$ 和 $2\\pi LR$。在材料坐标径向方程中，三类项含有相同的 $2\\pi L$ 因子，离散方程约去该因子，故局部场只由半径曲线和径向边界通量决定。\n\n"
        "本次用 $\\gamma=0.0000,0.5000,1.0000$ 显式组装三维控制体并算至生产基线终点，节点数201、步长60.0000秒。结果见 `Q4_长度失水收缩_对照结果.csv` 与 `Q4_长度失水收缩_关键时刻.csv`；所有情景的终止时间与含水率轨迹在数值精度内一致，长度收缩只改变总体积和体积含水量代理的换算。若干固体质量守恒，则总水质量也保持情景间一致。\n",
        encoding="utf-8",
    )
    manifest = {"命令": "python3 src/q4_length_shrink_compare.py", "输入": ["results/inputs.npz", "results/q4.npz"], "随机性": "无", "Python": platform.python_version(), "NumPy": np.__version__}
    (OUT / "Q4_长度失水收缩_复现清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(fmt(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
