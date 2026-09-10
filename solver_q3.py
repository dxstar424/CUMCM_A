# -*- coding: utf-8 -*-
"""
2026 高教社杯 A题 问题3 求解器
在 Q2 的附录3 变物性双向耦合模型上长时间积分, 寻找所有节点 C<0.15 的首次时刻。
输出: 表5(每6h x 距离0.5cm, 末行=烘干结束时间) + result3.xlsx(每60s x 0.1cm 水分浓度, 单表)
"""
import numpy as np
import openpyxl
from pathlib import Path
from solver_q2 import (R, h, hm, T0, C0, rho_f, cp_f, k_f, D_f,
                       load_ambient, build_grid, coupled_step)

base = Path(__file__).resolve().parent
outdir = base/"outputs"; outdir.mkdir(exist_ok=True)

Tamb, Camb = load_ambient(base/"附件"/"附件1.xlsx")

def run_q3(N=400, dt=30.0, t_end=324000.0):
    """主求解: dt 为内部时间步, 每 60s 保存一次全剖面; 返回结果字典。"""
    dr, r, rf, Af, AR, V = build_grid(N)
    T = np.full(N+1, T0); C = np.full(N+1, C0)
    times = np.arange(60, int(t_end)+1, 60, dtype=float)
    profiles = np.zeros((len(times), N+1))     # 仅存水分浓度
    t = 0.0; step = 0; iout = 0
    t_dry = None
    t_dry_grid = None
    while t < t_end - 1e-12:
        tnext = min(t+dt, t_end)
        T, C = coupled_step(T, C, tnext-t, Tamb(tnext), Camb(tnext),
                            N, Af, AR, V, dr)
        t = tnext; step += 1
        # 每 60s 输出(容差对齐)
        while iout < len(times) and t >= times[iout] - 1e-9:
            profiles[iout] = C
            if C.max() < 0.15 and t_dry_grid is None:
                t_dry_grid = times[iout]
                # 更细的首次时刻在调用端再定位
            iout += 1
        # 内部首次判据(dt 精度)
        if C.max() < 0.15 and t_dry is None:
            t_dry = t
    return r, times, profiles, t_dry, t_dry_grid

def refine_tdry(N, dt_coarse, t_start, T_start, C_start, dt_fine=1.0, window=120.0):
    """从 t_start 状态出发, 用 1s 步长精确找首次 Cmax<0.15。"""
    dr, r, rf, Af, AR, V = build_grid(N)
    T = T_start.copy(); C = C_start.copy(); t = t_start
    while t < t_start + window - 1e-12:
        tnext = min(t+dt_fine, t_start+window)
        T, C = coupled_step(T, C, tnext-t, Tamb(tnext), Camb(tnext),
                            N, Af, AR, V, dr)
        t = tnext
        if C.max() < 0.15:
            return t, C
    return None, None

if __name__ == "__main__":
    N = 400
    # ---- 主运行: dt=30s, 存每60s剖面 ----
    r, times, profiles, t_dry30, t_dry_grid = run_q3(N, dt=30.0, t_end=324000.0)
    # ---- 60s 输出网格上的首次达标 ----
    # (run_q3 已定位 t_dry_grid; 若 90h 内没找到则异常)
    assert t_dry_grid is not None, "90h 内未达到 C<0.15, 请增大 t_end"

    # ---- 1s 精确定位: 从 t_dry_grid-60s 的网格前一行状态出发 ----
    # 重新跑到 t_dry_grid-120s, 保存状态, 再做1s精修
    dr, r_, rf, Af, AR, V = build_grid(N)
    T = np.full(N+1, T0); C = np.full(N+1, C0)
    t = 0.0
    t_restart = max(0.0, t_dry_grid - 120.0)
    # 粗步到 restart
    while t < t_restart - 1e-12:
        tnext = min(t+30.0, t_restart)
        T, C = coupled_step(T, C, tnext-t, Tamb(tnext), Camb(tnext),
                            N, Af, AR, V, dr)
        t = tnext
    t_dry_1s, C_end = refine_tdry(N, 30.0, t, T.copy(), C.copy(), dt_fine=1.0, window=180.0)
    assert t_dry_1s is not None, "1s 精修未找到达标时刻"

    # ---- 表5: 每6h + 末行烘干结束时间 ----
    rtab = [0, 0.005, 0.01, 0.015, 0.02]
    # 由 profiles 取 6h 整点
    t_hours_6 = np.arange(6, int(t_dry_grid/3600)//6*6 + 1, 6)
    def interp_row(t_sec):
        idx = int(round(t_sec/60)) - 1   # times[i] = 60*(i+1)
        Cprof = profiles[idx]
        return [float(np.interp(rr, r, Cprof)) for rr in rtab]
    print("表5 药材烘干过程的水分浓度 (kg/kg)")
    print("t/h     r=0      0.5      1.0      1.5      2.0")
    table5_rows = []
    for hh in t_hours_6:
        row = interp_row(hh*3600)
        table5_rows.append((hh, row))
        print(f"{hh:6.1f}  " + " ".join(f"{v:8.4f}" for v in row))
    # 末行: 烘干结束时间(用1s精修值, 小时, 4位小数)
    t_end_h = t_dry_1s/3600.0
    row_end = interp_row(t_dry_grid)   # 末行水分用 60s 网格最后达标行
    table5_rows.append((round(t_end_h,4), row_end))
    print(f"{t_end_h:6.4f}  " + " ".join(f"{v:8.4f}" for v in row_end) + "   <- 烘干结束时间")

    # ---- 数值检查: 单调性(中心=最大) ----
    Cmax_center_diff = []
    for i in range(0, len(times), max(1,len(times)//20)):
        prof = profiles[i]
        if np.argmax(prof) != 0:
            Cmax_center_diff.append((times[i], float(prof.max()-prof[0])))
    print("\n[单调性检查] max(C)-C(0) 全程应为0; 抽样非零项:", Cmax_center_diff[:5] or "无(中心始终最大)")

    # ---- dt 无关性快速复核: dt=60 定位 t_dry_grid ----
    r2, times2, prof2, t_dry60, t_dry_grid60 = run_q3(N, dt=60.0, t_end=t_dry_grid+120.0)
    print(f"\n[dt无关性] dt=30s: t_dry(30s)={t_dry30/3600:.6f}h, t_dry_grid={t_dry_grid/3600:.6f}h, 1s精修={t_dry_1s/3600:.6f}h")
    print(f"          dt=60s: t_dry(60s)={t_dry60/3600:.6f}h, t_dry_grid={t_dry_grid60/3600:.6f}h")

    # ---- 写 result3.xlsx: 每60s x 0.1cm 单表水分浓度 ----
    rv = np.linspace(0, R, 21)
    nrows = int(t_dry_grid/60)          # 60,120,...,t_dry_grid 共 nrows 行
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "水分浓度"
    header = ["时间\\到药材中心的距离"] + [round(float(x)*100,1) for x in rv]
    ws.append(header)
    for i in range(nrows):
        tt = times[i]
        Cprof = profiles[i]
        C21 = np.interp(rv, r, Cprof)
        ws.cell(i+2, 1, float(tt))
        for j in range(21):
            ws.cell(i+2, j+2, round(float(C21[j]), 4))
    for row in ws.iter_rows(min_row=2, min_col=2, max_row=nrows+1, max_col=22):
        for cell in row:
            cell.number_format = '0.0000'
    p = outdir/"result3.xlsx"; wb.save(p)
    print(f"\n已写出: {p}  ({nrows}行 x 21列, 时间 60..{t_dry_grid:.0f}s 步长60s)")

    # ---- 关键摘要 ----
    print("\n===== Q3 结果摘要 =====")
    print(f"烘干所需时间(1s精修): t_dry = {t_dry_1s:.0f} s = {t_end_h:.6f} h")
    print(f"result3/表5末行采用 60s 输出网格: t_dry_grid = {t_dry_grid:.0f} s = {t_dry_grid/3600:.6f} h")
    print(f"结束时刻中心 C(0) = {C_end[0]:.6f} kg/kg (<0.15 达标)")
