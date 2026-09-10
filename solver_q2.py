# -*- coding: utf-8 -*-
"""
2026 高教社杯 A题 问题2 求解器
整个烘干过程(预热平衡 + 恒温干燥) 变物性双向耦合
方程保持散度形式, 系数在面处插值(调和平均), 物性在局部(T,C)状态计算
  传热: rho(C)c_p(C) dT/dt = (1/r) d/dr( r k(C) dT/dr )
  传质: dC/dt = (1/r) d/dr( r D(C,T) dC/dr )
附录3:
  rho=650+128C ; c_p=1450+2736 C/(C+1) ; k=0.21+0.38 C/(C+1)
  D=2.4e-3 exp(-0.45/C) exp(-3850/T_K)
数值: 有限体积(FV) + 全隐式向后欧拉 + T/C 交替 Picard
输出: result2.xlsx (温度/水分浓度 两表, 10800行 x 21列, 1s间隔) + 表3/表4
"""
import numpy as np
import openpyxl
from pathlib import Path

R = 0.02
h  = 25.0
hm = 8e-7
T0 = 28.0
C0 = 2.55

def rho_f(C): return 650.0 + 128.0*C
def cp_f(C):  return 1450.0 + 2736.0*C/(C+1.0)
def k_f(C):   return 0.21 + 0.38*C/(C+1.0)
def D_f(C, TK): return 2.4e-3*np.exp(-0.45/C)*np.exp(-3850.0/TK)

def load_ambient(xlsx):
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    t  = np.array([r[0] for r in rows], float)
    Ta = np.array([r[1] for r in rows], float)
    Ca = np.array([r[2] for r in rows], float)
    return (lambda tt: np.interp(tt, t, Ta)), (lambda tt: np.interp(tt, t, Ca))

def build_grid(N):
    dr = R/N
    r  = np.arange(N+1)*dr
    rf = (np.arange(N)+0.5)*dr
    Af = 2*np.pi*rf
    AR = 2*np.pi*R
    V  = np.zeros(N+1)
    V[0] = np.pi*(dr/2)**2
    V[1:-1] = np.pi*(rf[1:]**2 - rf[:-1]**2)
    V[-1] = np.pi*(R**2 - rf[-1]**2)
    return dr, r, rf, Af, AR, V

def thomas(a, b, c, d):
    n = len(d)
    cc = np.empty(n); dd = np.empty(n)
    cc[0] = c[0]/b[0]; dd[0] = d[0]/b[0]
    for i in range(1, n):
        m = b[i] - a[i]*cc[i-1]
        cc[i] = c[i]/m
        dd[i] = (d[i] - a[i]*dd[i-1])/m
    x = np.empty(n); x[-1] = dd[-1]
    for i in range(n-2, -1, -1):
        x[i] = dd[i] - cc[i]*x[i+1]
    return x

def coupled_step(T, C, dt, Ta, Ca, N, Af, AR, V, dr, freeze=None, maxit=25, tol=1e-9):
    Tn = T.copy(); Cn = C.copy()
    for _ in range(maxit):
        Told = Tn.copy(); Cold = Cn.copy()
        # ---------- 传热 ----------
        if freeze is not None:
            rc = np.full(N+1, freeze[0]); kface = np.full(N, freeze[1])
        else:
            rho_ = rho_f(Cn); cp_ = cp_f(Cn); k_ = k_f(Cn)
            rc = rho_*cp_
            kface = 2*k_[:-1]*k_[1:]/(k_[:-1]+k_[1:]+1e-300)
        a = np.zeros(N+1); b = np.zeros(N+1); c = np.zeros(N+1); d = np.zeros(N+1)
        b[1:-1] = rc[1:-1]*V[1:-1]/dt + (Af[:-1]*kface[:-1]+Af[1:]*kface[1:])/dr
        a[1:-1] = -(Af[:-1]*kface[:-1])/dr
        c[1:-1] = -(Af[1:]*kface[1:])/dr
        d[1:-1] = rc[1:-1]*V[1:-1]/dt*T[1:-1]
        b[0] = rc[0]*V[0]/dt + Af[0]*kface[0]/dr
        c[0] = -(Af[0]*kface[0])/dr
        d[0] = rc[0]*V[0]/dt*T[0]
        b[-1] = rc[-1]*V[-1]/dt + Af[-1]*kface[-1]/dr + h*AR
        a[-1] = -(Af[-1]*kface[-1])/dr
        d[-1] = rc[-1]*V[-1]/dt*T[-1] + h*AR*Ta
        Tn = thomas(a, b, c, d)
        # ---------- 传质 ----------
        if freeze is not None:
            Dface = np.full(N, freeze[2])
        else:
            TK = Tn + 273.15
            D_ = D_f(Cn, TK)
            Dface = 2*D_[:-1]*D_[1:]/(D_[:-1]+D_[1:]+1e-300)
        a = np.zeros(N+1); b = np.zeros(N+1); c = np.zeros(N+1); d = np.zeros(N+1)
        b[1:-1] = V[1:-1]/dt + (Af[:-1]*Dface[:-1]+Af[1:]*Dface[1:])/dr
        a[1:-1] = -(Af[:-1]*Dface[:-1])/dr
        c[1:-1] = -(Af[1:]*Dface[1:])/dr
        d[1:-1] = V[1:-1]/dt*C[1:-1]
        b[0] = V[0]/dt + Af[0]*Dface[0]/dr
        c[0] = -(Af[0]*Dface[0])/dr
        d[0] = V[0]/dt*C[0]
        b[-1] = V[-1]/dt + Af[-1]*Dface[-1]/dr + hm*AR
        a[-1] = -(Af[-1]*Dface[-1])/dr
        d[-1] = V[-1]/dt*C[-1] + hm*AR*Ca
        Cn = thomas(a, b, c, d)
        if np.max(np.abs(Tn-Told)) < tol and np.max(np.abs(Cn-Cold)) < tol:
            break
    return Tn, Cn

def solve(N, dt, t_end, Tamb, Camb, out_times=None, freeze=None, track_every=None, full_every=None):
    dr, r, rf, Af, AR, V = build_grid(N)
    r_out = np.linspace(0, R, 21)
    T = np.full(N+1, T0); C = np.full(N+1, C0)
    out = {}
    out_set = set(int(round(float(x))) for x in out_times) if out_times is not None else None
    track = []
    full = {}
    t = 0.0
    step = 0
    while t < t_end - 1e-12:
        tnext = min(t+dt, t_end)
        T, C = coupled_step(T, C, tnext-t, Tamb(tnext), Camb(tnext), N, Af, AR, V, dr, freeze=freeze)
        t = tnext; step += 1
        if out_set is not None:
            key = int(round(t))
            if abs(t-key) < 1e-9 and key in out_set:
                out[key] = (np.interp(r_out, r, T), np.interp(r_out, r, C))
        if track_every is not None and (step % track_every == 0 or abs(t-t_end)<1e-9):
            track.append((t, Tamb(t), T[0], T[-1], C[0], C[-1], C.max(), D_f(C[0], T[0]+273.15)))
        if full_every is not None and (step % full_every == 0):
            full[t] = (T.copy(), C.copy())
    return out, T, C, track, full, r

# ================= 主流程 =================
if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    Tamb, Camb = load_ambient(base/"附件"/"附件1.xlsx")
    N = 400; dr = R/N

    # 参考(冻结)物性: (T=28℃, C=2.55)
    freeze_ref = (rho_f(C0)*cp_f(C0), k_f(C0), D_f(C0, T0+273.15))

    # ---- 主运行: 0-4h, dt=1s, 输出0-3h每秒 + 跟踪中心/表面 ----
    out_times = np.arange(1, 10801)      # 1..10800 s
    out, Tf, Cf, track, full, r = solve(N, 1.0, 14400.0, Tamb, Camb,
                                         out_times=out_times, track_every=1, full_every=1800)
    # 3h 全剖面(用于梯度分析)
    
    # 冻结物性 3h
    out_frz, Tfrz, Cfrz, _, _, _ = solve(N, 1.0, 10800.0, Tamb, Camb, out_times=[10800.0], freeze=freeze_ref)
    # 长时 72h, dt=30s, 跟踪每3600s
    _, _, _, trackL, _, _ = solve(N, 30.0, 259200.0, Tamb, Camb, track_every=120)

    # ============ 表3/表4 (6x5) ============
    ttab = [1800,3600,5400,7200,9000,10800]
    rtab = [0,0.005,0.01,0.015,0.02]
    print("表3 温度 (℃)")
    print("t/h    r=0     0.5     1.0     1.5     2.0")
    for tt in ttab:
        row = out[tt][0]
        print(f"{tt/3600:4.1f}  " + " ".join(f"{row[int(round(rr/0.001))]:8.4f}" for rr in rtab))
    print("\n表4 水分浓度 (kg/kg)")
    print("t/h    r=0     0.5     1.0     1.5     2.0")
    for tt in ttab:
        row = out[tt][1]
        print(f"{tt/3600:4.1f}  " + " ".join(f"{row[int(round(rr/0.001))]:8.4f}" for rr in rtab))

    # ============ 问题a: 中心/表面温度何时接近烘箱温度 ============
    print("\n[Q] 中心/表面温度何时接近烘箱温度?")
    tk = np.array([x[0] for x in track]); Ta_t = np.array([x[1] for x in track])
    Tc = np.array([x[2] for x in track]); Ts = np.array([x[3] for x in track])
    for thr in [1.0,0.5,0.2,0.1]:
        for name, field in [("表面", Ts), ("中心", Tc)]:
            gap = np.abs(field - Ta_t)
            ipeak = int(np.argmax(gap))
            idx = np.where((gap <= thr) & (tk > tk[ipeak]))[0]
            if len(idx):
                print(f"  |{name}-T_a|<={thr:g}℃ 首次(峰值后): t={tk[idx[0]]:.0f}s = {tk[idx[0]]/3600:.2f}h")
            else:
                print(f"  |{name}-T_a|<={thr:g}℃ 在4h内未达到")

    # ============ 问题b: 含水率梯度最大位置 ============
    print("\n[Q] 含水率梯度最大位置 (t=3h)?")
    C3 = full[10800.0][1]
    grad = np.abs(np.diff(C3)/dr)
    iarg = int(np.argmax(grad))
    print(f"  最大 |dC/dr| = {grad.max():.3f} (kg/kg)/m  位于 r ≈ {r[iarg]*100:.3f}~{r[iarg+1]*100:.3f} cm")
    for rr in [0.0,0.005,0.01,0.015,0.019,0.02]:
        ii = int(round(rr/dr))
        if 0 < ii < N:
            print(f"   r={rr*100:.1f}cm: dC/dr ≈ {(C3[ii+1]-C3[ii-1])/(2*dr):+.3f} (kg/kg)/m")

    # ============ 问题c: 变物性 vs 冻结物性 对3h分布影响 ============
    print("\n[Q] 变物性 vs 冻结物性(参考 T=28℃,C=2.55) 对3h分布影响?")
    rowV = out[10800.0]; rowF = out_frz[10800.0]
    for rr in rtab:
        ii = int(round(rr/0.001))
        dT = rowV[0][ii]-rowF[0][ii]; dC = rowV[1][ii]-rowF[1][ii]
        print(f"   r={rr*100:.1f}cm:  ΔT={dT:+.4f}℃,  ΔC={dC:+.4f} kg/kg")
    print(f"   最大 |ΔT| = {np.max(np.abs(rowV[0]-rowF[0])):.4f}℃ ; 最大 |ΔC| = {np.max(np.abs(rowV[1]-rowF[1])):.4f} kg/kg")

    # ============ 问题d: D 是否随含水率下降而大幅降低 ============
    print("\n[Q] 含水率下降是否使 D 大幅降低(解释后期干燥变慢)?")
    tl = np.array([x[0] for x in trackL]); Cmax = np.array([x[6] for x in trackL])
    Ccnt = np.array([x[4] for x in trackL]); Dcnt = np.array([x[7] for x in trackL])
    for i in [0, len(tl)//6, len(tl)//3, len(tl)//2, len(tl)-1]:
        print(f"   t={tl[i]/3600:5.1f}h: C_max={Cmax[i]:.4f}, C_center={Ccnt[i]:.4f}, D_center={Dcnt[i]:.3e} m^2/s")
    print(f"   D_center(0h)/D_center(72h) = {Dcnt[0]/Dcnt[-1]:.1f} 倍")

    # ============ 写 result2.xlsx ============
    outdir = base/"outputs"; outdir.mkdir(exist_ok=True)
    rv = np.linspace(0, R, 21); times = np.arange(1,10801)
    wb = openpyxl.Workbook(); ws1=wb.active; ws1.title="温度"; ws2=wb.create_sheet("水分浓度")
    header=["时间\\到药材中心的距离"]+[round(float(x)*100,1) for x in rv]
    ws1.append(header); ws2.append(header)
    for k,tt in enumerate(times, start=2):
        Tr,Cr = out[tt]
        ws1.cell(k,1,float(tt)); ws2.cell(k,1,float(tt))
        for j in range(21):
            ws1.cell(k,j+2,round(float(Tr[j]),4)); ws2.cell(k,j+2,round(float(Cr[j]),4))
    for ws in (ws1,ws2):
        for row in ws.iter_rows(min_row=2,min_col=2,max_row=10801,max_col=22):
            for cell in row: cell.number_format='0.0000'
    p = outdir/"result2.xlsx"; wb.save(p)
    print(f"\n已写出: {p}  ({ws1.max_row-1}行 x {ws1.max_column-1}列)")

