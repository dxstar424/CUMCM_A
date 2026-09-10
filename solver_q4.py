# -*- coding: utf-8 -*-
"""
2026 高教社杯 A题 问题4 求解器
收缩圆柱(动边界): 归一化物质坐标 xi = r/R(t) 上的 FVM
方程(拉格朗日/物质形式, 无对流项, 不需要 Rdot):
  rho*cp * dT/dt = (1/(xi R^2)) d/dxi( xi k dT/dxi )
  dC/dt         = (1/(xi R^2)) d/dxi( xi D dC/dxi )
附录4物性; R(t) 由附件2插值; 环境由附件1插值(>4h 恒末值)。
输出 result4.xlsx(每60s, 物理距离0.1cm, 单表水分) + 表6。
"""
import numpy as np
import openpyxl
from pathlib import Path

base = Path(__file__).resolve().parent
outdir = base/"outputs"; outdir.mkdir(exist_ok=True)

R0 = 0.02
h  = 25.0
hm = 8e-7
T0 = 28.0
C0 = 2.55
N  = 400

def rho4(C): return 760.0 + 90.0*C
def cp4(C):  return 1850.0 + 2150.0*C/(C+1.0)
def k4(C):   return 0.12 + 0.20*C/(C+1.0)
def D4(C, TK): return 4.2e-4*np.exp(-0.30/C)*np.exp(-3850.0/TK)

def load_ambient(xlsx):
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    t  = np.array([r[0] for r in rows], float)
    Ta = np.array([r[1] for r in rows], float)
    Ca = np.array([r[2] for r in rows], float)
    return (lambda tt: np.interp(tt, t, Ta)), (lambda tt: np.interp(tt, t, Ca))

def load_radius(xlsx):
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    rows = list(wb.active.iter_rows(values_only=True))[1:]
    t = np.array([r[0] for r in rows], float)
    R = np.array([r[1] for r in rows], float)
    R = R/100.0   # cm -> m
    return (lambda tt: np.interp(tt, t, R))

Tamb, Camb = load_ambient(base/"附件"/"附件1.xlsx")
Rf = load_radius(base/"附件"/"附件2.xlsx")

# 固定 xi 网格几何
dxi = 1.0/N
xi  = np.arange(N+1)*dxi
xif = (np.arange(N)+0.5)*dxi
s = np.zeros(N+1)
s[0] = dxi*dxi/8.0
s[1:-1] = xi[1:-1]*dxi
s[-1] = (dxi - dxi*dxi/4.0)/2.0

def thomas(a,b,c,d):
    n=len(d); cc=np.empty(n); dd=np.empty(n)
    cc[0]=c[0]/b[0]; dd[0]=d[0]/b[0]
    for i in range(1,n):
        m=b[i]-a[i]*cc[i-1]
        cc[i]=c[i]/m; dd[i]=(d[i]-a[i]*dd[i-1])/m
    x=np.empty(n); x[-1]=dd[-1]
    for i in range(n-2,-1,-1):
        x[i]=dd[i]-cc[i]*x[i+1]
    return x

def coupled_step(T, C, dt, Ta, Ca, R, maxit=25, tol=1e-9):
    Tn=T.copy(); Cn=C.copy()
    vol=R*R*s
    for _ in range(maxit):
        Told=Tn.copy(); Cold=Cn.copy()
        # 传热
        rc=rho4(Cn)*cp4(Cn)
        k_=k4(Cn)
        kface=2*k_[:-1]*k_[1:]/(k_[:-1]+k_[1:]+1e-300)
        gT=xif*kface/dxi
        a=np.zeros(N+1); b=np.zeros(N+1); c=np.zeros(N+1); d=np.zeros(N+1)
        b[1:-1]=rc[1:-1]*vol[1:-1]/dt + gT[:-1]+gT[1:]
        a[1:-1]=-gT[:-1]; c[1:-1]=-gT[1:]
        d[1:-1]=rc[1:-1]*vol[1:-1]/dt*T[1:-1]
        b[0]=rc[0]*vol[0]/dt + gT[0]; c[0]=-gT[0]
        d[0]=rc[0]*vol[0]/dt*T[0]
        b[-1]=rc[-1]*vol[-1]/dt + gT[-1] + R*h
        a[-1]=-gT[-1]
        d[-1]=rc[-1]*vol[-1]/dt*T[-1] + R*h*Ta
        Tn=thomas(a,b,c,d)
        # 传质
        TK=Tn+273.15
        D_=D4(Cn,TK)
        Dface=2*D_[:-1]*D_[1:]/(D_[:-1]+D_[1:]+1e-300)
        gC=xif*Dface/dxi
        a=np.zeros(N+1); b=np.zeros(N+1); c=np.zeros(N+1); d=np.zeros(N+1)
        b[1:-1]=vol[1:-1]/dt + gC[:-1]+gC[1:]
        a[1:-1]=-gC[:-1]; c[1:-1]=-gC[1:]
        d[1:-1]=vol[1:-1]/dt*C[1:-1]
        b[0]=vol[0]/dt + gC[0]; c[0]=-gC[0]
        d[0]=vol[0]/dt*C[0]
        b[-1]=vol[-1]/dt + gC[-1] + R*hm
        a[-1]=-gC[-1]
        d[-1]=vol[-1]/dt*C[-1] + R*hm*Ca
        Cn=thomas(a,b,c,d)
        if np.max(np.abs(Tn-Told))<tol and np.max(np.abs(Cn-Cold))<tol:
            break
    return Tn,Cn

def run(dt, t_end, store_every=60.0):
    T=np.full(N+1,T0); C=np.full(N+1,C0)
    times=[]; Cprofs=[]; states=[]
    t=0.0; step=0; next_store=store_every
    t_dry=None
    while t < t_end-1e-12:
        tnext=min(t+dt,t_end)
        R=Rf(tnext)
        T,C=coupled_step(T,C,tnext-t,Tamb(tnext),Camb(tnext),R)
        t=tnext; step+=1
        if t>=next_store-1e-9:
            times.append(t); Cprofs.append(C.copy()); states.append((T.copy(),C.copy()))
            next_store+=store_every
        if t_dry is None and C.max()<0.15:
            t_dry=t
    return times, Cprofs, states, t_dry

def C_at_r(Cprof, R, r):
    """从 xi 节点剖面插值到物理半径 r; r>R 返回 np.nan"""
    if r>R+1e-12:
        return np.nan
    xir=r/R
    return float(np.interp(xir, xi, Cprof))

if __name__ == "__main__":
    print("运行 Q4 主求解 dt=60s ...")
    times, Cprofs, states, t_dry60 = run(60.0, 432000.0, store_every=60.0)
    print("t_dry(dt=60s):", t_dry60, "s =", None if t_dry60 is None else t_dry60/3600, "h")
    # 60s网格首次达标
    t_dry_grid=None
    for tt,Cp in zip(times,Cprofs):
        if Cp.max()<0.15:
            t_dry_grid=tt; break
    print("t_dry_grid:", t_dry_grid, "s =", t_dry_grid/3600 if t_dry_grid else None, "h")
    # 1s精修: 从 t_dry_grid-60 的状态出发
    assert t_dry_grid is not None
    i_restart = int(t_dry_grid/60)-2   # t_dry_grid-120
    if i_restart<0: i_restart=0
    t0=times[i_restart]; T,C = states[i_restart]
    t=t0; t_dry_1s=None
    while t < t0+180.0-1e-12:
        tnext=min(t+1.0, t0+180.0)
        R=Rf(tnext)
        T,C=coupled_step(T,C,tnext-t,Tamb(tnext),Camb(tnext),R)
        t=tnext
        if C.max()<0.15:
            t_dry_1s=t; Cend=C.copy(); break
    print("t_dry(1s精修):", t_dry_1s, "s =", t_dry_1s/3600, "h")

    # 表6: 每6h + 末行
    rtab=[0,0.005,0.01,0.015,0.02]
    hmax = int(t_dry_grid/3600)//6*6
    def row_at(tt):
        i=int(round(tt/60))-1
        Cp=Cprofs[i]; R=Rf(tt)
        return [C_at_r(Cp,R,rr) for rr in rtab]+[float(Cp[-1])]
    print("\n表6 水分浓度(kg/kg) 列: 0,0.5,1.0,1.5,2.0,表面")
    for hh in range(6, hmax+1, 6):
        vals=row_at(hh*3600)
        print(f"{hh:6.1f}  " + " ".join(("    -" if np.isnan(v) else f"{v:7.4f}") for v in vals))
    vals=row_at(t_dry_grid)
    print(f"{t_dry_1s/3600:6.4f}  " + " ".join(("    -" if np.isnan(v) else f"{v:7.4f}") for v in vals))

    # 单调性
    bad=[(tt,float(Cp.max()-Cp[0])) for tt,Cp in zip(times,Cprofs) if np.argmax(Cp)!=0]
    print("\n单调性检查(中心应为最大):", bad[:3] if bad else "OK")

    # 写 result4.xlsx
    rv=[round(0.001*j,3) for j in range(21)]  # 0..0.020 m
    wb=openpyxl.Workbook(); ws=wb.active; ws.title="水分浓度"
    header=["时间\\到药材中心的距离"]+[round(r*100,1) for r in rv]+["药材表面"]
    ws.append(header)
    nrows=int(t_dry_grid/60)
    for i in range(nrows):
        tt=times[i]; Cp=Cprofs[i]; R=Rf(tt)
        ws.cell(i+2,1,float(tt))
        for j,r in enumerate(rv):
            v=C_at_r(Cp,R,r)
            if not np.isnan(v):
                ws.cell(i+2,j+2,round(v,4))
        ws.cell(i+2,23,round(float(Cp[-1]),4))
    for row in ws.iter_rows(min_row=2,min_col=2,max_row=nrows+1,max_col=23):
        for cell in row:
            if cell.value is not None:
                cell.number_format='0.0000'
    p=outdir/"result4.xlsx"; wb.save(p)
    print(f"\n已写出: {p}  ({nrows}行 x {ws.max_column-1}列, 时间60..{t_dry_grid:.0f}s)")
    print(f"\nQ4 结果: t_dry = {t_dry_1s:.0f}s = {t_dry_1s/3600:.4f}h ; 60s网格={t_dry_grid/3600:.4f}h")

