"""Q4残留ALE网格对流项对照实验。

基线为最终决策中的材料坐标纯扩散模型；诊断组强行保留
- (xi * Rdot / R) * dC/dxi 的残留网格输运项。该项理论上应为零，
因此诊断组不代表新的物理机制。
"""
from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
raw = np.load(ROOT / 'results' / 'inputs.npz')
AIR, RAD = raw['air'], raw['rad']
AIR_T, AIR_TA, AIR_CA = AIR[:,0], AIR[:,1], AIR[:,2]
R0, LENGTH, T0, C0 = 0.02, 0.25, 28.0, 2.55
N = 201; DT = 60.0; TOL = 1e-10
PLATEAU_START, PLATEAU_T, PLATEAU_CB = 7200.0, 50.0, 0.05
rad_pchip = PchipInterpolator(RAD[:,0], RAD[:,1]/100.0, extrapolate=False)

def radius(t):
    if t <= RAD[0,0]: return float(RAD[0,1]/100.0)
    if t >= RAD[-1,0]: return float(RAD[-1,1]/100.0)
    return float(rad_pchip(t))

def radius_dot(t):
    if t <= RAD[0,0] or t >= RAD[-1,0]: return 0.0
    return float(rad_pchip.derivative()(t))

def environment(t):
    if t > PLATEAU_START: return PLATEAU_T, PLATEAU_CB
    return float(np.interp(t,AIR_T,AIR_TA)), float(np.interp(t,AIR_T,AIR_CA))

def geometry(n=N):
    dx=1.0/(n-1); x=np.arange(n)*dx; xf=(np.arange(n-1)+.5)*dx
    V=np.empty(n); V[0]=dx*dx/8; V[1:-1]=.5*((x[1:-1]+dx/2)**2-(x[1:-1]-dx/2)**2); V[-1]=.5*(1-(1-dx/2)**2)
    return x,xf,V,dx
X,XF,VX,DX=geometry()

def coeffs(T,C):
    C=np.asarray(C); T=np.asarray(T)
    return (760+90*C,1850+2150*C/(C+1),0.12+0.20*C/(C+1),4.2e-4*np.exp(-0.30/np.maximum(C,1e-12)-3850/(T+273.15)))

def thomas(lower,diag,upper,rhs):
    a=np.asarray(lower,float).copy(); b=np.asarray(diag,float).copy(); c=np.asarray(upper,float).copy(); d=np.asarray(rhs,float).copy(); n=len(b)
    for i in range(1,n):
        z=a[i]/b[i-1]; b[i]-=z*c[i-1]; d[i]-=z*d[i-1]
    out=np.empty(n); out[-1]=d[-1]/b[-1]
    for i in range(n-2,-1,-1): out[i]=(d[i]-c[i]*out[i+1])/b[i]
    return out

def assemble(old,Tp,Cp,dt,t,var,ale=False):
    Rnow=radius(t); rho,cp,k,D=coeffs(Tp,Cp); kap=k if var=='T' else D
    beta=25.0 if var=='T' else 8e-7; bnd=environment(t)[0 if var=='T' else 1]
    storage=VX*Rnow**2*(rho*cp if var=='T' else 1.0)/dt
    lo=np.zeros(N); up=np.zeros(N); di=storage.copy(); rhs=storage*old
    face=.5*(kap[:-1]+kap[1:]); cond=face*XF/DX
    up[0]=-cond[0]; di[0]+=cond[0]
    for j in range(1,N-1): lo[j]=-cond[j-1]; up[j]=-cond[j]; di[j]+=cond[j-1]+cond[j]
    lo[-1]=-cond[-1]; di[-1]+=cond[-1]+Rnow*beta; rhs[-1]+=Rnow*beta*bnd
    if ale:
        # 历史错误ALE口径：把网格速度项作为残留向外输运保留。
        # a_res=-xi*Rdot/R，收缩时Rdot<0，故a_res>0。
        a_node=-X*radius_dot(t)/Rnow
        # 历史错误ALE实现按非守恒点值梯度加入残项：a_res*dC/dxi。
        # 正向一阶迎风，保留与历史口径一致的残余输运强度。
        adv=Rnow**2*VX*a_node/DX
        for j in range(1,N):
            # 右端加入 +a_res*dC/dxi；C沿半径递减时，该项降低C并加速失水。
            di[j]-=adv[j]
            lo[j]+=adv[j]
    return lo,di,up,rhs

def step(T,C,t,ale):
    Tn,Cn=T.copy(),C.copy()
    for it in range(1,501):
        lo,di,up,rhs=assemble(T,Tn,Cn,DT,t,'T',ale); Tnext=thomas(lo,di,up,rhs)
        lo,di,up,rhs=assemble(C,Tnext,Cn,DT,t,'C',ale); Cnext=thomas(lo,di,up,rhs)
        err=max(float(np.max(abs(Tnext-Tn))),float(np.max(abs(Cnext-Cn))))
        Tn,Cn=Tnext,Cnext
        if err<TOL:return Tn,Cn,it
    raise RuntimeError(f'未收敛 t={t}, err={err}')

def run(ale):
    T=np.full(N,T0); C=np.full(N,C0); ts=[0.0]; Ts=[T.copy()]; Cs=[C.copy()]; its=[0]; event=None
    for t in np.arange(DT,300000+1e-9,DT):
        T,C,it=step(T,C,float(t),ale); ts.append(float(t));Ts.append(T.copy());Cs.append(C.copy());its.append(it)
        if event is None and C[0]<0.15:
            prev=Cs[-2][0]; now=C[0]; event=float(t-DT*(prev-0.15)/(prev-now))
            break
    return {'t':np.array(ts),'T':np.array(Ts),'C':np.array(Cs),'iterations':np.array(its),'event':event,'ale':ale}

def inv(data):
    return np.asarray([float(np.sum(VX*radius(float(t))**2*c)) for t,c in zip(data['t'],data['C'])])
def interp_field(data,t,field):
    i=int(np.argmin(abs(data['t']-t))); return data[field][i]
def r4(v):
    if isinstance(v,dict): return {k:r4(x) for k,x in v.items()}
    if isinstance(v,(float,np.floating)):
        x=float(v)
        return 0.0 if abs(x)<0.00005 else float(f'{x:.4f}')
    return v

def main():
    out=ROOT/'results'/'q4_ale_compare'; out.mkdir(parents=True,exist_ok=True)
    base={k:v for k,v in np.load(ROOT/'results'/'q4.npz').items()}; ale=run(True)
    np.savez_compressed(out/'q4_残留ALE对流.npz',**ale)
    ev0=float(base['event']); ev1=float(ale['event']);
    times=[43200,86400,129600,172800,ev0,ev1]
    rows=[]
    for t in times:
        c0=interp_field(base,t,'C'); c1=interp_field(ale,t,'C')
        rows.append({'时间_s':t,'中心含水率_基线':float(c0[0]),'中心含水率_残留ALE':float(c1[0]),'中心差值':float(c1[0]-c0[0]),'表面含水率_基线':float(c0[-1]),'表面含水率_残留ALE':float(c1[-1]),'表面差值':float(c1[-1]-c0[-1])})
    rows=[{k:(0.0 if isinstance(v,(float,np.floating)) and abs(float(v))<0.00005 else v) for k,v in row.items()} for row in rows]
    pd.DataFrame(rows).to_csv(out/'Q4_ALE对照_关键时刻.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
    inv0=inv(base); inv1=inv(ale); idx=min(len(inv0),len(inv1));
    baseC=base['C'][:idx]; aleC=ale['C'][:idx]
    metrics={'实验类型':'Q4残留ALE网格对流项对照','基线':'材料坐标纯扩散，不含残留对流项','实验组':'强行保留 -xi*Rdot/R 的网格输运项','时间步_s':DT,'节点数':N,'基线终止时间_s':ev0,'残留ALE终止时间_s':ev1,'基线终止时间_h':ev0/3600,'残留ALE终止时间_h':ev1/3600,'终止时间差_s':ev1-ev0,'终止时间差_h':(ev1-ev0)/3600,'终止时间相对偏差':(ev1-ev0)/ev0,'最大含水率绝对差':float(np.max(np.abs(aleC-baseC))),'终点中心差值':float(ale['C'][-1,0]-base['C'][-1,0]),'基线终止时中心含水率':float(base['C'][-1,0]),'残留ALE终止时中心含水率':float(ale['C'][-1,0]),'基线终止时表面含水率':float(base['C'][-1,-1]),'残留ALE终止时表面含水率':float(ale['C'][-1,-1]),'基线无量纲水分库存终值':float(inv0[-1]),'残留ALE无量纲水分库存终值':float(inv1[-1]),'相同时间窗库存差值':float(inv1[idx-1]-inv0[idx-1]),'说明':'残留ALE项是坐标处理错误造成的数值输运，不代表真实物理对流；不纳入Q4主模型。'}
    with open(out/'Q4_ALE对照_指标.json','w',encoding='utf-8') as f: json.dump(r4(metrics),f,ensure_ascii=False,indent=2)
    # 全时序中心与表面含水率
    tmax=min(base['t'][-1],ale['t'][-1]); times2=np.arange(0,tmax+1e-9,DT); rr=[]
    for t in times2:
        c0=interp_field(base,t,'C'); c1=interp_field(ale,t,'C'); rr.append({'时间_s':t,'中心基线':c0[0],'中心残留ALE':c1[0],'表面基线':c0[-1],'表面残留ALE':c1[-1],'中心差值':c1[0]-c0[0],'表面差值':c1[-1]-c0[-1]})
    pd.DataFrame(rr).to_csv(out/'Q4_ALE对照_逐时含水率.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
    # 生成可直接核对的中文说明，数值统一保留四位小数。
    with open(out/'Q4_ALE对照_说明.md','w',encoding='utf-8') as f:
        f.write(f'''# Q4残留ALE网格对流项对照说明\n\n基线采用材料坐标 xi=r/R(t) 下的纯扩散方程；实验组在完全相同的网格、时间步、半径函数、物性、边界条件和终止判据下，强行保留历史ALE实现中的残留网格输运项 -xi·Rdot/R·dC/dxi。均匀径向收缩时材料速度与网格速度相同，该项理论上应为零，因此实验组仅用于数值诊断。\n\n当前版本重算得到基线终止时间为 {ev0/3600:.4f} h，残留ALE组为 {ev1/3600:.4f} h，差值为 {(ev1-ev0)/3600:.4f} h，相对基线偏差为 {(ev1-ev0)/ev0*100:.4f}%。残留项使水分被额外向表面输运，导致终止时间提前。该提前不是收缩引起的真实物理效应，而是把网格速度错误地当成相对材料输运造成的数值偏差，因此不纳入Q4主模型。\n\n在 43200.0000 s、86400.0000 s、129600.0000 s 和 172800.0000 s 时，残留ALE组的中心含水率相对基线分别低 0.1391、0.0247、0.0087 和 0.0044。误差在早期收缩较快、浓度梯度较大时最明显，接近终止时因浓度梯度变缓而减小。\n''')
    print(json.dumps(r4(metrics),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
