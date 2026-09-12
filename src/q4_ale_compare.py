"""Q4材料坐标与错误保留网格导数项的受控对照。

正确ALE相对速度为(v_s-xi*Rdot)/R=0；诊断在右端故意保留
b*dU/dxi，b=-xi*Rdot/R。默认内点中心差分、表面Robin导数闭合。
"""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'q4_ale_compare'
RAW=np.load(ROOT/'results'/'inputs.npz')
AIR,RAD=RAW['air'],RAW['rad']
RP=PchipInterpolator(RAD[:,0],RAD[:,1]/100,extrapolate=False)
RPD=RP.derivative()
T0,C0=28.0,2.55

def radius(t):
    return float(RAD[0,1]/100 if t<=RAD[0,0] else RAD[-1,1]/100 if t>=RAD[-1,0] else RP(t))
def radius_dot(t):
    return float(RPD(t)) if RAD[0,0]<t<RAD[-1,0] else 0.0
def environment(t):
    return (50.0,0.05) if t>7200 else (float(np.interp(t,AIR[:,0],AIR[:,1])),float(np.interp(t,AIR[:,0],AIR[:,2])))
def geometry(n):
    x=np.arange(n)/(n-1); dx=1/(n-1); xf=(np.arange(n-1)+.5)*dx
    v=np.empty(n); v[0]=dx*dx/8; v[1:-1]=.5*((x[1:-1]+dx/2)**2-(x[1:-1]-dx/2)**2); v[-1]=.5*(1-(1-dx/2)**2)
    return x,xf,v,dx

def coeffs(T,C):
    return (760+90*C,1850+2150*C/(C+1),.12+.20*C/(C+1),4.2e-4*np.exp(-.30/np.maximum(C,1e-12)-3850/(T+273.15)))
def thomas(lo,di,up,rhs):
    a=lo.copy(); b=di.copy(); c=up.copy(); d=rhs.copy()
    for j in range(1,len(b)):
        z=a[j]/b[j-1]; b[j]-=z*c[j-1]; d[j]-=z*d[j-1]
    u=np.empty(len(b)); u[-1]=d[-1]/b[-1]
    for j in range(len(b)-2,-1,-1):u[j]=(d[j]-c[j]*u[j+1])/b[j]
    return u

def assemble(old,T,C,dt,t,var,active,geo,scheme='centered',closed=False):
    x,xf,v,dx=geo; n=len(x); R=radius(t)
    rho,cp,k,D=coeffs(T,C); cap=rho*cp if var=='T' else np.ones(n)
    kap=k if var=='T' else D; beta=(25.0 if var=='T' else 8e-7) if not closed else 0.0
    bnd=environment(t)[0 if var=='T' else 1]
    storage=v*R**2*cap/dt
    lo=np.zeros(n); up=np.zeros(n); di=storage.copy(); rhs=storage*old
    g=.5*(kap[:-1]+kap[1:])*xf/dx
    up[0]=-g[0];di[0]+=g[0]
    for j in range(1,n-1):
        lo[j]=-g[j-1];up[j]=-g[j];di[j]+=g[j-1]+g[j]
    lo[-1]=-g[-1];di[-1]+=g[-1]+R*beta;rhs[-1]+=R*beta*bnd
    source_weight=np.zeros(n)
    if active:
        b=-x*radius_dot(t)/R
        source_weight=v*R**2*cap*b
        if scheme=='centered':
            lo[1:-1]+=source_weight[1:-1]/(2*dx)
            up[1:-1]-=source_weight[1:-1]/(2*dx)
        elif scheme=='upwind':
            # RHS b*U_xi 等价左端速度 -b；收缩时用前向迎风。
            di[1:-1]+=source_weight[1:-1]/dx
            up[1:-1]-=source_weight[1:-1]/dx
        else:raise ValueError(scheme)
        # 表面 U_xi=-R*beta/kap*(Us-Ua)，与原Robin边界一致。
        sink=source_weight[-1]*R*beta/kap[-1]
        di[-1]+=sink;rhs[-1]+=sink*bnd
    return lo,di,up,rhs

def gradient(u,t,var,geo,scheme,closed=False):
    x,xf,v,dx=geo; du=np.zeros_like(u)
    if scheme=='centered':du[1:-1]=(u[2:]-u[:-2])/(2*dx)
    else:du[1:-1]=(u[2:]-u[1:-1])/dx
    # 此函数只用于水分收支；D需要调用方提供表面闭合。
    return du

def run(active,dt=60.0,n=201,end=184020.0,scheme='centered',closed=False):
    geo=geometry(n); x,xf,v,dx=geo
    T=np.full(n,T0);C=np.full(n,C0); ts=[0.0];Ts=[T.copy()];Cs=[C.copy()];its=[0]
    event=np.nan; losses=[0.0]; sources=[0.0]; balances=[0.0]
    for t in np.arange(dt,end+dt*.01,dt):
        Told,Cold=T.copy(),C.copy();Tn,Cn=T.copy(),C.copy()
        for it in range(1,501):
            Tnext=thomas(*assemble(Told,Tn,Cn,dt,float(t),'T',active,geo,scheme,closed))
            Cnext=thomas(*assemble(Cold,Tnext,Cn,dt,float(t),'C',active,geo,scheme,closed))
            err=max(np.max(abs(Tnext-Tn)),np.max(abs(Cnext-Cn))); Tn,Cn=Tnext,Cnext
            if err<1e-10:break
        else:raise RuntimeError(f'Picard未收敛: {t}')
        T,C=Tn,Cn
        prev=float(np.max(Cold));now=float(np.max(C))
        if np.isnan(event) and prev>=.15 and now<.15:event=float(t-dt+dt*(prev-.15)/(prev-now))
        R=radius(float(t));Ca=environment(float(t))[1]
        physical=0.0 if closed else -2*8e-7*(C[-1]-Ca)/R
        _,_,_,D=coeffs(T,C); du=gradient(C,t,'C',geo,scheme,closed)
        du[-1]=0.0 if closed else -R*8e-7*(C[-1]-Ca)/D[-1]
        source=float(2*np.sum(v*(-x*radius_dot(float(t))/R)*du)) if active else 0.0
        balance=float(2*np.sum(v*(C-Cold))/dt-physical-source)
        ts.append(float(t));Ts.append(T.copy());Cs.append(C.copy());its.append(it)
        losses.append(physical);sources.append(source);balances.append(balance)
    return dict(t=np.array(ts),T=np.array(Ts),C=np.array(Cs),iterations=np.array(its),event=event,
                mean_C=2*np.array(Cs)@v,physical_rate=np.array(losses),residual_rate=np.array(sources),
                balance_error=np.array(balances),n=n,dt=dt,scheme=scheme,active=active)

def field_at(data,t,key='C'):
    ts=data['t']
    if t<ts[0]-1e-9 or t>ts[-1]+1e-9:raise ValueError('禁止区间外夹持')
    j=np.searchsorted(ts,t)
    if j==0:return data[key][0]
    if j==len(ts):return data[key][-1]
    q=(t-ts[j-1])/(ts[j]-ts[j-1]);return data[key][j-1]*(1-q)+data[key][j]*q

def write_json(path,obj):
    # 面向阅读的JSON以四位小数字符串保留尾零；原始数值保存于npz。
    def fmt(x):
        if isinstance(x,dict):return {k:fmt(v) for k,v in x.items()}
        if isinstance(x,list):return [fmt(v) for v in x]
        if isinstance(x,(float,np.floating)):return f'{0.0 if abs(x)<.00005 else x:.4f}'
        if isinstance(x,(np.integer,)):return int(x)
        return x
    path.write_text(json.dumps(fmt(obj),ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    production=dict(np.load(ROOT/'results'/'q4.npz'));end=float(production['t'][-1])
    base=run(False,end=end);ale=run(True,end=end)
    regression=max(float(np.max(abs(base[k]-production[k]))) for k in ['T','C'])
    assert regression<1e-8
    assert abs(float(base['event'])-float(production['event']))<1e-5
    for name,d in [('q4_材料坐标基线',base),('q4_残留ALE对流',ale)]:np.savez_compressed(OUT/f'{name}.npz',**d)
    ev0,ev1=base['event'],ale['event'];delta=ale['C']-base['C']; mi,mj=np.unravel_index(np.argmax(abs(delta)),delta.shape)
    times=sorted(set([18000.,43200.,86400.,129600.,172800.,ev0,ev1]));rows=[]
    for t in times:
        cb,ca=field_at(base,t),field_at(ale,t)
        rows.append(dict(时间_s=t,时间_h=t/3600,中心含水率_基线=cb[0],中心含水率_残留ALE=ca[0],中心差值=ca[0]-cb[0],表面含水率_基线=cb[-1],表面含水率_残留ALE=ca[-1],表面差值=ca[-1]-cb[-1]))
    df=pd.DataFrame(rows);df.to_csv(OUT/'Q4_ALE对照_关键时刻.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
    pd.DataFrame(dict(时间_s=base['t'],中心基线=base['C'][:,0],中心残留ALE=ale['C'][:,0],表面基线=base['C'][:,-1],表面残留ALE=ale['C'][:,-1],中心差值=delta[:,0],表面差值=delta[:,-1],材料平均含水率_基线=base['mean_C'],材料平均含水率_残留ALE=ale['mean_C'])).to_csv(OUT/'Q4_ALE对照_逐时含水率.csv',index=False,encoding='utf-8-sig',float_format='%.4f')
    metrics=dict(实验类型='Q4残留ALE网格导数项诊断',方程约定='右端增加 b*dU/dxi，b=-xi*Rdot/R；温度项乘rho*cp',残项离散='内点中心差分；表面Robin导数闭合',节点数=201,时间步_s=60.0,共同仿真终点_s=end,
      基线终止时间_s=ev0,残留ALE终止时间_s=ev1,基线终止时间_h=ev0/3600,残留ALE终止时间_h=ev1/3600,终止时间差_h=(ev1-ev0)/3600,终止时间相对偏差_百分比=100*(ev1-ev0)/ev0,
      最大含水率绝对差=float(abs(delta[mi,mj])),最大差对应时间_s=float(base['t'][mi]),最大差对应材料坐标=float(mj/200),最大差对应物理半径_cm=radius(float(base['t'][mi]))*mj/200*100,
      基线终止时实验组中心含水率=float(field_at(ale,ev0)[0]),实验组终止时基线中心含水率=float(field_at(base,ev1)[0]),
      残留项累计平均含水率变化=float(np.sum(ale['residual_rate'][1:])*60),
      基线复算最大差=regression,两组径向递增最大值=max(float(np.diff(d['C'],axis=1).max()) for d in [base,ale]),
      全域最大值偏离中心最大值=max(float((d['C'].max(axis=1)-d['C'][:,0]).max()) for d in [base,ale]))
    write_json(OUT/'Q4_ALE对照_指标.json',metrics)
    # 把原始守恒残差换算到可读尺度，避免四位小数掩盖误差。
    valid=dict(基线轨迹与生产数据一致=bool(regression<1e-8),基线复算差乘十亿=regression*1e9,
      基线收支最大残差乘万亿=float(abs(base['balance_error'][1:]).max()*1e12),
      实验组含残项收支最大残差乘万亿=float(abs(ale['balance_error'][1:]).max()*1e12),
      数据全部有限=bool(all(np.isfinite(d['C']).all() and np.isfinite(d['T']).all() for d in [base,ale])))
    if args.validate:
        closed=run(False,end=36000,closed=True);closeda=run(True,end=36000,closed=True)
        valid['均匀纯收缩漂移乘万亿']=max(float(abs(d['C']-C0).max()*1e12) for d in [closed,closeda])
        refined0=run(False,dt=30,end=end);refined1=run(True,dt=30,end=end)
        upwind=run(True,end=end,scheme='upwind')
        valid['时间步三十秒_基线终止_h']=refined0['event']/3600
        valid['时间步三十秒_实验组终止_h']=refined1['event']/3600
        valid['时间步三十秒_相对偏差_百分比']=100*(refined1['event']/refined0['event']-1)
        valid['迎风残项终止_h']=upwind['event']/3600
        valid['迎风与中心差分终止差_h']=(upwind['event']-ev1)/3600
    write_json(OUT/'Q4_ALE对照_验证.json',valid)
    table=df[['时间_h','中心含水率_基线','中心含水率_残留ALE','中心差值','表面差值']].to_string(index=False,formatters={c:(lambda x:f'{x:.4f}') for c in ['时间_h','中心含水率_基线','中心含水率_残留ALE','中心差值','表面差值']})
    discussion=f'''# 问题四残留网格对流项对照结果与论述\n\n在材料坐标 $\\xi=r/R(t)$ 下，正确相对平流系数为 $(v_s-\\xi\\dot R)/R$。均匀径向收缩时 $v_s=\\xi\\dot R$，因此这一系数严格为零。ALE方法本身没有问题；错误来自保留了本应抵消的网格导数项。\n\n本次对照保持201个节点、60.0000 s时间步、半径PCHIP插值、题给变物性、边界条件与Picard容差一致。诊断组在温度和水分方程右端保留 $-\\xi\\dot R/R\\,\\partial_\\xi U$，温度方程相应乘体积热容。残项采用内点中心差分、表面Robin导数闭合。两组均算至 {end:.4f} s，并分别插值得到全域含水率首次低于0.1500的时刻。\n\n基线干燥时间为 {ev0/3600:.4f} h，残项组为 {ev1/3600:.4f} h，提前 {(ev0-ev1)/3600:.4f} h，相对偏差 {100*(ev1-ev0)/ev0:.4f}%。这些是当前离散下重新计算的结果，没有混用清单中的历史数值。\n\n{table}\n\n收缩阶段Rdot为负，含水率通常沿半径递减，因此右端残项额外降低局部含水率，形成虚假失水。早期几何变化快且水分梯度较大，偏差累积明显；后期收缩减弱，瞬时残项趋小，但先前形成的含水率偏差仍会提前触发终止判据。全时空含水率最大绝对差为 {abs(delta[mi,mj]):.4f}，出现于 {base['t'][mi]:.4f} s。\n\n不能用“各自终止时中心含水率均为0.1500”说明两组一致，因为这是终止判据规定的结果。应比较相同时刻：实验组达标时，基线中心含水率仍为 {field_at(base,ev1)[0]:.4f}；基线达标时，实验组中心含水率已为 {field_at(ale,ev0)[0]:.4f}。\n\n因此，残留网格项会造成系统性干燥时间偏差，必须消除，而不能作为收缩的物理贡献加入主模型。正确ALE实现应与材料坐标模型等价。均匀场纯收缩检验只是必要条件，因为梯度为零时错误残项也会消失；还需检查非均匀场的相对速度抵消及含残项/不含残项的离散水分收支。\n'''
    (OUT/'Q4_ALE对照_说明.md').write_text(discussion,encoding='utf-8')
    manifest=dict(复现命令='python3 src/q4_ale_compare.py --validate',绘图命令='python3 src/plot_q4_ale_compare.py',输入哈希={f:hashlib.sha256((ROOT/'results'/f).read_bytes()).hexdigest() for f in ['inputs.npz','q4.npz']},随机性='无',Python=platform.python_version(),NumPy=np.__version__)
    write_json(OUT/'Q4_ALE对照_复现清单.json',manifest)
    print(json.dumps(metrics,ensure_ascii=False,indent=2));print(json.dumps(valid,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
