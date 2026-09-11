"""Production runner for the unified CUMCM 2026 A model."""
from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import numpy as np
from openpyxl import load_workbook
try:
    from src.solver import (ROOT, DATA, OUT, Config, air, radius, Tinf, Cbinf, Rmeas, solve_1d, solve_2d, interp_radial, reconstruct_center, surface_values as surface_values_solver, sha256)
except ModuleNotFoundError:
    from solver import (ROOT, DATA, OUT, Config, air, radius, Tinf, Cbinf, Rmeas, solve_1d, solve_2d, interp_radial, reconstruct_center, surface_values as surface_values_solver, sha256)
R0=0.02; C0,T0=2.55,28.; N=160

def surface_values(kind, y, times, moving=False, n=N):
    """Backward-compatible surface reconstruction for paper/plot scripts."""
    arr=np.asarray(y); times=np.asarray(times,dtype=float)
    cfg=Config(kind=kind,n=n,moving=moving)
    data={"T":arr[:,:n],"C":arr[:,n:],"t":times}
    return surface_values_solver(cfg,data)

def interp_fixed(y,radii,R=R0): return interp_radial(np.asarray(y),radii,R)
def interp_moving(y,times,radii):
    y=np.asarray(y); out=np.full((len(times),len(radii)),np.nan)
    for j,t in enumerate(times):
        R=Rmeas(float(t)); out[j]=interp_radial(y[j:j+1],radii,R)[0]; out[j,np.asarray(radii)>R]=np.nan
    return out

def write_book(path,sheets,headers,times):
    wb=load_workbook(DATA/'附件3'/path.name); names=list(wb.sheetnames)
    for i,(name,arr) in enumerate(sheets.items()):
        ws=wb[names[i]] if i<len(names) else wb.create_sheet(name); ws.delete_rows(1,ws.max_row)
        if name in ('温度','水分浓度'): ws.title=name
        ws.append(['时间\\到药材中心的距离']+list(headers))
        for t,row in zip(times,np.asarray(arr)): ws.append([float(t)]+[None if not np.isfinite(v) else float(v) for v in row])
        for row in ws.iter_rows(min_row=2):
            for c in row[1:]:
                if c.value is not None: c.number_format='0.0000'
    wb.save(path)

def export_question(kind,data,name,moving=False):
    radii=np.arange(0,2.0001,.1)*1e-2
    T=interp_moving(data['T'],data['t'],radii) if moving else interp_fixed(data['T'],radii)
    C=interp_moving(data['C'],data['t'],radii) if moving else interp_fixed(data['C'],radii)
    cfg=Config(kind=kind,n=data['C'].shape[1],moving=moving); st,sc,_=surface_values(cfg,data)
    T[:,-1]=st; C[:,-1]=sc
    if moving: T=np.column_stack([T,st]); C=np.column_stack([C,sc]); headers=list(radii*100)+['药材表面']
    else: headers=list(radii*100)
    if name in ('result1.xlsx','result2.xlsx'): sheets={'温度':T,'水分浓度':C}
    else: sheets={'Sheet1': C}
    write_book(OUT/name,sheets,headers,data['t'])

def write_summary(data,kind,moving=False):
    r=np.arange(0,2.0001,.5)*1e-2; vt=interp_moving(data['T'],data['t'],r) if moving else interp_fixed(data['T'],r); vc=interp_moving(data['C'],data['t'],r) if moving else interp_fixed(data['C'],r)
    times=np.array([100,300,600,900,1200,1500,1800.]) if kind=='q1' else np.arange(.5,3.01,.5)*3600 if kind=='q2' else np.arange(6,55,6)*3600
    a=[]; b=[]
    for t in times:
        j=int(np.argmin(abs(data['t']-t))); a.append(vt[j]); b.append(vc[j])
    if kind=='q1':
        np.savetxt(OUT/'表1_温度.csv',np.c_[times,a],delimiter=',',header='time,'+','.join(map(str,r*100)),comments=''); np.savetxt(OUT/'表2_水分浓度.csv',np.c_[times,b],delimiter=',',header='time,'+','.join(map(str,r*100)),comments='')
    elif kind=='q2':
        np.savetxt(OUT/'表3_温度.csv',np.c_[times,a],delimiter=',',header='time,'+','.join(map(str,r*100)),comments=''); np.savetxt(OUT/'表4_水分浓度.csv',np.c_[times,b],delimiter=',',header='time,'+','.join(map(str,r*100)),comments='')
    else: np.savetxt(OUT/'表5_水分浓度.csv',np.c_[times,b],delimiter=',',header='time,'+','.join(map(str,r*100)),comments='')

def energy(data,cfg):
    st,sc,J=surface_values(cfg,data); t=data['t']; R=np.array([Rmeas(x) if cfg.moving else R0 for x in t]); A=2*np.pi*R*.25
    ec=float(np.trapezoid(A*cfg.h*(np.array([Tinf(x) for x in t])-st),t)); el=float(np.trapezoid(A*cfg.Lv*J,t)) if cfg.latent else 0.
    return {'E_conv_J':ec,'E_lat_J':el,'latent_fraction':el/ec if ec else 0.,'surface_T_min_C':float(st.min()),'surface_C_min':float(sc.min())}

def main():
    start=time.time()
    q1=solve_1d(Config(kind='q1',n=N,dt=5),1800,np.arange(0,1801,1.))
    q2=solve_1d(Config(kind='q2',n=N,dt=5),10800,np.arange(0,10801,1.))
    q3=solve_1d(Config(kind='q2',n=N,dt=120),2000000,np.arange(0,2000000+60,60.),event_threshold=.15)
    q4=solve_1d(Config(kind='q4',n=N,dt=120,moving=True),2000000,np.arange(0,2000000+60,60.),event_threshold=.15)
    export_question('q1',q1,'result1.xlsx'); export_question('q2',q2,'result2.xlsx'); export_question('q2',q3,'result3.xlsx'); export_question('q4',q4,'result4.xlsx',True)
    for name,d in [('q1',q1),('q2',q2),('q3',q3),('q4',q4)]:
        np.savez_compressed(OUT/f'{name}.npz',t=d['t'],T=d['T'],C=d['C'],iterations=d['iterations'],Jw=d['Jw'],qconv=d['qconv'],event_time=np.nan if d['event'] is None else d['event'])
        write_summary(d,'q1' if name=='q1' else 'q2' if name=='q2' else name,name=='q4')
    l3=solve_1d(Config(kind='q2',n=80,dt=120,latent=True),500000,np.arange(0,500001,60.),event_threshold=.15)
    l4=solve_1d(Config(kind='q4',n=80,dt=120,moving=True,latent=True),500000,np.arange(0,500001,60.),event_threshold=.15)
    ledger={'baseline_q3_event_s':q3['event'],'baseline_q4_event_s':q4['event'],'latent_q3_event_s':l3['event'],'latent_q4_event_s':l4['event'],'q3_latent_energy':energy(l3,Config(kind='q2',n=80,dt=120,latent=True)),'q4_latent_energy':energy(l4,Config(kind='q4',n=80,dt=120,moving=True,latent=True))}
    (OUT/'energy_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)); (OUT/'shrinkage_scenarios.json').write_text(json.dumps({str(g):{'H_over_H0_at_last':float((Rmeas(2000000)/R0)**g)} for g in (0,.5,1)},indent=2))
    d2a=solve_2d('q4',3600,nr=8,nz=12,dt=30); d2b=solve_2d('q4',3600,nr=12,nz=18,dt=30)
    (OUT/'two_d_validation.json').write_text(json.dumps({'coarse_grid':[8,12],'fine_grid':[12,18],'grid_change_max_C':float(abs(d2b['T'].max()-d2a['T'].max())),'grid_change_max_C_moisture':float(abs(d2b['C'].max()-d2a['C'].max())),'note':'2D endpoint-transfer scenario; Q4 production remains 1D.'},indent=2))
    event={'q3_continuous_s':q3['event'],'q3_continuous_h':None if q3['event'] is None else q3['event']/3600,'q4_continuous_s':q4['event'],'q4_continuous_h':None if q4['event'] is None else q4['event']/3600,'q3_first_strict_sample_s':None if q3['event'] is None else float(np.ceil(q3['event']/60)*60),'q4_first_strict_sample_s':None if q4['event'] is None else float(np.ceil(q4['event']/60)*60),'q3_max_at_first_sample':None if q3['event'] is None else float(max(np.max(q3['C'][int(np.argmin(abs(q3['t']-np.ceil(q3['event']/60)*60)))]),reconstruct_center(q3['C'][int(np.argmin(abs(q3['t']-np.ceil(q3['event']/60)*60)))]))),'q4_max_at_first_sample':None if q4['event'] is None else float(max(np.max(q4['C'][int(np.argmin(abs(q4['t']-np.ceil(q4['event']/60)*60)))]),reconstruct_center(q4['C'][int(np.argmin(abs(q4['t']-np.ceil(q4['event']/60)*60)))]))),'threshold':.15,'event_definition':'max(cell moisture, center reconstruction, Robin surface)-threshold','N':N,'dt_q1_q2_s':1,'dt_q3_q4_s':60,'environment_continuation':'last measured value'}
    (OUT/'event_summary.json').write_text(json.dumps(event,ensure_ascii=False,indent=2))
    inputs=[ROOT/'A题/A题.pdf',DATA/'附件1.xlsx',DATA/'附件2.xlsx']+list((DATA/'附件3').glob('*.xlsx'))
    manifest={'schema_version':2,'runtime':{'python':platform.python_version(),'numpy':np.__version__},'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for p in inputs],'configuration':{'N':N,'baseline_latent':False,'latent_Lv_J_per_kg':[2.3e6,2.4e6,2.5e6],'q4_gamma_scenarios':[0,.5,1],'event_threshold':.15},'outputs':['result1.xlsx','result2.xlsx','result3.xlsx','result4.xlsx','event_summary.json','energy_ledger.json','two_d_validation.json'],'reproduce_command':f'{sys.executable} src/model.py && {sys.executable} src/postprocess.py','elapsed_s':time.time()-start}
    (OUT/'复现清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)); print(json.dumps(event,ensure_ascii=False))
if __name__=='__main__': main()
