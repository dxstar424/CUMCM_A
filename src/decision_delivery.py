"""Generate final-decision baseline diagnostics and sensitivity deliverables.

This script only uses final决策清单.md, 论文初稿 1.docx and the fuvk
attachments through src/model.py.  The M0 result remains latent-off; latent
quantities are reported as a bounded surface diagnostic rather than coupled into
the production state.
"""
from __future__ import annotations
import argparse, json, platform, time, math
from pathlib import Path
import numpy as np
try:
    from src import model as m
except ModuleNotFoundError:
    import model as m

OUT = m.OUT

def surface_state(T, C, kind, t, phase):
    _, _, k, D = m.coeffs(kind, T, C)
    Ta, Cb = m.environment(t, phase)
    Ts = T[-1]
    Cs = C[-1]
    return float(Ts), float(Cs), float(Ta), float(Cb)

def mass_audit(data, kind, phase, moving=False):
    t, C = data['t'], data['C']
    weights = m.V if not moving else None
    if moving:
        weights = np.empty(m.N); dx=1/(m.N-1); x=np.arange(m.N)*dx
        weights[0]=dx*dx/8; weights[1:-1]=.5*((x[1:-1]+dx/2)**2-(x[1:-1]-dx/2)**2); weights[-1]=.5*(1-(1-dx/2)**2)
    if moving:
        # In material coordinates rho_d(t)~R0^2/R(t)^2 for fixed length;
        # using the dry-basis storage makes the geometry change explicit.
        rho_d0=(760.0+90.0*m.C0)/(1.0+m.C0)
        storage=np.array([rho_d0*m.R0**2*np.sum(C[i]*weights) for i in range(len(t))])
    else:
        storage=np.array([np.sum(C[i]*weights) for i in range(len(t))])
    Cb=[]; flux=[]; R=[]
    for tt, cc in zip(t,C):
        _, cb=m.environment(tt,phase); Cb.append(cb)
        rr=m.radius(tt) if moving else m.R0; R.append(rr)
        # The discrete boundary flux must use the same last-node value as the
        # production matrix, otherwise the audit mixes two boundary closures.
        rho_d=(rho_d0*m.R0**2/rr**2) if moving else 1.0
        flux.append(rr*(8e-7*m.HM_SCALE)*rho_d*(cc[-1]-cb))
    Cb=np.asarray(Cb); flux=np.asarray(flux); R=np.asarray(R)
    loss=np.concatenate([[0.0],np.cumsum(.5*(flux[1:]+flux[:-1])*np.diff(t))])
    return {'storage_delta':float(storage[-1]-storage[0]),'boundary_loss_signed':float(-loss[-1]),'residual':float(storage[-1]-storage[0]+loss[-1]),'relative_residual':float((storage[-1]-storage[0]+loss[-1])/max(abs(storage[0]),1e-30))}

def monotonicity(data):
    # center is expected to be the maximum; radial node values should not rise outward.
    d=np.diff(data['C'],axis=1)
    return {'max_outward_increase':float(np.max(d)),'center_max_violation':float(np.max(data['C'][:,1:]-data['C'][:,:1]))}

def q3_run(**changes):
    old={k:getattr(m,k) for k in ('FACE_MODE','D_SCALE','H_SCALE','HM_SCALE','PLATEAU_CB')}
    try:
        for k,v in changes.items(): setattr(m,k,v)
        return m.run_fixed('q2',1000000,60,60,True)
    finally:
        for k,v in old.items(): setattr(m,k,v)

def latent_ledger(data, kind, phase, Lv=2.4e6, rho_d=650.0):
    t=data['t']; qconv=[]; qlat=[]; Ts=[]; J=[]
    hm=8e-7*m.HM_SCALE
    for tt,T,C in zip(t,data['T'],data['C']):
        ts,cs,ta,cb=surface_state(T,C,kind,tt,phase); jw=rho_d*hm*(cs-cb)
        qconv.append(m.H_SCALE*25.0*(ta-ts)); qlat.append(Lv*jw); Ts.append(ts); J.append(jw)
    qconv=np.asarray(qconv); qlat=np.asarray(qlat); area=2*np.pi*m.R0*m.LENGTH
    ecol=float(np.trapezoid(area*qconv,t)); elat=float(np.trapezoid(area*qlat,t))
    ratio=np.divide(qlat,qconv,out=np.full_like(qlat,np.nan),where=(qconv>1.0)&(qlat>0))
    finite=ratio[np.isfinite(ratio)]
    return {'latent_enabled':False,'Lv_J_per_kg':Lv,'rho_d_ref_kg_m3':rho_d,'E_conv_J':ecol,'E_lat_diagnostic_J':elat,'E_lat_over_E_conv':float(elat/ecol if ecol else np.nan),'q_conv_ratio_floor_W_m2':1.0,'q_lat_over_q_conv_min_positive':float(np.min(finite)) if finite.size else None,'q_lat_over_q_conv_max_positive':float(np.max(finite)) if finite.size else None,'surface_T_min_C':float(np.min(Ts)),'Jw_max_kg_m2_s':float(np.max(J)),'interpretation':'surface latent diagnostic only; not coupled into M0; instantaneous ratios use q_conv>1 W/m2 because the denominator tends to zero late in drying'}

def latent_status():
    # The two allowed source documents do not provide a dry-density value or a
    # latent-heat value.  Keep the production run latent-off and do not attach
    # invented numerical bounds to the delivery.
    return {'latent_enabled': False,
            'status': 'not_quantified_from_allowed_sources',
            'reason': 'final决策清单 and 论文初稿 1.docx specify latent heat as a diagnostic/sensitivity item but do not specify Lv or a dry-density closure',
            'production_effect': 'none'}

def main(refresh_only=False):
    start=time.time()
    q1=np.load(OUT/'q1.npz'); q2=np.load(OUT/'q2.npz'); q3=np.load(OUT/'q3.npz'); q4=np.load(OUT/'q4.npz')
    d1={k:q1[k] for k in q1.files}; d2={k:q2[k] for k in q2.files}; d3={k:q3[k] for k in q3.files}; d4={k:q4[k] for k in q4.files}
    validation={'status':'numerically_checked_1d_baseline','model':'M0','q1_mass':mass_audit(d1,'q1','q1'),'q2_mass':mass_audit(d2,'q2','q2'),'q3_mass':mass_audit(d3,'q2','q3'),'q4_mass':mass_audit(d4,'q4','q3',True),'q1_monotonicity':monotonicity(d1),'q2_monotonicity':monotonicity(d2),'q3_monotonicity':monotonicity(d3),'q4_monotonicity':monotonicity(d4),'positive_finite':all(np.isfinite(d[k]).all() and np.min(d[k])>0 for d in (d1,d2,d3,d4) for k in ('T','C')),'Bi_T':1.3888888889,'Bi_m':3.2404,'Fo_T_1800':0.76,'Fo_m_1800':0.022,'two_d_status':'not_implemented_in_this_baseline_delivery','note':'Mass residuals use output-grid trapezoidal flux; the 2D percentage in the source is not reproduced here.'}
    # Arithmetic baseline versus harmonic face sensitivity.
    sensitivity=json.loads((OUT/'sensitivity.json').read_text()) if refresh_only and (OUT/'sensitivity.json').exists() else []
    if not sensitivity:
        for label,changes in [('face_harm',{'FACE_MODE':'harm'}),('D_0.9',{'D_SCALE':.9}),('D_1.1',{'D_SCALE':1.1}),('hm_0.9',{'HM_SCALE':.9}),('hm_1.1',{'HM_SCALE':1.1}),('h_0.9',{'H_SCALE':.9}),('h_1.1',{'H_SCALE':1.1}),('Cb_0.045',{'PLATEAU_CB':.045}),('Cb_0.055',{'PLATEAU_CB':.055})]:
            d=q3_run(**changes); first=(math.floor(float(d['event'])/60.0)+1)*60.0; sensitivity.append({'label':label,'changes':changes,'event_s':float(d['event']),'event_h':float(d['event']/3600),'first_strict_sample_s':first})
    else:
        for row in sensitivity:
            row['first_strict_sample_s']=(math.floor(float(row['event_s'])/60.0)+1)*60.0
    (OUT/'sensitivity.json').write_text(json.dumps(sensitivity,ensure_ascii=False,indent=2))
    gamma=[]; R_end=m.radius(float(d4['t'][-1]))
    for g in (0,.5,1):
        scale=R_end/m.R0; gamma.append({'gamma':g,'event_h':float(d4['event']/3600),'R_end_over_R0':float(scale),'L_end_over_L0':float(scale**g),'V_end_over_V0':float(scale**(2+g)),'field_effect':'none in 1D material-coordinate baseline; geometry/volume diagnostic only'})
    (OUT/'shrinkage_scenarios.json').write_text(json.dumps(gamma,ensure_ascii=False,indent=2))
    ledger={'baseline_M0':latent_status(),'source_defined_quantification':False}
    (OUT/'energy_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
    events=json.loads((OUT/'event_summary.json').read_text()); events.update({'plateau_start_s':7200.0,'plateau_T_C':50.0,'plateau_Cb':0.05,'q3_center_vs_max_max_abs':float(np.max(abs(d3['C'][:,0]-np.max(d3['C'],axis=1)))),'q4_center_vs_max_max_abs':float(np.max(abs(d4['C'][:,0]-np.max(d4['C'],axis=1)))),'strict_rule':'C(0,t)<0.15; max(C) self-check','face_mode':'arith'}); (OUT/'event_summary.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
    (OUT/'validation_report.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    manifest={'source_files':['final决策清单.md','论文初稿 1.docx'],'baseline':'M0 latent-off 1D radial','decision_mapping':{'简-1':'1D radial','简-14b':'T=50,Cb=0.05 after 7200s','模-6':'center event with max self-check','模-7':'material coordinate','算-1':'backward Euler','算-2':'node-centered FVM','算-4':'arithmetic baseline; harmonic sensitivity','算-5':'Picard','算-6':'Thomas','算-7':'linear event interpolation','算-8':'PCHIP radius','简-13':'gamma=0 baseline; gamma=.5,1 diagnostics','其-2':'latent surface diagnostic only'},'runtime':{'python':platform.python_version(),'numpy':np.__version__},'outputs':['result1.xlsx','result2.xlsx','result3.xlsx','result4.xlsx','表一_问题一温度.csv','表二_问题一含水率.csv','表三_问题二温度.csv','表四_问题二含水率.csv','表五_问题三长时含水率.csv','表六_问题四收缩含水率.csv','event_summary.json','validation_report.json','energy_ledger.json','shrinkage_scenarios.json','sensitivity.json'],'elapsed_s':time.time()-start,'two_d_validation':'由 src/axisym_q1_72h.py 独立执行，不混入一维生产基线'}
    (OUT/'decision_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'events':events,'sensitivity':sensitivity},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--refresh-only',action='store_true'); args=parser.parse_args(); main(args.refresh_only)
