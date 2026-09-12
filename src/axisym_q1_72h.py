"""2-D axisymmetric (r,z) Q1 comparison through 72 h.

Finite-volume, backward Euler, node-independent Q1 properties.  The measured
boundary series is used on its available interval; after its final sample the
last measured value is held explicitly as a long-time continuation so that the
72 h run is reproducible (this is not an extrapolation of a fitted trend).
"""
from __future__ import annotations
import json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized, spsolve
from scipy.interpolate import interp1d

HERE=Path(__file__).resolve().parent
REPO=HERE.parent
ROOT=REPO/'data'
OUT=REPO/'results'/'axisym_q1_72h'; OUT.mkdir(parents=True,exist_ok=True)
R0=0.02; L=0.25; RHO=820.; CP=2600.; K=0.36; D0=7e-9; H=25.; HM=8e-7
TINIT=28.; CINIT=2.55; DT=60.; END=72*3600
NR=41; NZ=51

a1=pd.read_excel(ROOT/'附件'/'附件1.xlsx',header=0).iloc[:,:3].to_numpy(float)
AT=a1[:,0]; ATEMP=a1[:,1]; ACB=a1[:,2]
# np.interp holds the final measured value; the mode is recorded in the manifest.
def env(t): return float(np.interp(t,AT,ATEMP)),float(np.interp(t,AT,ACB))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def grid():
    dr=R0/NR; dz=L/NZ
    rc=(np.arange(NR)+.5)*dr; ri=np.arange(NR)*dr; ro=(np.arange(NR)+1)*dr
    # volume without common factor 2*pi: integral r dr dz
    V=.5*(ro*ro-ri*ri)*dz
    zc=(np.arange(NZ)+.5)*dz
    return dr,dz,rc,zc,V
DR,DZ,RC,ZC,Vr=grid(); NC=NR*NZ

def idx(j,i): return j*NR+i

def const_T_matrix(dt=DT):
    A=lil_matrix((NC,NC),dtype=float); mass=RHO*CP*Vr/dt
    for j in range(NZ):
      for i in range(NR):
        p=idx(j,i); d=mass[i];
        if i>0:
          g=K*(i*DR)*DZ/DR; A[p,idx(j,i-1)]-=g; d+=g
        if i<NR-1:
          g=K*((i+1)*DR)*DZ/DR; A[p,idx(j,i+1)]-=g; d+=g
        else:
          g=R0*DZ/(DR/(2*K)+1/H); d+=g
        if j>0:
          g=K*RC[i]*DR/DZ; A[p,idx(j-1,i)]-=g; d+=g
        else:
          g=(Vr[i]/DZ)/(DZ/(2*K)+1/H); d+=g
        if j<NZ-1:
          g=K*RC[i]*DR/DZ; A[p,idx(j+1,i)]-=g; d+=g
        else:
          g=(Vr[i]/DZ)/(DZ/(2*K)+1/H); d+=g
        A[p,p]=d
    return A.tocsc(),np.tile(mass,NZ)

def build_matrix(D, dt=DT):
    A=lil_matrix((NC,NC),dtype=float); mass=Vr/dt
    for j in range(NZ):
      for i in range(NR):
        p=idx(j,i); d=mass[i]
        if i>0:
          g=.5*(D[j,i]+D[j,i-1])*(i*DR)*DZ/DR; A[p,idx(j,i-1)]-=g; d+=g
        if i<NR-1:
          g=.5*(D[j,i]+D[j,i+1])*((i+1)*DR)*DZ/DR; A[p,idx(j,i+1)]-=g; d+=g
        else:
          g=R0*DZ/(DR/(2*D[j,i])+1/HM); d+=g
        if j>0:
          g=.5*(D[j,i]+D[j-1,i])*RC[i]*DR/DZ; A[p,idx(j-1,i)]-=g; d+=g
        else:
          g=(Vr[i]/DZ)/(DZ/(2*D[j,i])+1/HM); d+=g
        if j<NZ-1:
          g=.5*(D[j,i]+D[j+1,i])*RC[i]*DR/DZ; A[p,idx(j+1,i)]-=g; d+=g
        else:
          g=(Vr[i]/DZ)/(DZ/(2*D[j,i])+1/HM); d+=g
        A[p,p]=d
    return A.tocsc(),np.tile(mass,NZ)

def rhs_boundary_T(t,mass,old):
    et,ec=env(t); b=mass*old.ravel();
    for j in (0,NZ-1):
      for i in range(NR): b[idx(j,i)] += (Vr[i]/DZ)/(DZ/(2*K)+1/H)*et
    for j in range(NZ): b[idx(j,NR-1)] += R0*DZ/(DR/(2*K)+1/H)*et
    return b

def rhs_boundary_C(t,mass,old):
    et,ec=env(t); b=mass*old.ravel(); d=D0*np.exp(-.89/np.maximum(old,1e-12))
    for j in (0,NZ-1):
      for i in range(NR): b[idx(j,i)] += (Vr[i]/DZ)/(DZ/(2*d[j,i])+1/HM)*ec
    for j in range(NZ): b[idx(j,NR-1)] += R0*DZ/(DR/(2*d[j,NR-1])+1/HM)*ec
    return b

def matching_1d():
    """Cell-centered 1-D Q1 solve using the same radial control volumes as 2-D."""
    Vr1=.5*((np.arange(NR)+1)**2-np.arange(NR)**2)*DR**2
    mass=RHO*CP*Vr1/DT
    A=lil_matrix((NR,NR),dtype=float)
    for i in range(NR):
      d=mass[i]
      if i>0:
        g=K*(i*DR)/DR; A[i,i-1]-=g; d+=g
      if i<NR-1:
        g=K*((i+1)*DR)/DR; A[i,i+1]-=g; d+=g
      else: d+=R0/(DR/(2*K)+1/H)
      A[i,i]=d
    solveT=factorized(A.tocsc()); T=np.full(NR,TINIT); C=np.full(NR,CINIT)
    ts=[0.]; Ts=[T.copy()]; Cs=[C.copy()]
    wanted={int(x) for x in np.r_[np.arange(1800,14401,1800),np.arange(18000,259201,3600)]}
    for n in range(1,int(END/DT)+1):
      t=n*DT; b=mass*T; b[-1]+=R0/(DR/(2*K)+1/H)*env(t)[0]; T=solveT(b)
      d=D0*np.exp(-.89/np.maximum(C,1e-12)); A=lil_matrix((NR,NR),dtype=float); mc=Vr1/DT
      for i in range(NR):
        di=mc[i]
        if i>0:
          g=.5*(d[i]+d[i-1])*(i*DR)/DR; A[i,i-1]-=g; di+=g
        if i<NR-1:
          g=.5*(d[i]+d[i+1])*((i+1)*DR)/DR; A[i,i+1]-=g; di+=g
        else: di+=R0/(DR/(2*d[i])+1/HM)
        A[i,i]=di
      b=mc*C; b[-1]+=R0/(DR/(2*d[-1])+1/HM)*env(t)[1]; C=spsolve(A.tocsc(),b)
      if n*DT in wanted: ts.append(float(t)); Ts.append(T.copy()); Cs.append(C.copy())
    return {'t':np.asarray(ts),'T':np.asarray(Ts),'C':np.asarray(Cs),'r':RC}

def surface_values(field,t,kind):
    if float(t) <= 0.0:
      return field[:,-1].copy(), field[0,:].copy(), field[-1,:].copy()
    et,ec=env(t); b=H if kind=='T' else HM; kap=K if kind=='T' else None
    # reconstruct Robin boundary values from adjacent cell centres
    if kind=='T': fac=K/(K+H*DR/2); envv=et; beta=H
    else:
      C=np.maximum(field,1e-12); d=D0*np.exp(-.89/C); fac=d[:,-1]/(d[:,-1]+HM*DR/2); envv=ec; beta=HM
    side=fac*field[:,-1]+(1-fac)*envv
    # axial end reconstruction uses half-cell distance; local conductivity
    if kind=='T': facz=K/(K+H*DZ/2)
    else:
      facz=np.maximum(D0*np.exp(-.89/np.maximum(field[0,:],1e-12)),1e-30)
      facz=facz/(facz+HM*DZ/2)
    z0=facz*field[0,:]+(1-facz)*envv; zL=facz*field[-1,:]+(1-facz)*envv
    return side,z0,zL

def save_profile(times,Ts,Cs,oneD):
    # oneD arrays dict t,T,C,r from long baseline; compare mid-plane and near-end.
    rows=[]; mid=NZ//2; end=0
    rr=np.linspace(0,R0,21)
    for t,T,C in zip(times,Ts,Cs):
      Tm=np.interp(rr,RC,T[mid,:]); Cm=np.interp(rr,RC,C[mid,:])
      # center and surface endpoint reconstructions
      sideT,z0T,zLT=surface_values(T,float(t),'T'); sideC,z0C,zLC=surface_values(C,float(t),'C')
      Tm[0]=T[mid,0]; Cm[0]=C[mid,0]; Tm[-1]=sideT[mid]; Cm[-1]=sideC[mid]
      T1=np.interp(rr,oneD['r'],oneD['T'][np.argmin(abs(oneD['t']-t))]); C1=np.interp(rr,oneD['r'],oneD['C'][np.argmin(abs(oneD['t']-t))])
      for r,tt,cc,t1,c1 in zip(rr,Tm,Cm,T1,C1): rows.append({'time_s':float(t),'r_m':float(r),'r_cm':float(r*100),'T_2d_mid_C':float(tt),'C_2d_mid':float(cc),'T_1d_C':float(t1),'C_1d':float(c1),'dT_2d_minus_1d_C':float(tt-t1),'dC_2d_minus_1d':float(cc-c1)})
    pd.DataFrame(rows).to_csv(OUT/'midplane_radial_compare.csv',index=False)
    ar=[]
    for t,T,C in zip(times,Ts,Cs):
      sideT,z0T,zLT=surface_values(T,float(t),'T'); sideC,z0C,zLC=surface_values(C,float(t),'C')
      for j,z in enumerate(ZC): ar.append({'time_s':float(t),'z_m':float(z),'z_cm':float(z*100),'T_axis_2d_C':float(T[j,0]),'C_axis_2d':float(C[j,0]),'T_side_2d_C':float(sideT[j]),'C_side_2d':float(sideC[j]),'T_end_axis_surface_C':float(z0T[0] if j==0 else zLT[0] if j==NZ-1 else T[j,0]),'C_end_axis_surface':float(z0C[0] if j==0 else zLC[0] if j==NZ-1 else C[j,0])})
    pd.DataFrame(ar).to_csv(OUT/'axial_compare.csv',index=False)

def main():
    try:
      from src import model as paper_baseline
      from src.model import geometry, set_grid, run_fixed
    except ModuleNotFoundError:
      import model as paper_baseline
      from model import geometry, set_grid, run_fixed
    # Match production 1-D radial grid and 60-s output step for the long-time comparison.
    set_grid(201); one=matching_1d()
    A,mT=const_T_matrix(); solveT=factorized(A)
    T=np.full((NZ,NR),TINIT); C=np.full((NZ,NR),CINIT); times=[0.]; Ts=[T.copy()]; Cs=[C.copy()]
    wanted=set([int(x) for x in np.r_[np.arange(1800,14401,1800),np.arange(18000,259201,3600)]])
    t0=time.time()
    for n in range(1,int(END/DT)+1):
      t=n*DT
      T=solveT(rhs_boundary_T(t,mT,T)).reshape(NZ,NR)
      d=D0*np.exp(-.89/np.maximum(C,1e-12)); A_c,mC=build_matrix(d); b=rhs_boundary_C(t,mC,C); C=spsolve(A_c,b).reshape(NZ,NR)
      if n*DT in wanted: times.append(float(t));Ts.append(T.copy());Cs.append(C.copy())
    save_profile(times,Ts,Cs,one)
    # event and discrepancy diagnostics over all 60 s steps are recomputed by running output profile only at requested times;
    # use 1-D long trajectory and 2-D snapshots for extrema.
    summary=[]
    for t,T,C in zip(times,Ts,Cs):
      sideT,z0T,zLT=surface_values(T,t,'T'); sideC,z0C,zLC=surface_values(C,t,'C'); j=NZ//2
      k=int(np.argmin(abs(one['t']-t))); T1=one['T'][k]; C1=one['C'][k]; T1i=np.interp(RC,one['r'],T1); C1i=np.interp(RC,one['r'],C1)
      T1i=np.interp(RC,one['r'],T1); C1i=np.interp(RC,one['r'],C1)
      summary.append({'time_s':float(t),'time_h':float(t/3600),'T_2d_center_C':float(T[j,0]),'C_2d_center':float(C[j,0]),'T_2d_mid_surface_C':float(sideT[j]),'C_2d_mid_surface':float(sideC[j]),'T_2d_end_center_surface_C':float(z0T[0]),'C_2d_end_center_surface':float(z0C[0]),'T_1d_center_C':float(T1[0]),'C_1d_center':float(C1[0]),'center_dT_C':float(T[j,0]-T1[0]),'center_dC':float(C[j,0]-C1[0]),'max_abs_mid_dT_C':float(np.max(np.abs(T[j,:]-T1i))), 'max_abs_mid_dC':float(np.max(np.abs(C[j,:]-C1i)))})
    pd.DataFrame(summary).to_csv(OUT/'summary_compare.csv',index=False)
    np.savez_compressed(OUT/'axisym_snapshots.npz',t=np.array(times),T=np.array(Ts),C=np.array(Cs),r=RC,z=ZC)
    manifest={'model':'2D axisymmetric r-z Q1 finite-volume backward Euler','Nr':NR,'Nz':NZ,'dt_s':DT,'end_h':72,'geometry':{'R_m':R0,'L_m':L},'parameters':{'rho':RHO,'cp':CP,'k':K,'D_formula':'7e-9 exp(-0.89/C)','h':H,'hm':HM},'boundary':'Robin on r=R and z=0,L; symmetry at r=0','environment':'Attachment1 measured samples; np.interp and hold final sample after 4h for 72h continuation','source_hashes':{'附件1.xlsx':sha(ROOT/'附件'/'附件1.xlsx'),'model.py':sha(HERE/'model.py')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'snapshots_h':[t/3600 for t in times],'runtime_s':time.time()-t0,'max_snapshot_center_dT_C':max(abs(x['center_dT_C']) for x in summary)},ensure_ascii=False))
if __name__=='__main__': main()
