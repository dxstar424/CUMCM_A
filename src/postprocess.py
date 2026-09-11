"""Post-process raw model arrays into summary CSVs and diagnostic figures."""
from __future__ import annotations
from pathlib import Path
import json, numpy as np
import matplotlib.pyplot as plt
try:
    from src.solver import ROOT, OUT, Rmeas, reconstruct_center, interp_radial, air, radius
except ModuleNotFoundError:
    from solver import ROOT, OUT, Rmeas, reconstruct_center, interp_radial, air, radius
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})

def load(name): return np.load(OUT/f'{name}.npz')
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/f'{name}.png',dpi=180); plt.savefig(FIG/f'{name}.svg'); plt.close()
def main():
    q1,q2,q3,q4=map(load,['q1','q2','q3','q4'])
    # Boundary histories
    fig,ax=plt.subplots(2,1,sharex=True,figsize=(6,4)); ax[0].plot(air[:,0]/3600,air[:,1]); ax[0].set_ylabel('T∞ / °C'); ax[1].plot(air[:,0]/3600,air[:,2]); ax[1].set_ylabel('Cb∞'); ax[1].set_xlabel('time / h'); savefig('raw_boundary')
    # Q1 profiles
    x=np.linspace(0,2,q1['T'].shape[1]); fig,ax=plt.subplots(1,2,figsize=(7,3))
    for t in [100,600,1800]:
        j=int(np.argmin(abs(q1['t']-t))); ax[0].plot(x,q1['T'][j],label=f'{t}s'); ax[1].plot(x,q1['C'][j],label=f'{t}s')
    ax[0].set(xlabel='r / cm',ylabel='T / °C'); ax[1].set(xlabel='r / cm',ylabel='C'); ax[0].legend(frameon=False); ax[1].legend(frameon=False); savefig('result_q1_profiles')
    # Q2 heatmaps
    fig,ax=plt.subplots(1,2,figsize=(8,3)); xx=np.linspace(0,2,q2['T'].shape[1]);
    for a,key,title in zip(ax,['T','C'],['temperature','moisture']):
        im=a.contourf(xx,q2['t']/3600,q2[key],levels=24); a.set(xlabel='r / cm',ylabel='time / h',title=title); fig.colorbar(im,ax=a)
    savefig('result_q2_heatmaps')
    # Q3 threshold
    z=np.maximum(q3['C'].max(axis=1),reconstruct_center(q3['C'])); e=json.loads((OUT/'event_summary.json').read_text()); fig,ax=plt.subplots(figsize=(6,3)); ax.plot(q3['t']/3600,z); ax.axhline(.15,ls='--');
    if e['q3_continuous_h'] is not None: ax.axvline(e['q3_continuous_h'],ls=':'); ax.set(xlabel='time / h',ylabel='max C'); savefig('result_q3_threshold')
    # Q4 shrinkage/profile
    fig,ax=plt.subplots(figsize=(6,3)); ax.plot(q4['t']/3600,[Rmeas(t)*100 for t in q4['t']]); ax.set(xlabel='time / h',ylabel='R / cm'); savefig('result_q4_radius')
    print('generated figures',len(list(FIG.glob('*.png'))))
if __name__=='__main__': main()
