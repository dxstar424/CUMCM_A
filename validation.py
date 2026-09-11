"""Independent lightweight validation gates for the production solver."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from src.solver import Config, solve_1d, solve_2d, reconstruct_center

def main():
    out={}
    a=solve_1d(Config(kind='q1',n=24,dt=10,h=0,hm=0),100,np.arange(0,101,20.)); out['zero_flux_max_C_error']=float(abs(a['C']-2.55).max())
    f=solve_1d(Config(kind='q4',n=24,dt=30),300,np.arange(0,301,60.)); m=solve_1d(Config(kind='q4',n=24,dt=30,moving=True,gamma=0),300,np.arange(0,301,60.)); out['fixed_moving_max_error']=float(abs(f['C']-m['C']).max())
    d2=solve_2d('q4',600,nr=6,nz=8,dt=30); out['2d_shape']=[int(x) for x in d2['C'].shape]
    out['positive']=bool(np.min(f['C'])>0 and np.min(f['T'])>-273.15)
    Path('results/validation_report.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
