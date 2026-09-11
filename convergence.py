"""Grid convergence using the current implicit-Euler production solver."""
import json
from src.solver import Config, solve_1d

def main():
    out={}
    for n in (40,80,120,160):
        for q,kind,moving in [('q3','q2',False),('q4','q4',True)]:
            d=solve_1d(Config(kind=kind,n=n,dt=120,moving=moving),500000,event_threshold=.15)
            out[f'{q}_N{n}']={'event_s':d['event'],'event_h':d['event']/3600}
    with open('results/convergence_current.json','w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
