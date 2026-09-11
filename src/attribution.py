"""Current-solver 2x2 attribution: physical properties x measured shrinkage."""
import json
try:
    from src.solver import Config, solve_1d
except ModuleNotFoundError:
    from solver import Config, solve_1d

def main():
    cases={}
    for label,kind,moving in [('A','q2',False),('B','q2',True),('C','q4',False),('D','q4',True)]:
        d=solve_1d(Config(kind=kind,n=80,dt=120,moving=moving),500000,event_threshold=.15)
        cases[label]={'time_s':d['event'],'time_h':None if d['event'] is None else d['event']/3600,'kind':kind,'moving':moving,'N':80}
    cases['geometry_effect_q2_s']=cases['B']['time_s']-cases['A']['time_s']; cases['geometry_effect_q4_s']=cases['D']['time_s']-cases['C']['time_s']; cases['interaction_s']=(cases['D']['time_s']-cases['C']['time_s'])-(cases['B']['time_s']-cases['A']['time_s'])
    with open('results/attribution.json','w') as f: json.dump(cases,f,ensure_ascii=False,indent=2)
    print(json.dumps(cases,ensure_ascii=False))
if __name__=='__main__': main()
