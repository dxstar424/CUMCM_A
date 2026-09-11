import json,sys,time
sys.path.insert(0,'.')
from src.model import solve
N=80
cases={}
for label,kind,moving in [('A','q2',False),('B','q2',True),('C','q4',False),('D','q4',True)]:
    s=solve(kind,(0,2000000),n=N,moving=moving,event=True,max_step=1800)
    assert s.success and len(s.t_events[0])
    cases[label]={'time_s':float(s.t_events[0][0]),'time_h':float(s.t_events[0][0]/3600),'kind':kind,'moving':moving,'N':N}
cases['geometry_effect_q2_s']=cases['B']['time_s']-cases['A']['time_s']
cases['geometry_effect_q4_s']=cases['D']['time_s']-cases['C']['time_s']
cases['interaction_s']=(cases['D']['time_s']-cases['C']['time_s'])-(cases['B']['time_s']-cases['A']['time_s'])
json.dump(cases,open('results/attribution.json','w'),ensure_ascii=False,indent=2)
print(json.dumps(cases,ensure_ascii=False))
