"""将提交的文本结果统一为四位小数。"""
from pathlib import Path
import csv, json, re
ROOT=Path(__file__).resolve().parents[1]

def round_obj(x):
    if isinstance(x,float): return round(x,4)
    if isinstance(x,list): return [round_obj(v) for v in x]
    if isinstance(x,dict): return {k:round_obj(v) for k,v in x.items()}
    return x
for p in (ROOT/'results').rglob('*.json'):
    try:
        obj=json.loads(p.read_text(encoding='utf-8')); p.write_text(json.dumps(round_obj(obj),ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception: pass
for p in (ROOT/'results').rglob('*.csv'):
    rows=[]
    with p.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.reader(f):
            out=[]
            for x in row:
                if re.fullmatch(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',x):
                    out.append(f'{float(x):.4f}')
                else: out.append(x)
            rows.append(out)
    with p.open('w',encoding='utf-8-sig',newline='') as f: csv.writer(f).writerows(rows)
