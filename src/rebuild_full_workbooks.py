"""将主模型数组写回题目完整结果工作簿，内部数值统一四舍五入四位。"""
from pathlib import Path
import numpy as np
import xlsxwriter
from scipy.interpolate import PchipInterpolator

HERE=Path(__file__).resolve().parent; REPO=HERE.parent; RES=REPO/'results'

def write_sheet(ws, head, num, blank, times, values, moving=False, radius_data=None):
    radii=np.arange(0,2.0001,0.1)
    headers=['时间/s']+[f'{x:.1f}' for x in radii]+(['药材表面'] if moving else [])
    ws.write_row(0,0,headers,head); ws.freeze_panes(1,1); ws.set_column(0,len(headers)-1,12)
    x=np.arange(values.shape[1],dtype=float)/(values.shape[1]-1)
    if moving:
        rt=PchipInterpolator(radius_data[:,0],radius_data[:,1]/100.0,extrapolate=True)
    for i,(t,f) in enumerate(zip(times,values),start=1):
        rr=float(rt(float(t))) if moving else 0.02
        vals=[]
        for r in radii/100:
            if r>rr+1e-12: vals.append(None)
            else: vals.append(float(np.interp(r/rr,x,f)) if moving else float(np.interp(r/0.02,x,f)))
        if moving: vals.append(float(f[-1]))
        ws.write_number(i,0,round(float(t),4),num)
        for j,v in enumerate(vals,start=1):
            if v is None: ws.write_blank(i,j,None,blank)
            else: ws.write_number(i,j,round(v,4),num)

def build():
    rad=np.load(RES/'inputs.npz')['rad']
    jobs=[('q1','result1.xlsx',False,True),('q2','result2.xlsx',False,True),('q3','result3.xlsx',False,False),('q4','result4.xlsx',True,False)]
    for q,name,moving,two in jobs:
        d=np.load(RES/f'{q}.npz'); times=d['t']; wb=xlsxwriter.Workbook(str(RES/name))
        head=wb.add_format({'bold':True,'align':'center','border':1}); num=wb.add_format({'num_format':'0.0000','align':'center','border':1}); blank=wb.add_format({'align':'center','border':1})
        if two:
            write_sheet(wb.add_worksheet('温度'),head,num,blank,times,d['T'],False,rad)
            write_sheet(wb.add_worksheet('水分浓度'),head,num,blank,times,d['C'],False,rad)
        else:
            write_sheet(wb.add_worksheet('Sheet1'),head,num,blank,times,d['C'],moving,rad)
        wb.close()
        print(name,len(times),times[0],times[-1])
if __name__=='__main__': build()
