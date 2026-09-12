"""将已生成的工作簿数值单元格内部文本统一为四位小数。"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import os, tempfile, xml.etree.ElementTree as ET

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'; ET.register_namespace('',NS)
RES=Path(__file__).resolve().parents[1]/'results'

def round_book(path):
    fd,tmp=tempfile.mkstemp(suffix='.xlsx',dir=str(path.parent)); os.close(fd)
    with ZipFile(path,'r') as zin, ZipFile(tmp,'w',ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data=zin.read(item.filename)
            if item.filename.startswith('xl/worksheets/sheet') and item.filename.endswith('.xml'):
                root=ET.fromstring(data)
                for c in root.iter('{%s}c'%NS):
                    if c.attrib.get('t') is not None: continue
                    v=c.find('{%s}v'%NS)
                    if v is None or v.text is None: continue
                    try: v.text=f'{float(v.text):.4f}'
                    except ValueError: pass
                data=ET.tostring(root,encoding='utf-8',xml_declaration=True)
            zout.writestr(item,data)
    os.replace(tmp,path)

if __name__=='__main__':
    for name in ['result1.xlsx','result2.xlsx','result3.xlsx','result4.xlsx','正文六个小表.xlsx','正文六表.xlsx']:
        round_book(RES/name); print(name)
