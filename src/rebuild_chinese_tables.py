"""按题目正文表格版式生成六个中文小表。"""
from pathlib import Path
import csv, json, shutil, sys
import numpy as np
import xlsxwriter

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RES = REPO / "results"
SPECS = [
    ("表一_问题一温度.csv", "表一问题一温度", "时间/s", "温度"),
    ("表二_问题一含水率.csv", "表二问题一含水率", "时间/s", "含水率"),
    ("表三_问题二温度.csv", "表三问题二温度", "时间/h", "温度"),
    ("表四_问题二含水率.csv", "表四问题二含水率", "时间/h", "含水率"),
    ("表五_问题三长时含水率.csv", "表五问题三含水率", "时间/h", "含水率"),
    ("表六_问题四收缩含水率.csv", "表六问题四含水率", "时间/h", "含水率"),
]

def _sample(field, radii_cm, index, moving_radius=None):
    f = np.asarray(field[index], dtype=float)
    x = np.arange(f.size, dtype=float) / (f.size - 1)
    if moving_radius is None:
        return [float(np.interp(r / 2.0, x, f)) for r in radii_cm]
    vals = [float(np.interp(r / moving_radius, x, f)) if r <= moving_radius + 1e-12 else None for r in radii_cm]
    vals.append(float(f[-1]))
    return vals

def ensure_csvs():
    """从主模型数组补齐六个中文 CSV；已有文件时不改写。"""
    if all((RES / spec[0]).exists() for spec in SPECS):
        return
    try:
        from src import model
    except ModuleNotFoundError:
        import model
    radii = [0.0, 0.5, 1.0, 1.5, 2.0]
    headers_s = ["时间（秒）", "中心（0厘米）", "半径0.5厘米", "半径1.0厘米", "半径1.5厘米", "表面半径2.0厘米"]
    headers_h = ["时间（小时）", "中心（0厘米）", "半径0.5厘米", "半径1.0厘米", "半径1.5厘米", "表面半径2.0厘米"]
    def write(name, header, rows):
        with (RES / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for row in rows: w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
    q1 = np.load(RES / "q1.npz"); q2 = np.load(RES / "q2.npz"); q3 = np.load(RES / "q3.npz"); q4 = np.load(RES / "q4.npz")
    q1t = [100, 300, 600, 900, 1200, 1500, 1800]; q2h = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    def fixed_rows(d, times, key, scale):
        out=[]
        for t in times:
            j=int(np.argmin(abs(d["t"]-t*scale))); out.append([float(t), *_sample(d[key], radii, j)])
        return out
    write(SPECS[0][0], headers_s, fixed_rows(q1, q1t, "T", 1.0))
    write(SPECS[1][0], headers_s, fixed_rows(q1, q1t, "C", 1.0))
    write(SPECS[2][0], headers_h, fixed_rows(q2, q2h, "T", 3600.0))
    write(SPECS[3][0], headers_h, fixed_rows(q2, q2h, "C", 3600.0))
    event = json.loads((RES / "event_summary.json").read_text(encoding="utf-8"))
    q3h = list(np.arange(6.0, event["q3_first_strict_sample_s"] / 3600.0, 6.0)) + [event["q3_first_strict_sample_s"] / 3600.0]
    q4h = list(np.arange(6.0, event["q4_first_strict_sample_s"] / 3600.0, 6.0)) + [event["q4_first_strict_sample_s"] / 3600.0]
    rows=[]
    for h in q3h:
        j=int(np.argmin(abs(q3["t"]-h*3600))); rows.append([float(h), *_sample(q3["C"], radii, j)])
    write(SPECS[4][0], headers_h, rows)
    rows=[]
    for h in q4h:
        j=int(np.argmin(abs(q4["t"]-h*3600))); rr=float(model.radius(h*3600)*100); rows.append([float(h), *_sample(q4["C"], radii, j, rr)])
    write(SPECS[5][0], headers_h + ["药材表面"], rows)

def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]; body = []
    for row in rows[1:]:
        body.append([None if x == "" else round(float(x), 4) for x in row])
    return header, body

def build(path):
    wb = xlsxwriter.Workbook(str(path))
    wb.set_properties({"title": "正文六个小表", "comments": "按题目表格版式，中文表头，数值保留四位小数"})
    fmt_title = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
    fmt_head = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "border": 1})
    fmt_num = wb.add_format({"align": "center", "valign": "vcenter", "border": 1, "num_format": "0.0000"})
    fmt_blank = wb.add_format({"align": "center", "valign": "vcenter", "border": 1})
    checks = {}
    for fn, sheet_name, time_unit, value_kind in SPECS:
        header, body = read_table(RES / fn); ncols = len(header)
        ws = wb.add_worksheet(sheet_name); ws.set_column(0, ncols - 1, 14); ws.set_row(0, 22); ws.set_row(1, 20)
        distance_end = ncols - 2 if sheet_name == "表六问题四含水率" else ncols - 1
        ws.merge_range(0, 0, 1, 0, time_unit, fmt_title)
        ws.merge_range(0, 1, 0, distance_end, "到药材中心的距离/cm", fmt_title)
        for j, label in enumerate(header[1:], start=1):
            if j == ncols - 1 and sheet_name == "表六问题四含水率":
                ws.merge_range(0, j, 1, j, "药材表面", fmt_title)
            else:
                label = {"中心（0厘米）":"0", "半径0.5厘米":"0.5", "半径1.0厘米":"1.0", "半径1.5厘米":"1.5", "表面半径2.0厘米":"2.0"}.get(label, label)
                ws.write(1, j, label, fmt_head)
        for i, row in enumerate(body, start=2):
            for j, value in enumerate(row): ws.write(i, j, value, fmt_blank if value is None else fmt_num)
        ws.freeze_panes(2, 1); ws.autofilter(1, 0, len(body) + 1, ncols - 1)
        checks[sheet_name] = {"行数": len(body), "列数": ncols, "首行时间": body[0][0], "末行时间": body[-1][0], "数值位数": 4}
    wb.close(); return checks

if __name__ == "__main__":
    ensure_csvs(); checks = build(RES / "正文六个小表.xlsx"); shutil.copy2(RES / "正文六个小表.xlsx", RES / "正文六表.xlsx")
    (RES / "六个小表检查.json").write_text(json.dumps({"工作簿":"正文六个小表.xlsx", "检查":checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
