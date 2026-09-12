"""Q2蒸发潜热有界情景对照实验。

保持最终基线的网格、时间步、物性、边界数据、算术面系数和Picard迭代不变，
仅在表面热边界中增加 -Lv*jw 项。该脚本不修改主模型。
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# 仅加载最终基线的原始输入快照，避免依赖Excel读取库。
raw = np.load(ROOT / "results" / "inputs.npz")
AIR = raw["air"]
R0, LENGTH, T0, C0 = 0.02, 0.25, 28.0, 2.55
N = 201
DR = R0 / (N - 1)
TOL = 1e-10
FACE_MODE = "arith"
H_SCALE = 1.0
HM_SCALE = 1.0
AIR_T, AIR_TA, AIR_CA = AIR[:, 0], AIR[:, 1], AIR[:, 2]
R, RF, V = (None, None, None)
def geometry(n=N):
    dr = R0 / (n - 1); r = np.arange(n) * dr
    rf = (np.arange(n - 1) + 0.5) * dr
    v = np.empty(n); v[0] = dr * dr / 8.0
    v[1:-1] = 0.5 * ((r[1:-1] + dr / 2) ** 2 - (r[1:-1] - dr / 2) ** 2)
    v[-1] = 0.5 * (R0 ** 2 - (R0 - dr / 2) ** 2)
    return r, rf, v
R, RF, V = geometry()
def environment(t, phase="q2"):
    return float(np.interp(float(t), AIR_T, AIR_TA)), float(np.interp(float(t), AIR_T, AIR_CA))
def coeffs(T, C):
    T = np.asarray(T); C = np.asarray(C)
    return (650 + 128*C, 1450 + 2736*C/(C+1),
            0.21 + 0.38*C/(C+1),
            2.4e-3*np.exp(-0.45/np.maximum(C,1e-12)-3850/(T+273.15)))
def thomas(lower, diag, upper, rhs):
    n=len(diag); a=np.asarray(lower,float).copy(); b=np.asarray(diag,float).copy(); c=np.asarray(upper,float).copy(); d=np.asarray(rhs,float).copy()
    for i in range(1,n):
        z=a[i]/b[i-1]; b[i]-=z*c[i-1]; d[i]-=z*d[i-1]
    x=np.empty(n,float); x[-1]=d[-1]/b[-1]
    for i in range(n-2,-1,-1): x[i]=(d[i]-c[i]*x[i+1])/b[i]
    return x
def assemble_moisture(old, Tprop, Cprop, dt, t):
    _, _, _, D = coeffs(Tprop, Cprop); _, Ca = environment(t)
    storage = V/dt; lower=np.zeros(N); upper=np.zeros(N); diag=storage.copy(); rhs=storage*old
    face=.5*(D[:-1]+D[1:]); conduct=face*RF/DR
    upper[0]=-conduct[0]; diag[0]+=conduct[0]
    for j in range(1,N-1):
        lower[j]=-conduct[j-1]; upper[j]=-conduct[j]; diag[j]+=conduct[j-1]+conduct[j]
    lower[-1]=-conduct[-1]; diag[-1]+=conduct[-1]+R0*(8e-7*HM_SCALE); rhs[-1]+=R0*(8e-7*HM_SCALE)*Ca
    return lower,diag,upper,rhs

OUT = HERE.parent / "results" / "q2_latent_compare"
OUT.mkdir(parents=True, exist_ok=True)


def assemble_latent(old, T_for_props, C_for_props, dt, t, Lv, rho_d):
    """组装含表面蒸发潜热的Q2温度方程。"""
    rho, cp, k, D = coeffs(T_for_props, C_for_props)
    kap = k
    bnd, _ = environment(t, 'q2')
    storage = V * (rho * cp) / dt
    lower = np.zeros(N); upper = np.zeros(N)
    diag = storage.copy(); rhs = storage * old
    if FACE_MODE == 'harm':
        face = 2.0 * kap[:-1] * kap[1:] / np.maximum(kap[:-1] + kap[1:], 1e-300)
    else:
        face = 0.5 * (kap[:-1] + kap[1:])
    conduct = face * RF / DR
    upper[0] = -conduct[0]; diag[0] += conduct[0]
    for j in range(1, N - 1):
        lower[j] = -conduct[j - 1]; upper[j] = -conduct[j]
        diag[j] += conduct[j - 1] + conduct[j]
    # 当前Picard迭代中的表面含水率用于蒸发通量。
    _, Ca = environment(t, 'q2')
    jw = rho_d * (8e-7 * HM_SCALE) * max(float(C_for_props[-1]) - float(Ca), 0.0)
    qlat = Lv * jw
    lower[-1] = -conduct[-1]
    diag[-1] += conduct[-1] + R0 * (25.0 * H_SCALE)
    rhs[-1] += R0 * ((25.0 * H_SCALE) * bnd - qlat)
    return lower, diag, upper, rhs, jw, qlat


def fixed_step_latent(T, C, dt, t, Lv, rho_d):
    Tn, Cn = T.copy(), C.copy()
    for it in range(1, 501):
        lo, di, up, rhs, _, _ = assemble_latent(T, Tn, Cn, dt, t, Lv, rho_d)
        Tnext = thomas(lo, di, up, rhs)
        lo, di, up, rhs = assemble_moisture(C, Tnext, Cn, dt, t)
        Cnext = thomas(lo, di, up, rhs)
        err = max(float(np.max(np.abs(Tnext - Tn))), float(np.max(np.abs(Cnext - Cn))))
        Tn, Cn = Tnext, Cnext
        if err < TOL:
            _, Ca = environment(t, 'q2')
            jw = rho_d * (8e-7 * HM_SCALE) * max(float(Cn[-1]) - float(Ca), 0.0)
            return Tn, Cn, it, jw, Lv * jw
    raise RuntimeError(f"含潜热Q2 Picard未收敛: t={t:.4f}, err={err:.6e}")


def run_latent(end, dt, Lv, rho_d):
    times = np.arange(0.0, end + 1e-9, dt)
    T = np.full(N, T0); C = np.full(N, C0)
    out_t = [0.0]; out_T = [T.copy()]; out_C = [C.copy()]
    iterations = [0]; jws = [0.0]; qlats = [0.0]; qconvs = [0.0]; qnets = [0.0]
    for t in times[1:]:
        T, C, it, jw, qlat = fixed_step_latent(T, C, dt, float(t), Lv, rho_d)
        Ta, _ = environment(float(t), 'q2')
        qconv = 25.0 * H_SCALE * (Ta - float(T[-1]))
        out_t.append(float(t)); out_T.append(T.copy()); out_C.append(C.copy())
        iterations.append(it); jws.append(jw); qlats.append(qlat); qconvs.append(qconv); qnets.append(qconv - qlat)
    return {
        't': np.asarray(out_t), 'T': np.asarray(out_T), 'C': np.asarray(out_C),
        'iterations': np.asarray(iterations), 'jw': np.asarray(jws),
        'q_lat': np.asarray(qlats), 'q_conv': np.asarray(qconvs), 'q_net': np.asarray(qnets),
        'Lv': float(Lv), 'rho_d': float(rho_d), 'dt': float(dt), 'end': float(end),
    }


def interp(field, radii_cm):
    return np.interp(np.asarray(radii_cm, float) / 100.0, R, field)


def selected_table(base, latent, field, times_s, radii_cm):
    rows = []
    for t in times_s:
        i0 = int(np.argmin(abs(base['t'] - t)))
        i1 = int(np.argmin(abs(latent['t'] - t)))
        b = interp(base[field][i0], radii_cm)
        l = interp(latent[field][i1], radii_cm)
        for r, vb, vl in zip(radii_cm, b, l):
            rows.append({
                '时间_s': float(t), '半径_cm': float(r),
                '无潜热': float(vb), '有潜热': float(vl), '差值_有潜热减无潜热': float(vl - vb)
            })
    return pd.DataFrame(rows)


def four_decimal_csv(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.4f')


def round4(value):
    if isinstance(value, dict):
        return {k: round4(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round4(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.4f}")
    return value

def selected_metrics_table(base, latent, times_s):
    rows = []
    for t in times_s:
        i0 = int(np.argmin(abs(base['t'] - t))); i1 = int(np.argmin(abs(latent['t'] - t)))
        rows.append({
            '时间_s': float(t),
            '中心温度_无潜热_C': float(base['T'][i0, 0]), '中心温度_有潜热_C': float(latent['T'][i1, 0]),
            '中心温度差_C': float(latent['T'][i1, 0] - base['T'][i0, 0]),
            '表面温度_无潜热_C': float(base['T'][i0, -1]), '表面温度_有潜热_C': float(latent['T'][i1, -1]),
            '表面温度差_C': float(latent['T'][i1, -1] - base['T'][i0, -1]),
            '中心含水率_无潜热': float(base['C'][i0, 0]), '中心含水率_有潜热': float(latent['C'][i1, 0]),
            '中心含水率差': float(latent['C'][i1, 0] - base['C'][i0, 0]),
            '表面含水率_无潜热': float(base['C'][i0, -1]), '表面含水率_有潜热': float(latent['C'][i1, -1]),
            '表面含水率差': float(latent['C'][i1, -1] - base['C'][i0, -1]),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--end', type=float, default=10800.0)
    ap.add_argument('--dt', type=float, default=1.0)
    ap.add_argument('--lv', type=float, default=2.4e6)
    ap.add_argument('--rho-d', type=float, default=650.0)
    args = ap.parse_args()
    base_npz = ROOT / "results" / 'q2.npz'
    base = {k: v for k, v in np.load(base_npz).items()}
    latent = run_latent(args.end, args.dt, args.lv, args.rho_d)
    np.savez_compressed(OUT / 'q2_含潜热.npz', **latent)

    times = [1800, 3600, 5400, 7200, 9000, 10800]
    radii = [0, 0.5, 1.0, 1.5, 2.0]
    tdf = selected_table(base, latent, 'T', times, radii)
    cdf = selected_table(base, latent, 'C', times, radii)
    four_decimal_csv(tdf, OUT / 'Q2_潜热对照_温度.csv')
    four_decimal_csv(cdf, OUT / 'Q2_潜热对照_含水率.csv')

    area = 2 * math.pi * R0 * LENGTH
    t = latent['t'];
    e_lat = float(np.trapezoid(latent['q_lat'] * area, t))
    e_conv = float(np.trapezoid(latent['q_conv'] * area, t))
    e_net = float(np.trapezoid(latent['q_net'] * area, t))
    base_idx = np.asarray([int(np.argmin(abs(base['t'] - x))) for x in latent['t']])
    base_T_match = base['T'][base_idx]; base_C_match = base['C'][base_idx]
    max_dt = float(np.max(np.abs(latent['T'] - base_T_match)))
    max_dc = float(np.max(np.abs(latent['C'] - base_C_match)))
    i_end = -1
    metrics = {
        '实验类型': 'Q2表面蒸发潜热有界情景对照',
        '对照组': '最终Q2基线，不含潜热边界项',
        '实验组': '仅增加 q_lat=L_v*rho_d*h_m*max(C_s-C_a,0)',
        '仿真终点_s': float(args.end), '时间步_s': float(args.dt), '节点数': int(N),
        '潜热情景_Lv_J_per_kg': float(args.lv), '干基密度情景_kg_per_m3': float(args.rho_d),
        '潜热参数来源说明': '题目附件与论文初稿未给出闭合值；本值为外部物性情景假设，不纳入主模型定解。',
        '终点中心温度_无潜热_C': float(base['T'][i_end,0]), '终点中心温度_有潜热_C': float(latent['T'][i_end,0]),
        '终点表面温度_无潜热_C': float(base['T'][i_end,-1]), '终点表面温度_有潜热_C': float(latent['T'][i_end,-1]),
        '终点中心含水率_无潜热': float(base['C'][i_end,0]), '终点中心含水率_有潜热': float(latent['C'][i_end,0]),
        '终点表面含水率_无潜热': float(base['C'][i_end,-1]), '终点表面含水率_有潜热': float(latent['C'][i_end,-1]),
        '全时空最大温度绝对差_C': max_dt, '全时空最大含水率绝对差': max_dc,
        '潜热累计能量_J': e_lat, '对流累计能量_J': e_conv, '净边界累计能量_J': e_net,
        '潜热占对流累计能量比例': float(e_lat / e_conv) if e_conv != 0 else float('nan'),
        '末时刻蒸发质量通量_kg_per_m2_s': float(latent['jw'][i_end]),
        '末时刻潜热通量_W_per_m2': float(latent['q_lat'][i_end]),
        '末时刻对流通量_W_per_m2': float(latent['q_conv'][i_end]),
        '末时刻净热通量_W_per_m2': float(latent['q_net'][i_end]),
        '是否纳入主模型': '否',
        '原因': '潜热参数不由题面唯一确定；强耦合情景可能产生湿球温度以下过冷，适合作为敏感性边界而非确定预测。',
    }
    metrics = round4(metrics)
    with open(OUT / 'Q2_潜热对照_指标.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    summary = selected_metrics_table(base, latent, times)
    four_decimal_csv(summary, OUT / 'Q2_潜热对照_关键时刻.csv')
    # 逐时能量与表面通量，便于论文复核
    flux = pd.DataFrame({
        '时间_s': latent['t'], '表面温度_C': latent['T'][:, -1], '表面含水率': latent['C'][:, -1],
        '环境温度_C': [environment(float(x), 'q2')[0] for x in latent['t']],
        '蒸发质量通量_kg_m-2_s': latent['jw'], '潜热通量_W_m-2': latent['q_lat'],
        '对流通量_W_m-2': latent['q_conv'], '净热通量_W_m-2': latent['q_net'],
    })
    four_decimal_csv(flux, OUT / 'Q2_潜热对照_表面通量.csv')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
