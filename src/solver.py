"""Unified conservative finite-volume solvers for CUMCM 2026 A.

The implementation keeps the problem statement's empirical closures as the
baseline.  A surface-only latent-heat switch and a coarse axisymmetric solver
are provided as explicitly labelled scenarios/diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import platform
import sys
from typing import Callable, Optional

import numpy as np
from openpyxl import load_workbook
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "A题" / "附件"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    kind: str = "q2"
    n: int = 160
    dt: float = 60.0
    h: float = 25.0
    hm: float = 8e-7
    latent: bool = False
    Lv: float = 2.4e6
    rho_dry: float = 650.0
    moving: bool = False
    gamma: float = 0.0
    end_mode: str = "adiabatic"


def read_xlsx(path: Path):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.values)
    wb.close()
    return np.asarray(rows[1:], dtype=float), rows[0]


air, _ = read_xlsx(DATA / "附件1.xlsx")
radius, _ = read_xlsx(DATA / "附件2.xlsx")
bt, bT, bC = air[:, 0], air[:, 1], air[:, 2]
rp = PchipInterpolator(radius[:, 0], radius[:, 1], extrapolate=False)


def Tinf(t: float) -> float:
    return float(np.interp(t, bt, bT)) if t <= bt[-1] else float(bT[-1])


def Cbinf(t: float) -> float:
    return float(np.interp(t, bt, bC)) if t <= bt[-1] else float(bC[-1])


def Rmeas(t: float) -> float:
    return float(rp(t)) * 1e-2 if t <= radius[-1, 0] else float(radius[-1, 1]) * 1e-2


def axial_scale(t: float, gamma: float, moving: bool) -> float:
    if not moving or gamma == 0:
        return 1.0
    return (Rmeas(t) / (radius[0, 1] * 1e-2)) ** gamma


def props(kind: str, T: np.ndarray, C: np.ndarray):
    C = np.asarray(C, dtype=float)
    if np.any(C <= 0):
        raise FloatingPointError("negative or zero dry-basis moisture in property evaluation")
    TK = np.asarray(T) + 273.15
    if np.any(TK <= 0):
        raise FloatingPointError("non-positive absolute temperature")
    if kind == "q1":
        return (np.full_like(C, 820.0), np.full_like(C, 2600.0),
                np.full_like(C, 0.36), 7e-9 * np.exp(-0.89 / C))
    if kind == "q2":
        return (650 + 128 * C, 1450 + 2736 * C / (C + 1),
                0.21 + 0.38 * C / (C + 1),
                2.4e-3 * np.exp(-0.45 / C - 3850 / TK))
    if kind == "q4":
        return (760 + 90 * C, 1850 + 2150 * C / (C + 1),
                0.12 + 0.20 * C / (C + 1),
                4.2e-4 * np.exp(-0.30 / C - 3850 / TK))
    raise ValueError(kind)


def harmonic(a, b):
    return 2 * a * b / (a + b)


def reconstruct_center(values):
    values = np.asarray(values)
    if values.shape[-1] < 2:
        return values[..., 0]
    return (9 * values[..., 0] - values[..., 1]) / 8


def _surface_state(cfg: Config, Tlast, Clast, t, dr):
    _, _, k, D = props(cfg.kind, np.array([Tlast]), np.array([Clast]))
    k, D = float(k[0]), float(D[0])
    gb = cfg.h / (1 + cfg.h * dr / (2 * k))
    gm = cfg.hm / (1 + cfg.hm * dr / (2 * D))
    Cs = (D * Clast + cfg.hm * dr / 2 * Cbinf(t)) / (D + cfg.hm * dr / 2)
    rho_d = cfg.rho_dry
    Jw = rho_d * gm * (Clast - Cbinf(t)) if cfg.latent else 0.0
    return gb, gm, Cs, Jw


def _thomas(lower, diag, upper, rhs):
    """Solve a tridiagonal system by the Thomas algorithm."""
    n = len(diag)
    if n == 1:
        return np.array([rhs[0] / diag[0]], dtype=float)
    a, b, c, d = map(np.asarray, (lower, diag, upper, rhs))
    cp = np.empty(n - 1, dtype=float); dp = np.empty(n, dtype=float)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1: cp[i] = c[i] / den
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / den
    x = np.empty(n, dtype=float); x[-1] = dp[-1]
    for i in range(n - 2, -1, -1): x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def step_1d(cfg: Config, t: float, T: np.ndarray, C: np.ndarray, dt: float):
    """One fully implicit Euler step with Picard iteration on properties."""
    n = cfg.n
    R = Rmeas(t + dt) if cfg.moving else 0.02
    xi_f = np.arange(n + 1, dtype=float) / n
    dr = R / n
    vol = np.pi * R * R * (xi_f[1:] ** 2 - xi_f[:-1] ** 2)
    area = 2 * np.pi * R * xi_f
    Tn, Cn = T.copy(), C.copy()
    for it in range(80):
        rho, cp, k, D = props(cfg.kind, Tn, Cn)
        kf = np.empty(n + 1); Df = np.empty(n + 1)
        kf[0], kf[-1], Df[0], Df[-1] = k[0], k[-1], D[0], D[-1]
        if n > 1:
            kf[1:-1] = harmonic(k[:-1], k[1:])
            Df[1:-1] = harmonic(D[:-1], D[1:])
        gt = area[1:-1] * kf[1:-1] / dr if n > 1 else np.empty(0)
        gm = area[1:-1] * Df[1:-1] / dr if n > 1 else np.empty(0)
        gb, gmb, Cs, Jw = _surface_state(cfg, Tn[-1], Cn[-1], t + dt, dr)
        lower = -gt.copy(); upper = -gt.copy()
        diag = rho * cp * vol / dt
        rhs = diag * T
        if n > 1:
            diag[:-1] += gt; diag[1:] += gt
        diag[-1] += area[-1] * gb
        rhs[-1] += area[-1] * gb * Tinf(t + dt) - (area[-1] * cfg.Lv * Jw if cfg.latent else 0.0)
        Tnew = _thomas(lower, diag, upper, rhs)

        lower = -gm.copy(); upper = -gm.copy()
        diag = vol / dt
        rhs = diag * C
        if n > 1:
            diag[:-1] += gm; diag[1:] += gm
        diag[-1] += area[-1] * gmb
        rhs[-1] += area[-1] * gmb * Cbinf(t + dt)
        Cnew = _thomas(lower, diag, upper, rhs)
        err = max(float(np.max(np.abs(Tnew - Tn))), float(np.max(np.abs(Cnew - Cn))))
        Tn, Cn = Tnew, Cnew
        if err < 2e-8:
            break
    else:
        raise RuntimeError(f"Picard iteration failed at t={t + dt:g}s")
    if np.min(Cn) <= 0 or np.min(Tn + 273.15) <= 0:
        raise FloatingPointError("implicit step left physical domain")
    return Tn, Cn, it + 1, float(Jw), float(area[-1] * cfg.h * (Tinf(t + dt) - Tn[-1]))


def solve_1d(cfg: Config, t_end: float, output_times=None, y0=None, event_threshold=None):
    """Integrate with implicit steps and linearly sample requested output times."""
    if output_times is None: output_times = np.arange(0, t_end + 1e-9, cfg.dt)
    output_times = np.asarray(output_times, dtype=float); n = cfg.n
    T = np.full(n, 28.0) if y0 is None else np.asarray(y0[:n], dtype=float).copy()
    C = np.full(n, 2.55) if y0 is None else np.asarray(y0[n:], dtype=float).copy()
    cur=0.0; ts=[0.0]; Ts=[T.copy()]; Cs=[C.copy()]; its=[0]; Js=[0.0]; Qs=[0.0]
    def event_value(tt, TT, CC):
        R = Rmeas(tt) if cfg.moving else 0.02; dr = R / n
        _, _, Cs, _ = _surface_state(cfg, TT[-1], CC[-1], tt, dr)
        return max(float(np.max(CC)), float(reconstruct_center(CC)), float(Cs))
    prev=event_value(0.0, T, C); event=None
    while cur < t_end - 1e-10:
        dt=min(cfg.dt,t_end-cur); T,C,it,Jw,qconv=step_1d(cfg,cur,T,C,dt); cur+=dt
        ts.append(cur); Ts.append(T.copy()); Cs.append(C.copy()); its.append(it); Js.append(Jw); Qs.append(qconv)
        now=event_value(cur, T, C)
        if event_threshold is not None and event is None and prev>=event_threshold and now<event_threshold:
            event=cur-dt+(prev-event_threshold)/(prev-now)*dt
            # The requested endpoint is the first strict crossing; no need to
            # integrate the already-completed tail of a million-second horizon.
            break
        prev=now
    ts=np.asarray(ts); Ts=np.asarray(Ts); Cs=np.asarray(Cs)
    if event is not None:
        output_times = output_times[output_times <= ts[-1] + 1e-8]
    def sample(arr):
        out=np.empty((len(output_times),n))
        for j,t in enumerate(output_times):
            k=min(np.searchsorted(ts,t,side='right'),len(ts)-1)
            if k==0 or ts[k]==t: out[j]=arr[k]
            else:
                w=(t-ts[k-1])/(ts[k]-ts[k-1]); out[j]=(1-w)*arr[k-1]+w*arr[k]
        return out
    return {'t':output_times,'T':sample(Ts),'C':sample(Cs),'iterations':np.interp(output_times,ts,its),'Jw':np.interp(output_times,ts,Js),'qconv':np.interp(output_times,ts,Qs),'event':event}


def interp_radial(field, radii_m, R=0.02):
    x = (np.arange(field.shape[-1]) + 0.5) * R / field.shape[-1]
    ans = np.empty((field.shape[0], len(radii_m)))
    for j, r in enumerate(np.asarray(radii_m)):
        if r <= 0: ans[:, j] = reconstruct_center(field)
        elif r >= R: ans[:, j] = field[:, -1]
        else: ans[:, j] = np.array([np.interp(r, x, row) for row in field])
    return ans


def surface_values(cfg: Config, data):
    T, C, tt = data["T"], data["C"], data["t"]
    outT = np.empty(len(tt)); outC = np.empty(len(tt)); outJ = np.empty(len(tt))
    for j, t in enumerate(tt):
        R = Rmeas(t) if cfg.moving else 0.02
        dr = R / cfg.n
        gb, gmb, Cs, Jw = _surface_state(cfg, T[j, -1], C[j, -1], t, dr)
        outT[j] = (gb * T[j, -1] + cfg.h * 0 + cfg.h * Tinf(t) * 0)  # overwritten below
        outT[j] = (float(props(cfg.kind, np.array([T[j, -1]]), np.array([C[j, -1]]))[2][0]) * T[j, -1] + cfg.h * dr / 2 * Tinf(t)) / (float(props(cfg.kind, np.array([T[j, -1]]), np.array([C[j, -1]]))[2][0]) + cfg.h * dr / 2)
        outC[j] = (float(props(cfg.kind, np.array([T[j, -1]]), np.array([C[j, -1]]))[3][0]) * C[j, -1] + cfg.hm * dr / 2 * Cbinf(t)) / (float(props(cfg.kind, np.array([T[j, -1]]), np.array([C[j, -1]]))[3][0]) + cfg.hm * dr / 2)
        outJ[j] = Jw
    return outT, outC, outJ


def solve_2d(kind="q4", t_end=3600.0, nr=12, nz=24, dt=20.0, end_mode="adiabatic"):
    """Coarse axisymmetric r-z reference solve for dimensionality checks.

    It uses the same empirical properties and side Robin boundary.  The ends
    are adiabatic by default; a convective end mode is available as a clearly
    separate scenario.  The geometry is fixed here so this solver is a
    dimensionality diagnostic rather than the Q4 production model.
    """
    dr, dz = 0.02 / nr, 0.25 / nz
    ncell = nr * nz
    y0 = np.r_[np.full(ncell, 28.0), np.full(ncell, 2.55)]
    def rhs(t, y):
        T = y[:ncell].reshape(nz, nr); C = y[ncell:].reshape(nz, nr)
        rho, cp, k, D = props(kind, T, C)
        dT = np.zeros_like(T); dC = np.zeros_like(C)
        for iz in range(nz):
            for ir in range(nr):
                rv = (ir + 0.5) * dr
                ar = 2 * np.pi * rv * 0.25
                vol = np.pi * (((ir + 1) * dr) ** 2 - (ir * dr) ** 2) * dz
                if ir > 0:
                    rf = ir * dr; af = 2 * np.pi * rf * 0.25
                    kf = harmonic(k[iz, ir - 1], k[iz, ir]); Df = harmonic(D[iz, ir - 1], D[iz, ir])
                    dT[iz, ir] += af * kf * (T[iz, ir - 1] - T[iz, ir]) / dr / (rho[iz, ir] * cp[iz, ir] * vol)
                    dC[iz, ir] += af * Df * (C[iz, ir - 1] - C[iz, ir]) / dr / vol
                if ir < nr - 1:
                    rf = (ir + 1) * dr; af = 2 * np.pi * rf * 0.25
                    kf = harmonic(k[iz, ir + 1], k[iz, ir]); Df = harmonic(D[iz, ir + 1], D[iz, ir])
                    dT[iz, ir] += af * kf * (T[iz, ir + 1] - T[iz, ir]) / dr / (rho[iz, ir] * cp[iz, ir] * vol)
                    dC[iz, ir] += af * Df * (C[iz, ir + 1] - C[iz, ir]) / dr / vol
                else:
                    gb = cfg_h = 25.0 / (1 + 25.0 * dr / (2 * k[iz, ir]))
                    gmb = 8e-7 / (1 + 8e-7 * dr / (2 * D[iz, ir]))
                    dT[iz, ir] += ar * gb * (Tinf(t) - T[iz, ir]) / (rho[iz, ir] * cp[iz, ir] * vol)
                    dC[iz, ir] += ar * gmb * (Cbinf(t) - C[iz, ir]) / vol
                if iz > 0:
                    afz = np.pi * (((ir + 1) * dr) ** 2 - (ir * dr) ** 2)
                    kf = harmonic(k[iz - 1, ir], k[iz, ir]); Df = harmonic(D[iz - 1, ir], D[iz, ir])
                    dT[iz, ir] += afz * kf * (T[iz - 1, ir] - T[iz, ir]) / dz / (rho[iz, ir] * cp[iz, ir] * vol)
                    dC[iz, ir] += afz * Df * (C[iz - 1, ir] - C[iz, ir]) / dz / vol
                if iz < nz - 1:
                    afz = np.pi * (((ir + 1) * dr) ** 2 - (ir * dr) ** 2)
                    kf = harmonic(k[iz + 1, ir], k[iz, ir]); Df = harmonic(D[iz + 1, ir], D[iz, ir])
                    dT[iz, ir] += afz * kf * (T[iz + 1, ir] - T[iz, ir]) / dz / (rho[iz, ir] * cp[iz, ir] * vol)
                    dC[iz, ir] += afz * Df * (C[iz + 1, ir] - C[iz, ir]) / dz / vol
        return np.r_[dT.ravel(), dC.ravel()]
    sol = solve_ivp(rhs, (0, t_end), y0, method="BDF", rtol=2e-5, atol=1e-7,
                    t_eval=[t_end], max_step=dt)
    if not sol.success:
        raise RuntimeError(sol.message)
    return {"t": np.array([t_end]), "T": sol.y[:ncell, -1].reshape(1, nz, nr),
            "C": sol.y[ncell:, -1].reshape(1, nz, nr), "nr": nr, "nz": nz}


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

