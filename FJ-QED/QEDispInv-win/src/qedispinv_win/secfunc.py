"""本征方程与世俗函数计算。"""

from __future__ import annotations

import math

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _dnka(
    wvno2: float,
    gam: float,
    gammk: float,
    rho: float,
    a0: float,
    cpcq: float,
    cpy: float,
    cpz: float,
    cqw: float,
    cqx: float,
    xy: float,
    xz: float,
    wy: float,
    wz: float,
) -> np.ndarray:
    ca = np.zeros((5, 5), dtype=np.float64)
    gamm1 = gam - 1.0
    twgm1 = gam + gamm1
    gmgmk = gam * gammk
    gmgm1 = gam * gamm1
    gm1sq = gamm1 * gamm1
    rho2 = rho * rho
    a0pq = a0 - cpcq
    t = -2.0 * wvno2

    ca[0, 0] = cpcq - 2.0 * gmgm1 * a0pq - gmgmk * xz - wvno2 * gm1sq * wy
    ca[0, 1] = (wvno2 * cpy - cqx) / rho
    ca[0, 2] = -(twgm1 * a0pq + gammk * xz + wvno2 * gamm1 * wy) / rho
    ca[0, 3] = (cpz - wvno2 * cqw) / rho
    ca[0, 4] = -(2.0 * wvno2 * a0pq + xz + wvno2 * wvno2 * wy) / rho2

    ca[1, 0] = (gmgmk * cpz - gm1sq * cqw) * rho
    ca[1, 1] = cpcq
    ca[1, 2] = gammk * cpz - gamm1 * cqw
    ca[1, 3] = -wz
    ca[1, 4] = ca[0, 3]

    ca[3, 0] = (gm1sq * cpy - gmgmk * cqx) * rho
    ca[3, 1] = -xy
    ca[3, 2] = gamm1 * cpy - gammk * cqx
    ca[3, 3] = cpcq
    ca[3, 4] = ca[0, 1]

    ca[4, 0] = -(
        2.0 * gmgmk * gm1sq * a0pq + gmgmk * gmgmk * xz + gm1sq * gm1sq * wy
    ) * rho2
    ca[4, 1] = ca[3, 0]
    ca[4, 2] = -(
        gammk * gamm1 * twgm1 * a0pq
        + gam * gammk * gammk * xz
        + gamm1 * gm1sq * wy
    ) * rho
    ca[4, 3] = ca[1, 0]
    ca[4, 4] = ca[0, 0]

    ca[2, 0] = t * ca[4, 2]
    ca[2, 1] = t * ca[3, 2]
    ca[2, 2] = a0 + 2.0 * (cpcq - ca[0, 0])
    ca[2, 3] = t * ca[1, 2]
    ca[2, 4] = t * ca[0, 2]
    return ca


@nb.njit(cache=True)
def _var(
    p: float,
    q: float,
    ra: float,
    rb: float,
    wvno: float,
    xka: float,
    xkb: float,
    dpth: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    pex = 0.0
    if wvno < xka:
        sinp = math.sin(p)
        w = sinp / ra
        x = -ra * sinp
        cosp = math.cos(p)
    elif wvno == xka:
        cosp = 1.0
        w = dpth
        x = 0.0
    else:
        pex = p
        fac = math.exp(-2.0 * p) if p < 16.0 else 0.0
        cosp = (1.0 + fac) * 0.5
        sinp = (1.0 - fac) * 0.5
        w = sinp / ra
        x = ra * sinp

    sex = 0.0
    if wvno < xkb:
        sinq = math.sin(q)
        y = sinq / rb
        z = -rb * sinq
        cosq = math.cos(q)
    elif wvno == xkb:
        cosq = 1.0
        y = dpth
        z = 0.0
    else:
        sex = q
        fac = math.exp(-2.0 * q) if q < 16.0 else 0.0
        cosq = (1.0 + fac) * 0.5
        sinq = (1.0 - fac) * 0.5
        y = sinq / rb
        z = rb * sinq

    exa = pex + sex
    a0 = math.exp(-exa) if exa < 60.0 else 0.0
    cpcq = cosp * cosq
    cpy = cosp * y
    cpz = cosp * z
    cqw = cosq * w
    cqx = cosq * x
    xy = x * y
    xz = x * z
    wy = w * y
    wz = w * z
    return w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz


@nb.njit(cache=True)
def evaluate_psv_numba(thk: np.ndarray, dns: np.ndarray, vs: np.ndarray, vp: np.ndarray, iwater: int, f: float, c: float) -> float:
    nl = vs.shape[0]
    omega = 2.0 * math.pi * f
    wvno = omega / c
    e = np.zeros(5, dtype=np.float64)

    wvno2 = wvno * wvno
    xka = omega / vp[nl - 1]
    xkb = omega / vs[nl - 1]
    wvnop = wvno + xka
    wvnom = abs(wvno - xka)
    ra = math.sqrt(wvnop * wvnom)
    wvnop = wvno + xkb
    wvnom = abs(wvno - xkb)
    rb = math.sqrt(wvnop * wvnom)
    t = vs[nl - 1] / omega

    gammk = 2.0 * t * t
    gam = gammk * wvno2
    gamm1 = gam - 1.0
    rho1 = dns[nl - 1]
    e[0] = rho1 * rho1 * (gamm1 * gamm1 - gam * gammk * ra * rb)
    e[1] = -rho1 * ra
    e[2] = rho1 * (gamm1 - gammk * ra * rb)
    e[3] = rho1 * rb
    e[4] = wvno2 - ra * rb

    for m in range(nl - 2, iwater - 1, -1):
        xka = omega / vp[m]
        xkb = omega / vs[m]
        t = vs[m] / omega
        gammk = 2.0 * t * t
        gam = gammk * wvno2
        wvnop = wvno + xka
        wvnom = abs(wvno - xka)
        ra = math.sqrt(wvnop * wvnom)
        wvnop = wvno + xkb
        wvnom = abs(wvno - xkb)
        rb = math.sqrt(wvnop * wvnom)
        dpth = thk[m]
        rho1 = dns[m]
        p = ra * dpth
        q = rb * dpth
        w, cosp, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz = _var(
            p, q, ra, rb, wvno, xka, xkb, dpth
        )
        ca = _dnka(wvno2, gam, gammk, rho1, a0, cpcq, cpy, cpz, cqw, cqx, xy, xz, wy, wz)
        e = ca.T @ e

    if iwater == 1:
        xka = omega / vp[0]
        wvnop = wvno + xka
        wvnom = abs(wvno - xka)
        ra = math.sqrt(wvnop * wvnom)
        dpth = thk[0]
        rho1 = dns[0]
        zul = 1.0e-5
        w, cosp, _, _, _, _, _, _, _, _, _, _ = _var(
            ra * dpth, zul, ra, zul, wvno, xka, xkb, dpth
        )
        return cosp * e[0] - rho1 * w * e[1]
    return e[0]


@nb.njit(cache=True)
def evaluate_sh_numba(thk: np.ndarray, dns: np.ndarray, vs: np.ndarray, iwater: int, f: float, c: float) -> float:
    nl = vs.shape[0]
    omega = 2.0 * math.pi * f
    wvno = omega / c
    beta1 = vs[nl - 1]
    rho1 = dns[nl - 1]
    xkb = omega / beta1
    wvnop = wvno + xkb
    wvnom = abs(wvno - xkb)
    rb = math.sqrt(wvnop * wvnom)
    e1 = rho1 * rb
    e2 = 1.0 / (beta1 * beta1)

    for m in range(nl - 2, iwater - 1, -1):
        beta1 = vs[m]
        rho1 = dns[m]
        xmu = rho1 * beta1 * beta1
        xkb = omega / beta1
        wvnop = wvno + xkb
        wvnom = abs(wvno - xkb)
        rb = math.sqrt(wvnop * wvnom)
        q = thk[m] * rb
        if wvno < xkb:
            sinq = math.sin(q)
            y = sinq / rb
            z = -rb * sinq
            cosq = math.cos(q)
        elif wvno == xkb:
            cosq = 1.0
            y = thk[m]
            z = 0.0
        else:
            fac = math.exp(-2.0 * q) if q < 16.0 else 0.0
            cosq = (1.0 + fac) * 0.5
            sinq = (1.0 - fac) * 0.5
            y = sinq / rb
            z = rb * sinq
        e10 = e1 * cosq + e2 * xmu * z
        e20 = e1 * y / xmu + e2 * cosq
        e1 = e10
        e2 = e20
    return e1


class SecularFunction:
    """世俗函数计算器。

    输入模型列定义:
        第 1 列: 层号
        第 2 列: 深度，单位 km
        第 3 列: 密度，单位 g/cm^3
        第 4 列: Vs，单位 km/s
        第 5 列: Vp，单位 km/s
    """

    def __init__(self, model: np.ndarray, sh: bool = False) -> None:
        model = np.asarray(model, dtype=np.float64)
        self.nl = int(model.shape[0])
        self.thk = model[1:, 1] - model[:-1, 1]
        self.dns = model[:, 2].astype(np.float64, copy=True)
        self.vs = model[:, 3].astype(np.float64, copy=True)
        self.vp = model[:, 4].astype(np.float64, copy=True)
        self.sh = bool(sh)
        self.is_water = bool(self.vs[0] == 0.0)
        self.iwater = 1 if self.is_water else 0
        if self.is_water:
            self.vs[0] = 1.0e-8

    def evaluate(self, f: float, c: float) -> float:
        """计算给定频率和相速度下的世俗函数值。

        输入:
            f:
                类型: `float`
                单位: `Hz`
                含义: 频率。
            c:
                类型: `float`
                单位: `km/s`
                含义: 相速度。
        输出:
            `float`
                单位: 无量纲
                含义: 世俗函数值，零点对应色散根。
        """
        if self.sh:
            return float(evaluate_sh_numba(self.thk, self.dns, self.vs, self.iwater, float(f), float(c)))
        return float(
            evaluate_psv_numba(
                self.thk, self.dns, self.vs, self.vp, self.iwater, float(f), float(c)
            )
        )

