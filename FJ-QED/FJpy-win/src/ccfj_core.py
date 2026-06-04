"""FJpy-win 核心计算模块。

该模块对参考项目 `CC-FJpy-master` 中的主要计算路径做了
Windows 纯 Python 迁移，并在 F-J / F-H 积分与多窗加权环节上
加入 `numba` 并行优化，便于在不依赖 Cython/CUDA 的环境下复核结果。
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from numba import njit, prange
from scipy import signal


def _as_float32_1d(x: Iterable[float], name: str) -> np.ndarray:
    """将一维输入统一为 `float32`，便于后续数值核函数处理。"""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    return arr


def _as_int32_1d(x: Iterable[int], name: str) -> np.ndarray:
    """将索引类输入统一为一维 `int32`。"""
    arr = np.asarray(x, dtype=np.int32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    return arr


def _hilbert_rows(x: np.ndarray) -> np.ndarray:
    # 参考项目 CPU 版使用 `scipy.fftpack.hilbert`。
    # 这里采用 `signal.hilbert` 的虚部并额外取负号，以保持与原项目符号一致。
    return (-np.imag(signal.hilbert(x, axis=1))).astype(np.float32, copy=False)


def _normalize_by_freq(out: np.ndarray) -> np.ndarray:
    """按频率列归一化，保持与原项目输出风格一致。"""
    out = np.asarray(out, dtype=np.float32)
    for i in range(out.shape[1]):
        m = float(np.max(np.abs(out[:, i])))
        if m > 0:
            out[:, i] /= m
    return out


@njit(cache=True)
def _bessj0(x: float) -> float:
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        ans1 = 57568490574.0 + y * (
            -13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456))))
        )
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
        return ans1 / ans2
    z = 8.0 / ax
    y = z * z
    xx = ax - 0.785398164
    ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
    ans2 = -0.1562499995e-1 + y * (
        0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7))
    )
    return math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)


@njit(cache=True)
def _bessj1(x: float) -> float:
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        ans1 = x * (
            72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606)))))
        )
        ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        return ans1 / ans2
    z = 8.0 / ax
    y = z * z
    xx = ax - 2.356194491
    ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
    ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
    ans = math.sqrt(0.636619772 / ax) * (math.cos(xx) * ans1 - z * math.sin(xx) * ans2)
    if x < 0.0:
        ans = -ans
    return ans


@njit(cache=True)
def _bessy0(x: float) -> float:
    if x < 8.0:
        y = x * x
        ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))))
        ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y * 1.0))))
        return (ans1 / ans2) + 0.636619772 * _bessj0(x) * math.log(x)
    z = 8.0 / x
    y = z * z
    xx = x - 0.785398164
    ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
    ans2 = -0.1562499995e-1 + y * (
        0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7)))
    )
    return math.sqrt(0.636619772 / x) * (math.sin(xx) * ans1 + z * math.cos(xx) * ans2)


@njit(cache=True)
def _bessy1(x: float) -> float:
    if x < 8.0:
        y = x * x
        ans1 = x * (
            -0.4900604943e13
            + y * (0.1275274390e13 + y * (-0.5153438139e11 + y * (0.7349264551e9 + y * (-0.4237922726e7 + y * 0.8511937935e4))))
        )
        ans2 = 0.2499580570e14 + y * (
            0.4244419664e12 + y * (0.3733650367e10 + y * (0.2245904002e8 + y * (0.1020426050e6 + y * (0.3549632885e3 + y))))
        )
        return (ans1 / ans2) + 0.636619772 * (_bessj1(x) * math.log(x) - 1.0 / x)
    z = 8.0 / x
    y = z * z
    xx = x - 2.356194491
    ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
    ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
    return math.sqrt(0.636619772 / x) * (math.sin(xx) * ans1 + z * math.cos(xx) * ans2)


@njit(cache=True)
def _stvh0(x: float) -> float:
    pi = math.pi
    s = 1.0
    r = 1.0
    if x <= 20.0:
        a0 = 2.0 * x / pi
        for k in range(1, 61):
            kk = 2.0 * k + 1.0
            r = -r * x / kk * x / kk
            s += r
            if abs(r) < abs(s) * 1.0e-12:
                break
        return a0 * s
    km = int(0.5 * (x + 1.0))
    if x >= 50.0:
        km = 25
    for k in range(1, km + 1):
        val = 2.0 * k - 1.0
        r = -r * val * val / x / x
        s += r
        if abs(r) < abs(s) * 1.0e-12:
            break
    t = 4.0 / x
    t2 = t * t
    p0 = ((((-0.37043e-5 * t2 + 0.173565e-4) * t2 - 0.487613e-4) * t2 + 0.17343e-3) * t2 - 0.1753062e-2) * t2 + 0.3989422793
    q0 = t * (((((0.32312e-5 * t2 - 0.142078e-4) * t2 + 0.342468e-4) * t2 - 0.869791e-4) * t2 + 0.4564324e-3) * t2 - 0.0124669441)
    ta0 = x - 0.25 * pi
    by0 = 2.0 / math.sqrt(x) * (p0 * math.sin(ta0) + q0 * math.cos(ta0))
    return 2.0 / (pi * x) * s + by0


@njit(cache=True)
def _stvh1(x: float) -> float:
    pi = math.pi
    r = 1.0
    if x <= 20.0:
        s = 0.0
        a0 = -2.0 / pi
        for k in range(1, 61):
            r = -r * x * x / (4.0 * k * k - 1.0)
            s += r
            if abs(r) < abs(s) * 1.0e-12:
                break
        return a0 * s
    s = 1.0
    km = int(0.5 * x)
    if x > 50.0:
        km = 25
    for k in range(1, km + 1):
        r = -r * (4.0 * k * k - 1.0) / (x * x)
        s += r
        if abs(r) < abs(s) * 1.0e-12:
            break
    t = 4.0 / x
    t2 = t * t
    p1 = ((((0.42414e-5 * t2 - 0.20092e-4) * t2 + 0.580759e-4) * t2 - 0.223203e-3) * t2 + 0.29218256e-2) * t2 + 0.3989422819
    q1 = t * (((((-0.36594e-5 * t2 + 0.1622e-4) * t2 - 0.398708e-4) * t2 + 0.1064741e-3) * t2 - 0.63904e-3) * t2 + 0.0374008364)
    ta1 = x - 0.75 * pi
    by1 = 2.0 / math.sqrt(x) * (p1 * math.sin(ta1) + q1 * math.cos(ta1))
    return 2.0 / pi * (1.0 + s / (x * x)) + by1


@njit(cache=True)
def _kernel_values(x: float, func_code: int) -> tuple[float, float, float, float]:
    if func_code == 0:
        return _bessj0(x), _bessj1(x), _stvh0(x), _stvh1(x)
    return _bessy0(x), _bessy1(x), _stvh0(x), _stvh1(x)


@njit(cache=True, parallel=True)
def _integral_kernel_numba(uf: np.ndarray, r: np.ndarray, f: np.ndarray, c: np.ndarray, func_code: int) -> np.ndarray:
    nr = r.shape[0]
    nc = c.shape[0]
    nf = f.shape[0]
    out = np.zeros((nc, nf), dtype=np.float32)
    for i in prange(nf):
        fl = float(f[i])
        for j in range(nc):
            cl = float(c[j])
            k = 2.0 * math.pi * fl / cl
            if abs(k) < 1e-12:
                continue
            kernel = 0.0
            for ir in range(1, nr):
                g1 = float(uf[ir - 1, i])
                g2 = float(uf[ir, i])
                r1 = float(r[ir - 1])
                r2 = float(r[ir])
                dr0 = max(r2 - r1, 0.1)
                a = g1 - r1 * (g2 - g1) / dr0
                b = (g2 - g1) / dr0

                kr1 = k * r1
                kr2 = k * r2
                b01_0, b01_1, h01_0, h01_1 = _kernel_values(kr1, func_code)
                b02_0, b02_1, h02_0, h02_1 = _kernel_values(kr2, func_code)

                kernel += a * (r2 * b02_1 - r1 * b01_1) / k
                kernel += b * (r2 * r2 * b02_1 - r1 * r1 * b01_1) / k
                kernel += b * (r2 * b02_0 - r1 * b01_0) / (k * k)

                term2 = kr2 * b02_0 + math.pi * kr2 * (b02_1 * h02_0 - b02_0 * h02_1) / 2.0
                term1 = kr1 * b01_0 + math.pi * kr1 * (b01_1 * h01_0 - b01_0 * h01_1) / 2.0
                kernel += -b * (term2 - term1) / (k * k * k)
            out[j, i] = kernel
    return out


@njit(cache=True, parallel=True)
def _trap_kernel_numba(uf: np.ndarray, r: np.ndarray, f: np.ndarray, c: np.ndarray, func_code: int) -> np.ndarray:
    nr = r.shape[0]
    nc = c.shape[0]
    nf = f.shape[0]
    out = np.zeros((nc, nf), dtype=np.float32)
    for i in prange(nf):
        fl = float(f[i])
        for j in range(nc):
            cl = float(c[j])
            k = 2.0 * math.pi * fl / cl
            if abs(k) < 1e-12:
                continue
            kernel = 0.0
            for ir in range(1, nr):
                g1 = float(uf[ir - 1, i])
                g2 = float(uf[ir, i])
                r1 = float(r[ir - 1])
                r2 = float(r[ir])
                dr0 = max(r2 - r1, 0.1)
                if func_code == 0:
                    kernel += (g1 * _bessj0(k * r1) * r1 + g2 * _bessj0(k * r2) * r2) * dr0 * 0.5
                else:
                    kernel += (g1 * _bessy0(k * r1) * r1 + g2 * _bessy0(k * r2) * r2) * dr0 * 0.5
            out[j, i] = kernel
    return out


def _integral_kernel(uf: np.ndarray, r: np.ndarray, f: np.ndarray, c: np.ndarray, func: str) -> np.ndarray:
    """线性近似积分版本，对应原项目 `itype=1`。"""
    return _integral_kernel_numba(uf, r, f, c, 0 if func == "j" else 1)


def _trap_kernel(uf: np.ndarray, r: np.ndarray, f: np.ndarray, c: np.ndarray, func: str) -> np.ndarray:
    """梯形积分版本，对应原项目 `itype=0`。"""
    return _trap_kernel_numba(uf, r, f, c, 0 if func == "j" else 1)


def GetStationPairs(nsta: int) -> np.ndarray:
    """生成上三角台站对索引，顺序与参考项目保持一致。"""
    station_pair = []
    for ii in range(nsta):
        for jj in range(ii + 1, nsta):
            station_pair.append(ii)
            station_pair.append(jj)
    return np.asarray(station_pair, dtype=np.int32)


def CC(
    npts,
    nsta,
    nf,
    fftlen,
    Pairs,
    startend,
    data,
    overlaprate=0.0,
    nThreads=8,
    fstride=1,
    ifonebit=0,
    ifspecwhittenning=1,
):
    """计算多台站频域互相关。

    输入波形按台站平铺为 `(nsta, npts)`，输出为 `(nPairs, nf)` 的复数互相关频谱。
    """
    del nThreads

    if not (0 <= overlaprate < 1):
        raise ValueError("overlaprate must satisfy 0 <= overlaprate < 1.")

    pairs = _as_int32_1d(Pairs, "Pairs")
    startend = _as_int32_1d(startend, "startend")
    if startend.size != nsta * 2:
        raise ValueError("startend length must equal nsta * 2.")

    data = np.asarray(data, dtype=np.float32).reshape(nsta, npts)
    if pairs.size % 2 != 0:
        raise ValueError("Pairs length must be even.")

    npairs = pairs.size // 2
    fftlen = int(min(fftlen, npts))
    steplen = int(max(1, fftlen * (1.0 - overlaprate)))
    if fftlen < 1:
        raise ValueError("fftlen must be positive.")

    freq_idx = np.arange(0, nf * fstride, fstride, dtype=np.int64)
    nfreq_avail = fftlen // 2 + 1
    if freq_idx.size == 0:
        raise ValueError("nf must be positive.")
    if int(freq_idx[-1]) >= nfreq_avail:
        raise ValueError("nf and fstride exceed available FFT bins.")

    # one-bit 归一化仅改变符号，不改变窗口和频点布局。
    work = np.sign(data).astype(np.float32, copy=False) if ifonebit else data.copy()
    ncfsr = np.zeros((npairs, nf), dtype=np.float32)
    ncfsi = np.zeros((npairs, nf), dtype=np.float32)
    cc_numbers = np.zeros(npairs, dtype=np.int32)

    if fftlen >= npts:
        nshifts = 1
        fftlen = npts
    else:
        nshifts = 1 + (npts - fftlen) // steplen

    for k in range(nshifts):
        offset = k * steplen
        valid = (offset >= startend[0::2]) & (offset + fftlen <= startend[1::2])
        if not np.any(valid):
            continue

        # 每个平移窗口独立构造频谱，再按台站对累积互相关。
        spectra = np.zeros((nsta, nf), dtype=np.complex64)
        for ista in np.where(valid)[0]:
            segment = work[ista, offset : offset + fftlen]
            spec = np.fft.rfft(segment, n=fftlen)[freq_idx]
            if ifspecwhittenning:
                amp = np.maximum(np.abs(spec), 1e-8)
                spec = spec / amp
            spectra[ista] = spec.astype(np.complex64, copy=False)

        for ipair in range(npairs):
            a = int(pairs[2 * ipair])
            b = int(pairs[2 * ipair + 1])
            if valid[a] and valid[b]:
                sa = spectra[a]
                sb = spectra[b]
                ncfsr[ipair] += sa.real * sb.real + sa.imag * sb.imag
                ncfsi[ipair] += sa.real * sb.imag - sa.imag * sb.real
                cc_numbers[ipair] += 1

    for ipair in range(npairs):
        if cc_numbers[ipair] > 0:
            ncfsr[ipair] /= cc_numbers[ipair]
            ncfsi[ipair] /= cc_numbers[ipair]

    return ncfsr + 1j * ncfsi


def win(npts, Fs, T1, T2, taper=0.8):
    """生成单个时窗的余弦 taper 权重。"""
    n1 = max(0, int(np.floor(T1 * Fs)))
    n2 = min(int(npts), int(np.floor(T2 * Fs)))
    out = np.ones(int(npts), dtype=np.float32)
    delta = int(np.floor((n2 - n1) * (1 - taper) / 2))
    out[:n1] = 0
    out[n2:] = 0
    if delta > 0:
        for i in range(delta):
            val = np.sin(np.pi / 2 * i / delta)
            out[n1 + i] = val
            out[n2 - i - 1] = val
    return out


@njit(cache=True)
def _win_numba(npts: int, fs: float, t1: float, t2: float, taper: float) -> np.ndarray:
    n1 = max(0, int(math.floor(t1 * fs)))
    n2 = min(int(npts), int(math.floor(t2 * fs)))
    out = np.ones(int(npts), dtype=np.float32)
    delta = int(math.floor((n2 - n1) * (1.0 - taper) / 2.0))
    out[:n1] = 0.0
    out[n2:] = 0.0
    if delta > 0:
        for i in range(delta):
            val = math.sin(math.pi / 2.0 * i / delta)
            out[n1 + i] = val
            out[n2 - i - 1] = val
    return out


@njit(cache=True, parallel=True)
def _apply_windows_numba(u: np.ndarray, fs: float, winl_row: np.ndarray, winr_row: np.ndarray, taper: float) -> np.ndarray:
    nr, npts = u.shape
    out = np.zeros((nr, npts), dtype=np.float32)
    for j in prange(nr):
        tmp = _win_numba(npts, fs, float(winl_row[j]), float(winr_row[j]), taper)
        out[j, :] = u[j, :] * tmp
    return out


def fj(uf, r, c, f, fstride=1, itype=1, nThread=20):
    """F-J 成像的 Bessel 分支。"""
    del nThread
    r = _as_float32_1d(r, "r")
    c = _as_float32_1d(c, "c")
    f = _as_float32_1d(f, "f")
    uf = np.asarray(uf, dtype=np.float32)
    if uf.ndim != 2:
        raise ValueError("uf must be a 2-D array.")
    uf = np.ascontiguousarray(uf[:, ::fstride][:, : len(f)], dtype=np.float32)
    if uf.shape[0] != len(r) or uf.shape[1] != len(f):
        raise ValueError("uf shape does not match r and f.")
    if itype == 1:
        return _integral_kernel(uf, r, f, c, "j")
    return _trap_kernel(uf, r, f, c, "j")


def fh(uf, r, c, f, fstride=1, itype=1, nThread=20):
    """F-H 成像的 Hankel/Bessel-Y 分支。"""
    del nThread
    r = _as_float32_1d(r, "r")
    c = _as_float32_1d(c, "c")
    f = _as_float32_1d(f, "f")
    uf = np.asarray(uf, dtype=np.float32)
    if uf.ndim != 2:
        raise ValueError("uf must be a 2-D array.")
    uf = np.ascontiguousarray(uf[:, ::fstride][:, : len(f)], dtype=np.float32)
    if uf.shape[0] != len(r) or uf.shape[1] != len(f):
        raise ValueError("uf shape does not match r and f.")
    if itype == 1:
        return _integral_kernel(uf, r, f, c, "y")
    return _trap_kernel(uf, r, f, c, "y")


def fj_noise(uf, r, c, f, fstride=1, itype=1, func=0, num=20):
    """对环境噪声互相关结果执行频散成像。"""
    del num
    indx = np.argsort(r)
    r = np.asarray(r, dtype=np.float32)[indx]
    uf = np.asarray(uf, dtype=np.float32)[indx]
    if func == 0:
        out = fj(uf, r, c, f, fstride=fstride, itype=itype)
    elif func == 1:
        # `func=1` 需要先对每一条互相关曲线做 Hilbert 变换，再组合 F-J / F-H。
        outr = fj(uf, r, c, f, fstride=fstride, itype=itype)
        ufi = _hilbert_rows(uf)
        outi = fh(ufi, r, c, f, fstride=fstride, itype=itype)
        out = outr - outi
    else:
        raise ValueError("func must be 0 or 1.")
    return _normalize_by_freq(out)


def fj_earthquake(u, r, c, f, fstride=1, itype=1, func=0, num=20):
    """直接对原始地震波形执行 F-J / F-H 成像。"""
    del num
    indx = np.argsort(r)
    r = np.asarray(r, dtype=np.float32)[indx]
    u = np.asarray(u, dtype=np.float32)[indx]
    uf = np.fft.rfft(u, axis=1)
    uf = uf[:, 0 : len(f) * fstride : fstride]
    ufr = np.real(uf).astype(np.float32, copy=False)
    ufi = np.imag(uf).astype(np.float32, copy=False)
    if func == 0:
        outr = fj(ufr, r, c, f, fstride=1, itype=itype)
        outi = fj(ufi, r, c, f, fstride=1, itype=itype)
        out = np.sqrt(outr * outr + outi * outi)
    elif func == 1:
        outr = fj(ufr, r, c, f, fstride=1, itype=itype) - fh(ufi, r, c, f, fstride=1, itype=itype)
        outi = fj(ufi, r, c, f, fstride=1, itype=itype) + fh(ufr, r, c, f, fstride=1, itype=itype)
        out = np.sqrt(outr * outr + outi * outi)
    else:
        raise ValueError("func must be 0 or 1.")
    return _normalize_by_freq(out)


def MWFJ(u, r, c, f, Fs, nwin, winl, winr, taper=0.9, fstride=1, itype=1, func=0, num=20):
    """多窗 F-J 入口函数。

    对每个时间窗先施加窗函数，再调用 `fj_earthquake` 逐窗成像。
    """
    del num
    r = np.asarray(r, dtype=np.float32)
    u = np.ascontiguousarray(np.asarray(u, dtype=np.float32))
    winl = np.ascontiguousarray(np.asarray(winl, dtype=np.float32))
    winr = np.ascontiguousarray(np.asarray(winr, dtype=np.float32))
    if u.ndim != 2:
        raise ValueError("u must be a 2-D array.")
    npts = u.shape[1]
    out = np.zeros((nwin, len(c), len(f)), dtype=np.float32)
    for i in range(nwin):
        # 每个窗单独加权，避免修改原始输入波形。
        u0 = _apply_windows_numba(u, float(Fs), winl[i], winr[i], float(taper))
        out[i, :, :] = fj_earthquake(u0, r, c, f, fstride=fstride, itype=itype, func=func)
    return out
