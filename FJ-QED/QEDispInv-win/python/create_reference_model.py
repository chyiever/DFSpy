#!/usr/bin/env python
"""根据基阶色散曲线生成参考模型。"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, make_smoothing_spline


def create_model_brocher(dep: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """按 Brocher 经验关系生成参考模型。

    输入:
        dep:
            类型: `numpy.ndarray`
            形状: `(n,)`
            单位: `km`
            含义: 深度节点。
        vs:
            类型: `numpy.ndarray`
            形状: `(n,)`
            单位: `km/s`
            含义: 对应深度的 `Vs`。
    输出:
        `numpy.ndarray`
            类型: `float64`
            形状: `(n, 5)`
            单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            含义: 完整层状模型。
    """
    model = np.zeros((vs.size, 5), dtype=np.float64)
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    rho = 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
    model[:, 0] = np.arange(vs.size) + 1.0
    model[:, 1] = dep
    model[:, 2] = rho
    model[:, 3] = vs
    model[:, 4] = vp
    return model


def create_model_nearsurface(dep: np.ndarray, vs: np.ndarray, vp2vs: float) -> np.ndarray:
    """按浅层经验关系生成参考模型。"""
    a = -0.22374079
    b = 1.32248261
    c = 1.54840433
    rho = a * vs**2 + b * vs + c
    vp = vp2vs * vs
    model = np.zeros((vs.size, 5), dtype=np.float64)
    model[:, 0] = np.arange(vs.size) + 1.0
    model[:, 1] = dep
    model[:, 2] = rho
    model[:, 3] = vs
    model[:, 4] = vp
    return model


def create_model_with_dmodel(dep: np.ndarray, vs: np.ndarray, dmodel: np.ndarray) -> np.ndarray:
    """用目标模型的 `Vp` 与密度约束生成参考模型。"""
    i1 = 0
    lines: list[list[float]] = []
    intp = interp1d(dep, vs)
    for i in range(1, dmodel.shape[0]):
        i2 = np.searchsorted(dep, dmodel[i, 1])
        for j in range(i1, i2):
            lines.append([0.0, dep[j], dmodel[i - 1, 2], vs[j], dmodel[i - 1, 4]])
        i1 = i2
        z1 = dmodel[i, 1] - 1.0e-7
        z2 = dmodel[i, 1]
        lines.append([0.0, z1, dmodel[i - 1, 2], float(intp(z1)), dmodel[i - 1, 4]])
        lines.append([0.0, z2, dmodel[i, 2], float(intp(z2)), dmodel[i, 4]])
    for j in range(i1, dep.shape[0]):
        lines.append([0.0, dep[j], dmodel[-1, 2], vs[j], dmodel[-1, 4]])
    model = np.asarray(lines, dtype=np.float64)
    model[:, 0] = np.arange(model.shape[0]) + 1.0
    return model


def main() -> None:
    """根据基阶色散曲线估计参考模型并可视化。"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_disp", help="filename of dispersion curves")
    parser.add_argument("--vp2vs", type=float, default=1.732, help="ratio of vp and vs")
    parser.add_argument("-s", "--smooth", type=float, help="smoothing parameter")
    parser.add_argument("--dmodel", help="filename of data model")
    parser.add_argument("--zmax", type=float)
    parser.add_argument("-o", "--out", default="mref.txt", help="filename of output")
    parser.add_argument("--savefig", help="name of the output figure file")
    args = parser.parse_args()

    disp = np.loadtxt(args.file_disp)
    if disp.ndim == 1:
        disp = disp.reshape(1, -1)
    disp = disp[disp[:, 2].astype(int) == 0]
    f = disp[:, 0]
    c = disp[:, 1]
    dep_data = np.sort(c / f / 3.0)
    vs_data = c[np.argsort(c / f / 3.0)] * 1.1
    norm_d = dep_data.max()
    norm_v = vs_data.max()
    dn = dep_data / norm_d
    vn = vs_data / norm_v
    spl = make_smoothing_spline(dn, vn, lam=args.smooth) if args.smooth is not None else interp1d(dn, vn, kind="linear")
    dep_mid = np.linspace(dep_data[0], dep_data[-1], 50)
    vs_mid = np.maximum(spl(dep_mid / norm_d) * norm_v, 0.05)

    if dep_mid[0] > 0:
        z1 = dep_mid[0]
        v1 = vs_mid[0]
        k1 = max((vs_mid[1] - vs_mid[0]) / (dep_mid[1] - dep_mid[0]), 0.0)
        drop = min(0.5 * k1 * z1, 0.05 * v1)
        dep_head = np.linspace(0.0, z1, 15)[:-1]
        vs_head = v1 - drop + drop * (dep_head / z1) ** 2
    else:
        dep_head = np.array([], dtype=np.float64)
        vs_head = np.array([], dtype=np.float64)

    target_zmax = args.zmax if args.zmax else dep_mid[-1] * 1.5
    if target_zmax > dep_mid[-1]:
        z_n = dep_mid[-1]
        v_n = vs_mid[-1]
        k_n = max((vs_mid[-1] - vs_mid[-2]) / (dep_mid[-1] - dep_mid[-2]), 0.0)
        dz_total = target_zmax - z_n
        tail_dv = min(0.5 * k_n * dz_total, 0.1 * v_n)
        dep_tail = np.linspace(z_n, target_zmax, 25)[1:]
        x_tail = (dep_tail - z_n) / dz_total
        vs_tail = v_n + tail_dv * (2 * x_tail - x_tail**2)
    else:
        dep_tail = np.array([], dtype=np.float64)
        vs_tail = np.array([], dtype=np.float64)

    dep = np.concatenate((dep_head, dep_mid, dep_tail))
    vs = np.concatenate((vs_head, vs_mid, vs_tail))
    if args.dmodel:
        model = create_model_with_dmodel(dep, vs, np.loadtxt(args.dmodel))
    elif dep[-1] > 1.0:
        model = create_model_brocher(dep, vs)
    else:
        model = create_model_nearsurface(dep, vs, args.vp2vs)
    np.savetxt(args.out, model, fmt="%5.0f%15.8f%12.5f%12.5f%12.5f")

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(vs_data, dep_data, "k.", alpha=0.8)
    ax.plot(vs, dep, "r-", alpha=0.8, linewidth=2)
    ax.set_ylim([dep[0], args.zmax if args.zmax else dep[-1]])
    ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Reference Model")
    if args.savefig:
        fig.savefig(args.savefig, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
