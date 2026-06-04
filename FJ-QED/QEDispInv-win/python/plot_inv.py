#!/usr/bin/env python
"""绘制反演结果。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qedispinv_win.storage import load_npz_dict


def plot_1disp(ax, disp: np.ndarray, mode_show: list[int], marker: str, linestyle: str, color: str, alpha: float):
    """绘制一组色散曲线。

    输入:
        ax:
            类型: `matplotlib.axes.Axes`
            单位: 无
            含义: 目标坐标轴。
        disp:
            类型: `numpy.ndarray`
            形状: `(n, 3)`
            单位: 第 1 列 `Hz`，第 2 列 `km/s`
            含义: 色散数据。
        mode_show:
            类型: `list[int]`
            单位: 无
            含义: 需要绘制的模态号列表。
        marker, linestyle, color:
            类型: `str`
            单位: 无
            含义: 绘图样式参数。
        alpha:
            类型: `float`
            单位: 无
            含义: 透明度。
    输出:
        `matplotlib.lines.Line2D | None`
            含义: 最后一次绘制得到的线对象。
    """
    modes = disp[:, 2].astype(int)
    handle = None
    for mode in mode_show:
        sub = disp[modes == mode]
        if sub.size == 0:
            continue
        (handle,) = ax.plot(sub[:, 0], sub[:, 1], marker=marker, linestyle=linestyle, color=color, alpha=alpha)
    return handle


def plot_model(data: dict[str, np.ndarray], file_model_data: str | None, xlim: list[float] | None) -> None:
    """绘制反演得到的速度统计剖面。"""
    vs_hist = np.ma.masked_array(data["vs_hist2d"], mask=data["vs_hist2d"] <= 0)
    z_sample = data["z_sample"]
    vs_sample = data["vs_sample"]
    dx = vs_sample[1] - vs_sample[0]
    dy = z_sample[1] - z_sample[0]
    x = vs_sample - dx / 2.0
    y = z_sample - dy / 2.0
    fig, ax = plt.subplots(layout="constrained")
    ax.pcolormesh(x, y, vs_hist, cmap="Wistia", alpha=0.8)
    ax.plot(data["vs_ref"], z_sample, "k-", alpha=0.6, lw=2, label="Reference")
    ax.plot(data["vs_median"], z_sample, "-", c="blue", alpha=0.8, lw=2, label="Median")
    ax.plot(data["vs_cred10"], z_sample, "--", c="blue", lw=1, alpha=0.8, label="P10/P90")
    ax.plot(data["vs_cred90"], z_sample, "k--", lw=1, alpha=0.6)
    if file_model_data:
        model_data = np.loadtxt(file_model_data)
        vs = model_data[:, 3]
        z = model_data[:, 1]
        if z[-1] < z_sample[-1]:
            z = np.append(z, z_sample[-1])
            vs = np.append(vs, vs[-1])
        ax.step(vs, z, "-", c="r", alpha=0.7, lw=2, label="Target")
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim([0, z_sample[-1]])
    ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.legend()
    ax.grid(linestyle=":")


def plot_disp(data_obs: np.ndarray, disp_syn: list[np.ndarray], fitness: np.ndarray, mode_used: np.ndarray, show_full_disp: bool, xlim: list[float] | None) -> None:
    """绘制观测与反演色散曲线拟合。"""
    mode_show = sorted({int(x) for x in data_obs[:, 2]}) if show_full_disp else [int(x) for x in mode_used]
    val = 1.0 / fitness
    vmin = np.min(val)
    vmax = np.max(val)
    alpha = (val - vmin) / (vmax - vmin + 1.0e-12) * 0.8
    ind = np.argsort(alpha)
    alpha = alpha[ind]
    disp_syn = [disp_syn[i] for i in ind]
    fig, ax = plt.subplots(layout="constrained")
    for i, disp in enumerate(disp_syn):
        plot_1disp(ax, disp, mode_show, "", "-", "k", float(alpha[i]))
    p1 = ax.plot([], [], "k-", alpha=0.8)[0]
    p2 = plot_1disp(ax, data_obs, mode_show, ".", "", "r", 0.8)
    ax.legend([p1, p2], ["inv", "data"], loc="upper right")
    f_show = np.hstack([data_obs[data_obs[:, 2].astype(int) == mode][:, 0] for mode in mode_show])
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([np.min(f_show), np.max(f_show)])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (km/s)")


def plot_fitness(fitness: np.ndarray) -> None:
    """绘制目标函数值排序图。"""
    idx = np.argsort(fitness)[::-1]
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(fitness[idx], "k.-", alpha=0.8)
    ax.set_ylabel("Fitness (km^2/s^2)")
    ax.set_xlabel("Index of inversion")


def main() -> None:
    """读取反演结果并按参数选择绘图。"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_inv", help="filename of inversion results npz")
    parser.add_argument("--plot_model", action="store_true", help="display the subsurface model")
    parser.add_argument("-d", "--model_data", help="target model")
    parser.add_argument("--plot_disp", action="store_true", help="overlay observed and predicted dispersion")
    parser.add_argument("--plot_fit", action="store_true", help="plot sorted fitness")
    parser.add_argument("--full_disp", action="store_true", help="show all modes")
    parser.add_argument("--xlim", nargs=2, type=float, help="x limits")
    parser.add_argument("--savefig", help="figure file")
    args = parser.parse_args()
    data = load_npz_dict(args.file_inv)
    disp_syn = [item for item in data["disp_syn_list"]]
    if args.plot_model:
        plot_model(data, args.model_data, args.xlim)
    if args.plot_disp:
        plot_disp(data["data"], disp_syn, data["fitness"], data["mode_used"], args.full_disp, args.xlim)
    if args.plot_fit:
        plot_fitness(data["fitness"])
    if args.savefig:
        plt.savefig(args.savefig, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
