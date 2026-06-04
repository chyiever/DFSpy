#!/usr/bin/env python
"""导出反演模型。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qedispinv_win.storage import load_npz_dict


def linear2step(model_mean: np.ndarray) -> np.ndarray:
    """将线性节点模型转换为阶梯层状模型。

    输入:
        model_mean:
            类型: `numpy.ndarray`
            形状: `(nl, 5)`
            单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            含义: 线性节点模型。
    输出:
        `numpy.ndarray`
            类型: `float64`
            形状: `(nl, 5)`
            单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            含义: 转换后的阶梯层状模型。
    """
    z = model_mean[:, 1]
    model2 = np.zeros_like(model_mean)
    model2[0, :] = model_mean[0, :]
    for j in range(model_mean.shape[0] - 1):
        model2[j + 1, 1] = (z[j] + z[j + 1]) / 2.0
        model2[j + 1, 2:] = model_mean[j + 1, 2:]
    model2[:, 0] = np.arange(model_mean.shape[0]) + 1.0
    return model2


def main() -> None:
    """读取反演结果并输出线性或阶梯层状模型。

    输入:
        命令行参数 `file_inv`:
            类型: `str`
            单位: 无
            含义: 反演结果 `.npz` 文件路径。
        命令行参数 `--step`:
            类型: `bool`
            单位: 无
            含义: 是否输出阶梯模型。
        命令行参数 `-o/--out`:
            类型: `str | None`
            单位: 无
            含义: 可选输出文件路径。
    输出:
        无。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_inv", help="filename of npz file from inversion result")
    parser.add_argument("--step", action="store_true", help="to generate stepwise model")
    parser.add_argument("-o", "--out")
    args = parser.parse_args()
    data = load_npz_dict(args.file_inv)
    model_mean = data["model_mean"]
    if args.step:
        model_mean = linear2step(model_mean)
    for row in model_mean:
        print("{:5.0f}{:12.5f}{:12.5f}{:12.5f}{:12.5f}".format(*row))
    if args.out:
        np.savetxt(args.out, model_mean, fmt="%5.0f%12.5f%12.5f%12.5f%12.5f")


if __name__ == "__main__":
    main()
