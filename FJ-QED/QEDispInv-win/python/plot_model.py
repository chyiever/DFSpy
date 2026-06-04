#!/usr/bin/env python
"""绘制层状模型。"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """读取层状模型文件并绘制 `Vs-Depth` 剖面。

    输入:
        命令行参数 `file_model`:
            类型: `str`
            单位: 无
            含义: 模型文件路径，至少包含深度列与 `Vs` 列。
        命令行参数 `--linear`:
            类型: `bool`
            单位: 无
            含义: 是否使用线性连接方式绘制。
    输出:
        无。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_model", help="model file")
    parser.add_argument("--linear", action="store_true", help="plot as linear segments")
    args = parser.parse_args()
    model = np.loadtxt(args.file_model)
    z = model[:, 1]
    vs = model[:, 3]
    fig, ax = plt.subplots(layout="constrained")
    if args.linear:
        ax.plot(vs, z, "k-", lw=2)
    else:
        ax.step(vs, z, "k-", where="post", lw=2)
    ax.invert_yaxis()
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    plt.show()


if __name__ == "__main__":
    main()
