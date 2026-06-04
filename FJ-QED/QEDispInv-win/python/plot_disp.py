#!/usr/bin/env python
"""绘制色散曲线。"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """读取文本色散文件并按模态绘图。

    输入:
        命令行参数 `file_disp`:
            类型: `str`
            单位: 无
            含义: 色散数据文件路径，列为 `频率 / 相速度 / 模态号`。
    输出:
        无。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_disp", help="dispersion file")
    args = parser.parse_args()
    disp = np.loadtxt(args.file_disp)
    if disp.ndim == 1:
        disp = disp.reshape(1, -1)
    fig, ax = plt.subplots(layout="constrained")
    for mode in sorted({int(m) for m in disp[:, 2]}):
        sub = disp[disp[:, 2].astype(int) == mode]
        ax.plot(sub[:, 0], sub[:, 1], ".", label=f"mode {mode}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (km/s)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
