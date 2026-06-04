#!/usr/bin/env python
"""绘制核函数。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qedispinv_win.storage import load_npz_dict


def main() -> None:
    """读取核函数结果并绘制某一模态的深度-频率图。

    输入:
        命令行参数 `file_ker`:
            类型: `str`
            单位: 无
            含义: `kernel.npz` 文件路径。
        命令行参数 `--comp`:
            类型: `str`
            单位: 无
            含义: 组件选择，可为 `vp`、`vs`、`rho`。
        命令行参数 `-m/--mode`:
            类型: `int`
            单位: 无
            含义: 要显示的模态号。
    输出:
        无。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("file_ker", help="kernel npz file")
    parser.add_argument("--comp", default="vs", help="component of kernel (vp, vs, rho)")
    parser.add_argument("-m", "--mode", type=int, default=0, help="mode index")
    args = parser.parse_args()
    data = load_npz_dict(args.file_ker)
    comp = {"vs": "kvs", "vp": "kvp", "rho": "krho"}[args.comp]
    disp = data["disp"]
    modes = disp[:, 2].astype(int)
    idx = modes == args.mode
    z = data["z"]
    kernel = data[comp][:, idx]
    fig, ax = plt.subplots(layout="constrained")
    ax.pcolormesh(disp[idx, 0], z, kernel, shading="auto", cmap="seismic")
    ax.invert_yaxis()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Depth (km)")
    plt.show()


if __name__ == "__main__":
    main()
