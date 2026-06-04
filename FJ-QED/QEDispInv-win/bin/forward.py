"""Windows 版前向计算命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qedispinv_win.dispersion import DispersionSolver
from qedispinv_win.io_utils import loadtxt, parse_toml, resolve_forward_config
from qedispinv_win.sensitivity import compute_phase_velocity_kernel
from qedispinv_win.storage import save_npz_dict


def compute_kernel_bundle(
    model: np.ndarray,
    disp_out: np.ndarray,
    sh: bool,
) -> dict[str, np.ndarray]:
    """为整条色散曲线计算敏感核矩阵。

    输入:
        model:
            类型: `numpy.ndarray`
            形状: `(nl, 5)`
            单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            含义: 层状模型。
        disp_out:
            类型: `numpy.ndarray`
            形状: `(nd, 3)`
            单位: 第 1 列 `Hz`，第 2 列 `km/s`
            含义: 已计算好的色散曲线。
        sh:
            类型: `bool`
            单位: 无
            含义: 是否为 Love 波。
    输出:
        `dict[str, numpy.ndarray]`
            含义: 与参考项目兼容的 `disp/kvp/kvs/krho/z` 结果字典。
    """
    nd = disp_out.shape[0]
    nl = model.shape[0]
    kvp = np.zeros((nl, nd), dtype=np.float64)
    kvs = np.zeros((nl, nd), dtype=np.float64)
    krho = np.zeros((nl, nd), dtype=np.float64)
    for i, row in enumerate(disp_out):
        kernel = compute_phase_velocity_kernel(model, float(row[0]), float(row[1]), sh)
        kvp[:, i] = kernel.vp
        kvs[:, i] = kernel.vs
        krho[:, i] = kernel.rho
    return {
        "disp": disp_out,
        "kvp": kvp,
        "kvs": kvs,
        "krho": krho,
        "z": model[:, 1].copy(),
    }


def main() -> None:
    """执行前向色散计算并按需输出敏感核。"""
    parser = argparse.ArgumentParser(description="Calculate dispersion curves given a model.")
    parser.add_argument("-c", "--config", default="config.toml", help="toml-type configure file")
    parser.add_argument("-m", "--mode", type=int, default=0, help="maximum mode up to")
    parser.add_argument("--disp", default="", help="input dispersion file to reuse target frequencies")
    parser.add_argument("--sh", action="store_true", help="whether to compute Love waves")
    parser.add_argument("--compute_kernel", action="store_true", help="whether to compute kernel")
    parser.add_argument("--model", default="", help="filename of model")
    parser.add_argument("-o", "--out", default="disp.txt", help="filename of output")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = parse_toml(config_path)
    forward_conf = resolve_forward_config(config)
    file_model = args.model or str((config_path.parent / forward_conf.file_model).resolve())
    model = loadtxt(file_model)
    disp = DispersionSolver(model, args.sh)

    rows = []
    if not args.disp:
        freqs = np.linspace(forward_conf.fmin, forward_conf.fmax, forward_conf.nf)
        for f in freqs:
            cs = disp.search(float(f), args.mode + 1)
            for mode, c in enumerate(cs):
                rows.append([f, c, mode])
    else:
        disp_target = loadtxt(Path(args.disp).resolve())
        if disp_target.ndim == 1:
            disp_target = disp_target.reshape(1, -1)
        for row in disp_target:
            f = float(row[0])
            mode = int(row[2])
            c = disp.search_mode(f, mode)
            if not np.isnan(c):
                rows.append([f, c, mode])
    out = np.asarray(rows, dtype=np.float64)
    np.savetxt(args.out, out, fmt="%15.5f%15.7f%15d")

    if args.compute_kernel:
        bundle = compute_kernel_bundle(model, out, args.sh)
        save_npz_dict(Path(args.out).with_name("kernel.npz"), bundle)


if __name__ == "__main__":
    main()
