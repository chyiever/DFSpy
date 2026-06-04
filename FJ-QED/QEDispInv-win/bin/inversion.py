"""Windows 版反演命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qedispinv_win.inversion import DataSet, InversionConfig, InversionRunner
from qedispinv_win.io_utils import loadtxt, parse_toml
from qedispinv_win.storage import save_npz_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the inversion.")
    parser.add_argument("-c", "--config", default="config.toml", help="toml-type configure file")
    parser.add_argument("-d", "--data", required=True, help="filename of dispersion curves")
    parser.add_argument("--model_ref", default="", help="filename of reference model")
    parser.add_argument("--sh", action="store_true", help="whether are Love waves")
    parser.add_argument("-o", "--out", default="inv.npz", help="filename of output")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = parse_toml(config_path)
    conf = config["inversion"]
    file_mref = args.model_ref or str((config_path.parent / conf["model_ref"]).resolve())
    inv_conf = InversionConfig(
        vs2model=str(conf["vs2model"]),
        vs_width=float(conf["vs_width"]),
        lambda_=float(conf["lambda"]),
        reg_type=int(conf["reg_type"]),
        num_init=int(conf["num_init"]),
        num_noise=int(conf["num_noise"]),
        rand_depth=bool(conf["rand_depth"]),
        rand_vs=bool(conf["rand_vs"]),
        zmax=float(conf["zmax"]),
        r0=float(conf["r0"]),
        rmin=float(conf["rmin"]),
        rmax=float(conf["rmax"]),
        weight=[float(x) for x in conf["weight"]],
        maxiter=int(conf["maxiter"]) if "maxiter" in conf else 100,
        vp2vs=float(conf["vp2vs"]) if "vp2vs" in conf else None,
        sigma=[float(x) for x in conf["sigma"]] if "sigma" in conf else None,
    )
    model_ref = loadtxt(file_mref)
    data = DataSet(loadtxt(Path(args.data).resolve()))
    runner = InversionRunner(model_ref, data, inv_conf, sh=args.sh)
    result = runner.run()
    save_npz_dict(args.out, result)


if __name__ == "__main__":
    main()
