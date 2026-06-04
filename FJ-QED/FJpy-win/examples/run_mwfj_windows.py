from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ccfj  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

REF_EXAMPLES = (
    ROOT.parent / "ref" / "CC-FJpy-master" / "examples"
    if (ROOT.parent / "ref").exists()
    else ROOT.parent / "FJ-QED" / "ref" / "CC-FJpy-master" / "examples"
)


def _extract_zip(zip_path: Path) -> Path:
    """解压地震示例压缩包到临时目录。"""
    tmp_root = Path(tempfile.mkdtemp(prefix="fjpy_win_mwfj_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_root)
    return tmp_root


def _load_station_records(data_root: Path, max_station: int = 4):
    """按台站读取 SAC 记录，返回少量台站用于快速复核。"""
    sacs = sorted(data_root.glob("*.SAC"))
    if not sacs:
        raise RuntimeError("未找到 SAC 文件")

    grouped = {}
    for sac in sacs:
        parts = sac.name.split(".")
        if len(parts) < 2:
            continue
        station = ".".join(parts[:2])
        grouped.setdefault(station, []).append(sac)

    stations = sorted(grouped)[:max_station]
    traces = []
    names = []
    for sta in stations:
        files = sorted(grouped[sta])
        tr = read(str(files[0]))[0]
        tr.detrend("demean")
        traces.append(tr)
        names.append(sta)
    return names, traces


def main() -> None:
    """运行多窗 F-J 示例，并输出图像与统计摘要。"""
    src = REF_EXAMPLES / "eqdata.zip"
    outdir = ROOT / "output"
    outdir.mkdir(exist_ok=True)

    tmp_root = _extract_zip(src)
    try:
        names, traces = _load_station_records(tmp_root, max_station=4)
        nsta = len(traces)
        npts = min(tr.stats.npts for tr in traces)
        Fs = float(traces[0].stats.sampling_rate)
        u0 = np.zeros((nsta, npts), dtype=np.float32)
        r = np.zeros(nsta, dtype=np.float32)
        for i, tr in enumerate(traces):
            u0[i, :] = tr.data[:npts].astype(np.float32, copy=False)
            if hasattr(tr.stats, "sac") and "dist" in tr.stats.sac:
                r[i] = float(tr.stats.sac["dist"])
            else:
                r[i] = float(i + 1) * 10_000.0

        nwin = 3
        nf = min(64, npts // 2 + 1)
        nc = 80
        minc = 2000.0
        maxc = 6000.0
        f = Fs * np.linspace(0, nf - 1, nf, dtype=np.float32) / npts
        c = np.linspace(minc, maxc, nc, dtype=np.float32)
        EQT = 60.0
        V1 = [3.2, 3.7]
        V2 = [3.7, 4.2]
        winl = np.zeros((nwin, nsta), dtype=np.float32)
        winr = np.zeros((nwin, nsta), dtype=np.float32)
        for i in range(nsta):
            winl[0, i] = 0
            winr[0, i] = npts - 1
            winl[1, i] = r[i] / V2[0] + EQT - 5
            winr[1, i] = r[i] / V1[0] + EQT + 5
            winl[2, i] = r[i] / V2[1] + EQT - 5
            winr[2, i] = r[i] / V1[1] + EQT + 5

        indx = np.argsort(r)
        u0 = u0[indx, :]
        # 相速度 c 的单位是 m/s，因此 r 也统一转换为米。
        r = r[indx] * 1e3
        winl = winl[:, indx]
        winr = winr[:, indx]
        names = [names[i] for i in indx]

        out = ccfj.MWFJ(u0, r, c, f, Fs, nwin, winl, winr, taper=0.9, fstride=1, itype=1, func=0, num=4)
        out1 = ccfj.MWFJ(u0, r, c, f, Fs, nwin, winl, winr, taper=0.9, fstride=1, itype=1, func=1, num=4)

        summary = {
            "stations": names,
            "shape_bessel": list(out.shape),
            "shape_hankel": list(out1.shape),
            "bessel_max": float(np.max(out)),
            "bessel_min": float(np.min(out)),
            "bessel_mean": float(np.mean(out)),
            "bessel_std": float(np.std(out)),
            "bessel_window_mean": [float(np.mean(out[i])) for i in range(out.shape[0])],
            "hankel_max": float(np.max(out1)),
            "hankel_min": float(np.min(out1)),
            "hankel_mean": float(np.mean(out1)),
            "hankel_std": float(np.std(out1)),
            "hankel_window_mean": [float(np.mean(out1[i])) for i in range(out1.shape[0])],
        }
        np.savez(outdir / "mwfj_windows_result.npz", out=out, out1=out1, f=f, c=c, stations=names)

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), constrained_layout=True)
        for i in range(3):
            ax[0][i].imshow(
                np.flip(out[i, :, :], 0),
                extent=[float(np.min(f)), float(np.max(f)), float(np.min(c / 1e3)), float(np.max(c / 1e3))],
                aspect="auto",
                vmax=0.8,
                cmap="jet",
            )
            ax[0][i].set_title(f"Window {i + 1} - Bessel")
            ax[0][i].set_xlabel("Frequency (Hz)")
            ax[0][i].set_ylabel("Phase velocity (km/s)")
            ax[1][i].imshow(
                np.flip(out1[i, :, :], 0),
                extent=[float(np.min(f)), float(np.max(f)), float(np.min(c / 1e3)), float(np.max(c / 1e3))],
                aspect="auto",
                vmax=0.8,
                cmap="jet",
            )
            ax[1][i].set_title(f"Window {i + 1} - Hankel")
            ax[1][i].set_xlabel("Frequency (Hz)")
            ax[1][i].set_ylabel("Phase velocity (km/s)")
        fig.savefig(outdir / "mwfj_windows.png", dpi=160)

        (outdir / "mwfj_windows_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Multi-windows F-J test finished.")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
