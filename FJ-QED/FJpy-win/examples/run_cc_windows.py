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
    """解压示例压缩包到临时目录，避免污染工作区。"""
    tmp_root = Path(tempfile.mkdtemp(prefix="fjpy_win_cc_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_root)
    return tmp_root


def _common_day(station_dirs: list[Path]) -> str:
    """在多个台站目录中寻找共同日期。"""
    day_sets = []
    for sta_dir in station_dirs:
        days = set()
        for sac in sta_dir.glob("*.SAC"):
            parts = sac.name.split("-")
            if len(parts) >= 4:
                days.add(parts[3])
        if days:
            day_sets.append(days)
    common = set.intersection(*day_sets) if day_sets else set()
    if not common:
        raise RuntimeError("台站之间没有共同日期。")
    return sorted(common)[0]


def main() -> None:
    """运行三台站互相关示例，并导出结果图与统计摘要。"""
    src = REF_EXAMPLES / "ThreeStationsCCExample.zip"
    outdir = ROOT / "output"
    outdir.mkdir(exist_ok=True)

    tmp_root = _extract_zip(src)
    try:
        data_root = tmp_root / "DataDemo"
        station_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
        if len(station_dirs) < 2:
            raise RuntimeError("三台站示例数据不足。")

        day = _common_day(station_dirs[:3])
        traces = []
        stations = []
        for sta_dir in station_dirs[:3]:
            files = sorted(sta_dir.glob(f"*{day}*BHZ.SAC"))
            if not files:
                continue
            tr = read(str(files[0]))[0]
            tr.detrend("demean")
            traces.append(tr)
            stations.append(sta_dir.name)

        if len(traces) < 2:
            raise RuntimeError("可用台站少于 2 个，无法计算互相关。")

        nsta = len(traces)
        npts = min(tr.stats.npts for tr in traces)
        Fs = float(traces[0].stats.sampling_rate)
        fftlen = min(int(Fs * 3600), npts)
        nf = min(512, fftlen // 2 + 1)
        fstride = 1

        # data 按台站依次平铺，布局与参考项目保持一致。
        data = np.zeros(nsta * npts, dtype=np.float32)
        startend = np.zeros(nsta * 2, dtype=np.int32)
        for i, tr in enumerate(traces):
            data[i * npts : (i + 1) * npts] = tr.data[:npts].astype(np.float32, copy=False)
            startend[i * 2] = 0
            startend[i * 2 + 1] = npts

        pairs = ccfj.GetStationPairs(nsta)
        ncfs = ccfj.CC(
            npts,
            nsta,
            nf,
            fftlen,
            pairs,
            startend,
            data,
            overlaprate=0.5,
            nThreads=4,
            fstride=fstride,
            ifonebit=0,
            ifspecwhittenning=1,
        )

        f = np.arange(nf, dtype=np.float32) * Fs / fftlen * fstride
        summary = {
            "stations": stations,
            "day": day,
            "shape": list(ncfs.shape),
            "abs_max": float(np.max(np.abs(ncfs))),
            "abs_mean": float(np.mean(np.abs(ncfs))),
        }
        np.savez(outdir / "cc_windows_result.npz", ncfs=ncfs, pairs=pairs, f=f, stations=stations)

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        im = ax.pcolormesh(f, np.arange(ncfs.shape[0]), np.real(ncfs), cmap="jet", shading="auto")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Station-pair index")
        ax.set_title("Cross-correlation functions in frequency domain")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(outdir / "cc_windows.png", dpi=160)

        (outdir / "cc_windows_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("互相关函数测试完成。")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
