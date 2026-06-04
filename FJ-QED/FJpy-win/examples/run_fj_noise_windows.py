from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ccfj  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

REF_EXAMPLES = (
    ROOT.parent / "ref" / "CC-FJpy-master" / "examples"
    if (ROOT.parent / "ref").exists()
    else ROOT.parent / "FJ-QED" / "ref" / "CC-FJpy-master" / "examples"
)


def main() -> None:
    """运行环境噪声互相关结果的 F-J / F-H 成像测试。"""
    src = REF_EXAMPLES / "summed.npz"
    outdir = ROOT / "output"
    outdir.mkdir(exist_ok=True)

    data = np.load(src)
    # 这里使用较小规模输入，兼顾结果可复核性和 Windows 版本执行时间。
    ncfs = data["ncfs"][:16, :128]
    r = data["r"][:16] * 1e3
    f = data["f"][:128]

    c = np.linspace(2000, 5000, 180, dtype=np.float32)
    cases = [
        ("fj_noise_j_trap", dict(fstride=1, itype=0, func=0)),
        ("fj_noise_j_int", dict(fstride=1, itype=1, func=0)),
        ("fj_noise_y_trap", dict(fstride=1, itype=0, func=1)),
        ("fj_noise_y_int", dict(fstride=1, itype=1, func=1)),
    ]

    summary = {}
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 12), constrained_layout=True)
    for idx, (name, kwargs) in enumerate(cases):
        row, col = divmod(idx, 2)
        ds = ccfj.fj_noise(np.real(ncfs), r, c, f, **kwargs)
        summary[name] = {
            "shape": list(ds.shape),
            "max": float(np.max(ds)),
            "min": float(np.min(ds)),
            "mean": float(np.mean(ds)),
        }
        im = ax[row][col].pcolormesh(f, c / 1e3, ds, cmap="jet", vmin=0, vmax=0.8, shading="auto")
        ax[row][col].set_xlim([0, 0.5])
        ax[row][col].set_xlabel("Frequency (Hz)")
        ax[row][col].set_ylabel("Phase velocity (km/s)")
        ax[row][col].set_title(name)
        fig.colorbar(im, ax=ax[row][col], fraction=0.046, pad=0.04)

    fig.savefig(outdir / "fj_noise_windows.png", dpi=160)
    (outdir / "fj_noise_windows_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("噪声 F-J 测试完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
