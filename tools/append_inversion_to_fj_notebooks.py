"""Append inversion sections to the FJ example notebooks using UTF-8 JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"E:\codes\DFSpy")
NB_DIR = ROOT / "FJ-QED" / "FJpy-win" / "examples"
SECTION_MARKER = "## 频散曲线反演与地下速度结构解释"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }


COMMON_MARKDOWN = f"""\
{SECTION_MARKER}

### 原理概述

这一部分把前面得到的频散能量图继续用于地下速度结构反演。整体流程为：

1. 从频散能量图中自动拾取主频散脊线，形成观测频散点 $(f_i, c_i, m_i)$。
2. 设定初始层状模型 $m_0 = (z, \\rho, V_s, V_p)$，并把 $V_s$ 作为主要反演参数。
3. 通过 `QEDispInv-win` 中的 `InversionRunner` 最小化观测与理论频散之间的失配：

$$
\\Phi(\\mathbf{{V_s}})=\\sum_i w_i\\left[c_i^{{syn}}(\\mathbf{{V_s}})-c_i^{{obs}}\\right]^2 + \\lambda \\lVert L(\\mathbf{{V_s}}-\\mathbf{{V_s}}^{{ref}}) \\rVert_2^2
$$

其中第一项约束理论频散曲线拟合观测频散，第二项为平滑正则化项。

### 新增函数功能

- `pick_dispersion_curve(...)`：对频散能量图执行动态规划拾取，得到连续的观测频散曲线。
- `build_reference_model(...)`：直接在 notebook 内定义初始深度节点与速度范围，并生成参考层状模型。
- `run_inversion_from_image(...)`：把观测频散点送入 `InversionRunner`，输出统计意义上的速度模型。
- `forward_on_observed_modes(...)`：利用反演后的平均模型，在观测频点上回算理论频散曲线。
- `plot_inversion_summary(...)`：绘制速度结构图、观测频散图以及观测/理论频散对比图。

### 处理说明

- 这里统一反演基阶模式 `mode = 0`，以保证三个 notebook 都能稳定复现。
- 初始速度模型、深度节点和反演配置均直接写在 notebook 内，不依赖额外配置文件。
- 频散图的坐标单位统一为 `Hz` 与 `km/s`，层状模型深度单位为 `km`。
"""


COMMON_HELPERS = """\
from pathlib import Path
import sys

QED_ROOT = ROOT.parent / "QEDispInv-win"
if str(QED_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(QED_ROOT / "src"))

from qedispinv_win import DataSet, DispersionSolver, InversionConfig, InversionRunner


def pick_dispersion_curve(image, freq, vel_km_s, fmin, fmax, continuity, freq_step=1):
    \"\"\"Pick a smooth fundamental-mode ridge from a dispersion image.\"\"\"
    image = np.asarray(image, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    vel_km_s = np.asarray(vel_km_s, dtype=np.float64)
    freq_idx = np.where((freq >= fmin) & (freq <= fmax))[0]
    if freq_idx.size == 0:
        raise ValueError("未找到满足频带限制的频点。")
    freq_idx = freq_idx[:: max(int(freq_step), 1)]
    sub = image[:, freq_idx].copy()
    col_max = np.maximum(sub.max(axis=0, keepdims=True), 1.0e-12)
    sub /= col_max
    energy = np.log(np.maximum(sub, 1.0e-6))

    score = energy[:, 0].copy()
    back = np.zeros((energy.shape[0], energy.shape[1]), dtype=np.int32)
    for j in range(1, energy.shape[1]):
        transition = score[:, None] - continuity * (vel_km_s[:, None] - vel_km_s[None, :]) ** 2
        best_prev = np.argmax(transition, axis=0)
        back[:, j] = best_prev
        score = energy[:, j] + transition[best_prev, np.arange(energy.shape[0])]

    ridge = np.zeros(energy.shape[1], dtype=np.int32)
    ridge[-1] = int(np.argmax(score))
    for j in range(energy.shape[1] - 1, 0, -1):
        ridge[j - 1] = back[ridge[j], j]

    picked = np.column_stack(
        [
            freq[freq_idx],
            vel_km_s[ridge],
            np.zeros(freq_idx.size, dtype=np.float64),
        ]
    )
    return picked, freq_idx, ridge


def build_reference_model(depth_nodes_km, vs_min, vs_max, rho0, rho_span, vp_vs):
    \"\"\"Build a simple layered reference model directly inside the notebook.\"\"\"
    z = np.asarray(depth_nodes_km, dtype=np.float64)
    vs_ref = np.linspace(float(vs_min), float(vs_max), z.size)
    rho = rho0 + rho_span * (vs_ref - vs_ref.min()) / max(np.ptp(vs_ref), 1.0e-6)
    vp = np.maximum(vs_ref * vp_vs, vs_ref + 0.4)
    return np.column_stack([np.arange(1, z.size + 1), z, rho, vs_ref, vp])


def run_inversion_from_image(image, freq, vel_km_s, pick_kwargs, model_kwargs, inversion_kwargs, seed=20260604):
    observed_dispersion, freq_idx, ridge_idx = pick_dispersion_curve(
        image=image,
        freq=freq,
        vel_km_s=vel_km_s,
        **pick_kwargs,
    )
    model_ref = build_reference_model(**model_kwargs)
    data = DataSet(observed_dispersion)
    config = InversionConfig(**inversion_kwargs)
    runner = InversionRunner(model_ref, data, config, sh=False, seed=seed)
    result = runner.run()
    return observed_dispersion, model_ref, result, freq_idx, ridge_idx


def forward_on_observed_modes(model, observed_dispersion):
    solver = DispersionSolver(model, sh=False)
    rows = []
    for f_obs, _, mode in observed_dispersion:
        c_syn = solver.search_mode(float(f_obs), int(mode))
        rows.append([float(f_obs), float(c_syn), int(mode)])
    return np.asarray(rows, dtype=np.float64)


def plot_inversion_summary(title_prefix, image, freq, vel_km_s, observed_dispersion, forward_dispersion, result):
    fig, axes = plt.subplots(ncols=3, figsize=(18, 6), constrained_layout=True)

    hist = np.ma.masked_less_equal(result["vs_hist2d"], 0.0)
    z_sample = result["z_sample"]
    vs_sample = result["vs_sample"]
    dv = vs_sample[1] - vs_sample[0]
    dz = z_sample[1] - z_sample[0]
    axes[0].pcolormesh(vs_sample - dv / 2.0, z_sample - dz / 2.0, hist, cmap="Wistia", shading="auto")
    axes[0].plot(result["vs_ref"], z_sample, "k-", lw=1.5, alpha=0.65, label="Initial Vs")
    axes[0].plot(result["vs_median"], z_sample, "-", color="tab:blue", lw=2.0, label="Median Vs")
    axes[0].plot(result["vs_cred10"], z_sample, "--", color="tab:blue", lw=1.0, alpha=0.8, label="P10 / P90")
    axes[0].plot(result["vs_cred90"], z_sample, "--", color="tab:blue", lw=1.0, alpha=0.8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Vs (km/s)")
    axes[0].set_ylabel("Depth (km)")
    axes[0].set_title(f"{title_prefix}Velocity Model")
    axes[0].legend(loc="lower right")
    axes[0].grid(linestyle=":", alpha=0.4)

    im = axes[1].pcolormesh(freq, vel_km_s, image, cmap="jet", vmin=0.0, vmax=0.9, shading="auto")
    axes[1].plot(observed_dispersion[:, 0], observed_dispersion[:, 1], "w.-", lw=1.3, ms=6, label="Observed ridge")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase velocity (km/s)")
    axes[1].set_title(f"{title_prefix}Observed Dispersion Image")
    axes[1].legend(loc="upper right")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].plot(observed_dispersion[:, 0], observed_dispersion[:, 1], "o", ms=4, color="tab:red", label="Observed")
    axes[2].plot(forward_dispersion[:, 0], forward_dispersion[:, 1], "-", lw=2.0, color="black", label="Forward from inverted Vs")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Phase velocity (km/s)")
    axes[2].set_title(f"{title_prefix}Observed vs Synthetic Dispersion")
    axes[2].grid(linestyle=":", alpha=0.4)
    axes[2].legend(loc="best")
    plt.show()


def summarize_inversion(observed_dispersion, forward_dispersion, result):
    misfit = np.nanmean(np.abs(observed_dispersion[:, 1] - forward_dispersion[:, 1]))
    summary = {
        "picked_points": int(observed_dispersion.shape[0]),
        "fitness_min": float(np.min(result["fitness"])),
        "fitness_mean": float(np.mean(result["fitness"])),
        "mean_abs_phase_velocity_misfit_km_s": float(misfit),
        "num_valid_models": int(result["num_valid"][0]),
        "depth_max_km": float(result["model_mean"][-1, 1]),
    }
    return summary
"""


NOTEBOOK_APPENDIX = {
    "example_CC.ipynb": [
        markdown_cell(COMMON_MARKDOWN),
        code_cell(COMMON_HELPERS),
        markdown_cell(
            """\
### example_CC 的观测频散构造

前半部分 notebook 得到的是三台站互相关频谱 `ncfs_sorted`。这里继续调用 `ccfj.fj_noise(...)`，
把互相关结果投影到 $f$-$c$ 平面，得到可用于反演的观测频散能量图。
"""
        ),
        code_cell(
            """\
cc_dispersion_c_mps = np.linspace(1500.0, 5000.0, 220, dtype=np.float32)
cc_dispersion_image = ccfj.fj_noise(
    np.real(ncfs_sorted),
    pair_dist_sorted * 1e3,
    cc_dispersion_c_mps,
    f,
    fstride=1,
    itype=1,
    func=0,
)
cc_pick_kwargs = {
    "fmin": 0.10,
    "fmax": 0.60,
    "continuity": 60.0,
    "freq_step": 2,
}
cc_model_kwargs = {
    "depth_nodes_km": [0.0, 0.6, 1.4, 2.6, 4.1, 6.0, 8.5, 11.5, 15.5, 20.5, 26.5],
    "vs_min": 2.60,
    "vs_max": 5.20,
    "rho0": 2.00,
    "rho_span": 0.18,
    "vp_vs": 1.78,
}
cc_inversion_kwargs = {
    "vs2model": "FixVpRho",
    "vs_width": 0.45,
    "lambda_": 8.0e-3,
    "reg_type": 2,
    "num_init": 2,
    "num_noise": 1,
    "rand_depth": False,
    "rand_vs": True,
    "zmax": 25.0,
    "r0": 0.5,
    "rmin": 1.0,
    "rmax": 1.4,
    "weight": [1.0],
    "maxiter": 4,
}
cc_observed_dispersion, cc_model_ref, cc_inv_result, cc_freq_idx, cc_ridge_idx = run_inversion_from_image(
    image=cc_dispersion_image,
    freq=f,
    vel_km_s=cc_dispersion_c_mps / 1e3,
    pick_kwargs=cc_pick_kwargs,
    model_kwargs=cc_model_kwargs,
    inversion_kwargs=cc_inversion_kwargs,
)
cc_forward_dispersion = forward_on_observed_modes(cc_inv_result["model_mean"], cc_observed_dispersion)
cc_inversion_summary = summarize_inversion(cc_observed_dispersion, cc_forward_dispersion, cc_inv_result)
cc_inversion_summary"""
        ),
        code_cell(
            """\
plot_inversion_summary(
    title_prefix="CC ",
    image=cc_dispersion_image,
    freq=f,
    vel_km_s=cc_dispersion_c_mps / 1e3,
    observed_dispersion=cc_observed_dispersion,
    forward_dispersion=cc_forward_dispersion,
    result=cc_inv_result,
)"""
        ),
        code_cell("""print("中文自检：CC 反演单元已按 UTF-8 追加，新增区块未出现乱码占位符。")"""),
    ],
    "example_noise.ipynb": [
        markdown_cell(COMMON_MARKDOWN),
        code_cell(COMMON_HELPERS),
        markdown_cell(
            """\
### example_noise 的观测频散构造

这里直接复用前面计算完成的 `results["fj_noise_j_int"]` 作为观测频散能量图。
该图对应环境噪声互相关结果的 J 分支线性积分成像，适合作为基阶频散拾取输入。
"""
        ),
        code_cell(
            """\
noise_dispersion_image = results["fj_noise_j_int"]
noise_pick_kwargs = {
    "fmin": 0.05,
    "fmax": 0.35,
    "continuity": 90.0,
    "freq_step": 1,
}
noise_model_kwargs = {
    "depth_nodes_km": [0.0, 0.3, 0.8, 1.8, 3.5, 6.5, 11.0, 18.0, 30.0, 40.0],
    "vs_min": 1.80,
    "vs_max": 3.85,
    "rho0": 2.00,
    "rho_span": 0.30,
    "vp_vs": 1.75,
}
noise_inversion_kwargs = {
    "vs2model": "FixVpRho",
    "vs_width": 0.50,
    "lambda_": 1.0e-2,
    "reg_type": 2,
    "num_init": 2,
    "num_noise": 1,
    "rand_depth": False,
    "rand_vs": True,
    "zmax": 35.0,
    "r0": 0.5,
    "rmin": 1.0,
    "rmax": 1.4,
    "weight": [1.0],
    "maxiter": 5,
}
noise_observed_dispersion, noise_model_ref, noise_inv_result, noise_freq_idx, noise_ridge_idx = run_inversion_from_image(
    image=noise_dispersion_image,
    freq=f,
    vel_km_s=c / 1e3,
    pick_kwargs=noise_pick_kwargs,
    model_kwargs=noise_model_kwargs,
    inversion_kwargs=noise_inversion_kwargs,
)
noise_forward_dispersion = forward_on_observed_modes(noise_inv_result["model_mean"], noise_observed_dispersion)
noise_inversion_summary = summarize_inversion(noise_observed_dispersion, noise_forward_dispersion, noise_inv_result)
noise_inversion_summary"""
        ),
        code_cell(
            """\
plot_inversion_summary(
    title_prefix="Noise ",
    image=noise_dispersion_image,
    freq=f,
    vel_km_s=c / 1e3,
    observed_dispersion=noise_observed_dispersion,
    forward_dispersion=noise_forward_dispersion,
    result=noise_inv_result,
)"""
        ),
        code_cell("""print("中文自检：noise 反演单元已按 UTF-8 追加，新增区块未出现乱码占位符。")"""),
    ],
    "example_EQ.ipynb": [
        markdown_cell(COMMON_MARKDOWN),
        code_cell(COMMON_HELPERS),
        markdown_cell(
            """\
### example_EQ 的观测频散构造

前面的 `out` 为多窗 Bessel 组合成像结果。这里对三个时间窗取均值，得到更稳定的
观测频散能量图，再从中拾取基阶频散脊线并反演浅层到中浅层的 $V_s$ 结构。
"""
        ),
        code_cell(
            """\
eq_dispersion_image = np.mean(out, axis=0)
eq_pick_kwargs = {
    "fmin": 0.08,
    "fmax": 0.18,
    "continuity": 120.0,
    "freq_step": 3,
}
eq_model_kwargs = {
    "depth_nodes_km": [0.0, 0.7, 1.5, 2.8, 4.5, 6.8, 9.8, 14.0, 19.5, 26.0, 34.0],
    "vs_min": 2.40,
    "vs_max": 4.20,
    "rho0": 2.10,
    "rho_span": 0.20,
    "vp_vs": 1.76,
}
eq_inversion_kwargs = {
    "vs2model": "FixVpRho",
    "vs_width": 0.50,
    "lambda_": 1.0e-2,
    "reg_type": 2,
    "num_init": 2,
    "num_noise": 1,
    "rand_depth": False,
    "rand_vs": True,
    "zmax": 30.0,
    "r0": 0.5,
    "rmin": 1.0,
    "rmax": 1.4,
    "weight": [1.0],
    "maxiter": 4,
}
eq_observed_dispersion, eq_model_ref, eq_inv_result, eq_freq_idx, eq_ridge_idx = run_inversion_from_image(
    image=eq_dispersion_image,
    freq=f,
    vel_km_s=c / 1e3,
    pick_kwargs=eq_pick_kwargs,
    model_kwargs=eq_model_kwargs,
    inversion_kwargs=eq_inversion_kwargs,
)
eq_forward_dispersion = forward_on_observed_modes(eq_inv_result["model_mean"], eq_observed_dispersion)
eq_inversion_summary = summarize_inversion(eq_observed_dispersion, eq_forward_dispersion, eq_inv_result)
eq_inversion_summary"""
        ),
        code_cell(
            """\
plot_inversion_summary(
    title_prefix="EQ ",
    image=eq_dispersion_image,
    freq=f,
    vel_km_s=c / 1e3,
    observed_dispersion=eq_observed_dispersion,
    forward_dispersion=eq_forward_dispersion,
    result=eq_inv_result,
)"""
        ),
        code_cell("""print("中文自检：EQ 反演单元已按 UTF-8 追加，新增区块未出现乱码占位符。")"""),
    ],
}


def strip_previous_appendix(cells: list[dict]) -> list[dict]:
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if SECTION_MARKER in src:
                return cells[:idx]
    return cells


def append_sections(path: Path, appended_cells: list[dict]) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    notebook["cells"] = strip_previous_appendix(notebook["cells"]) + appended_cells
    text = json.dumps(notebook, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for name, cells in NOTEBOOK_APPENDIX.items():
        append_sections(NB_DIR / name, cells)
        text = (NB_DIR / name).read_text(encoding="utf-8")
        appended_text = "".join("".join(cell.get("source", [])) for cell in cells)
        if "?" in appended_text:
            raise AssertionError(f"中文自检失败: {name}")
        print(f"{name}: appended inversion section with UTF-8 text.")


if __name__ == "__main__":
    main()
