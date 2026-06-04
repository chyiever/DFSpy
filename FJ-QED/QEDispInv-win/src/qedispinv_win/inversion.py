"""反演流程实现。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .dispersion import DispersionSolver
from .modeling import (
    Brocher05Converter,
    FixVpRhoConverter,
    GardnerConverter,
    NearSurfaceConverter,
    StatisticsResult,
    compute_hist2d,
    compute_statistics,
    detect_outliers,
    generate_depth_by_layer_ratio,
)
from .sensitivity import compute_phase_velocity_kernel


@dataclass
class DataSet:
    """观测色散数据集。"""

    raw: np.ndarray

    def __post_init__(self) -> None:
        """按模态整理观测色散数据。

        输入:
            `raw`:
                类型: `numpy.ndarray`
                形状: `(n, 3)` 或 `(n, 4)`
                单位: 第 1 列 `Hz`，第 2 列 `km/s`，第 4 列可选 `km/s`
                含义: 原始色散数据，列为 `频率 / 相速度 / 模态号 / 可选标准差`。
        输出:
            无。
        """
        disp = np.asarray(self.raw, dtype=np.float64)
        self.raw = disp
        self.mode_set = sorted({int(m) for m in disp[:, 2]})
        self.freq: dict[int, np.ndarray] = {}
        self.phase_velocity: dict[int, np.ndarray] = {}
        self.sigma: dict[int, np.ndarray] = {}
        wavelengths = []
        for mode in self.mode_set:
            sub = disp[disp[:, 2].astype(int) == mode]
            self.freq[mode] = sub[:, 0].copy()
            self.phase_velocity[mode] = sub[:, 1].copy()
            if disp.shape[1] >= 4:
                self.sigma[mode] = sub[:, 3].copy()
            if mode == 0:
                wavelengths.extend((sub[:, 1] / sub[:, 0]).tolist())
        self.cmin = float(np.min(disp[:, 1]))
        self.cmax = float(np.max(disp[:, 1]))
        self.lmin = float(np.min(wavelengths)) if wavelengths else 0.0
        self.lmax = float(np.max(wavelengths)) if wavelengths else 0.0

    def add_sigma(self, sigma_by_mode: list[float]) -> None:
        """为不含误差列的数据补充按模态常数标准差。

        输入:
            sigma_by_mode:
                类型: `list[float]`
                形状: `(nmodes,)`
                单位: `km/s`
                含义: 每个模态对应的相速度标准差。
        输出:
            无。
        """
        for mode, sigma in enumerate(sigma_by_mode):
            if mode in self.freq:
                self.sigma[mode] = np.full(self.freq[mode].shape, float(sigma), dtype=np.float64)

    def resample(self, rng: np.random.Generator) -> "DataSet":
        """按现有标准差对观测数据做重采样。

        输入:
            rng:
                类型: `numpy.random.Generator`
                单位: 无
                含义: 随机数生成器。
        输出:
            `DataSet`
                含义: 加噪后的新观测数据集。
        """
        data = self.raw.copy()
        if self.raw.shape[1] >= 4:
            for mode in self.mode_set:
                idx = data[:, 2].astype(int) == mode
                data[idx, 1] += rng.uniform(-1.0, 1.0, np.sum(idx)) * self.sigma[mode]
        else:
            for mode in self.mode_set:
                if mode in self.sigma:
                    idx = data[:, 2].astype(int) == mode
                    data[idx, 1] += rng.uniform(-1.0, 1.0, np.sum(idx)) * self.sigma[mode]
        return DataSet(data)


@dataclass
class InversionConfig:
    """反演配置参数。"""

    vs2model: str
    vs_width: float
    lambda_: float
    reg_type: int
    num_init: int
    num_noise: int
    rand_depth: bool
    rand_vs: bool
    zmax: float
    r0: float
    rmin: float
    rmax: float
    weight: list[float]
    maxiter: int = 100
    vp2vs: float | None = None
    sigma: list[float] | None = None


class InversionRunner:
    """多初值 L-BFGS-B 反演执行器。"""

    def __init__(
        self,
        model_ref: np.ndarray,
        data: DataSet,
        config: InversionConfig,
        sh: bool = False,
        seed: int = 20260604,
    ) -> None:
        """初始化反演器。

        输入:
            model_ref:
                类型: `numpy.ndarray`
                形状: `(nl, 5)`
                单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
                含义: 参考层状模型。
            data:
                类型: `DataSet`
                含义: 观测色散数据集。
            config:
                类型: `InversionConfig`
                含义: 反演配置。
            sh:
                类型: `bool`
                单位: 无
                含义: 是否执行 Love 波反演。
            seed:
                类型: `int`
                单位: 无
                含义: 随机种子。
        输出:
            无。
        """
        self.model_ref = np.asarray(model_ref, dtype=np.float64)
        self.data = data
        self.config = config
        self.sh = bool(sh)
        self.rng = np.random.default_rng(seed)
        self.converter = self._build_converter()

    def _build_converter(self):
        """按配置选择 `Vs -> 模型` 转换器。"""
        mode = self.config.vs2model.lower()
        if mode == "nearsurface":
            if self.config.vp2vs is None:
                raise ValueError("NearSurface 模式需要 vp2vs。")
            return NearSurfaceConverter(self.model_ref, self.config.vp2vs)
        if mode == "gardner":
            return GardnerConverter(self.model_ref)
        if mode == "fixvprho":
            return FixVpRhoConverter(self.model_ref)
        if mode == "brocher05":
            return Brocher05Converter(self.model_ref)
        raise ValueError(f"无效的 vs2model: {self.config.vs2model}")

    def _regularization_matrix(self, z_model: np.ndarray, vs_ref: np.ndarray) -> np.ndarray:
        """构造正则化矩阵。"""
        nx = z_model.size
        mat_l = np.zeros((nx - 1, nx), dtype=np.float64)
        if self.config.reg_type == 1:
            for i in range(nx - 1):
                mat_l[i, i] = 1.0
                mat_l[i, i + 1] = -1.0
        else:
            diff = np.abs(vs_ref[:-1] - vs_ref[1:])
            a = np.max(diff) * 0.1 if np.max(diff) > 0 else 1.0
            w = a / (a + diff)
            for i in range(nx - 1):
                mat_l[i, i] = w[i]
                mat_l[i, i + 1] = -w[i]
        return self.config.lambda_ * mat_l.T @ mat_l

    def _flatten_observed(self, data: DataSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将分模态观测数据展开为并行数组。"""
        f_obs = []
        c_obs = []
        m_obs = []
        for mode in data.mode_set:
            if mode >= len(self.config.weight) or self.config.weight[mode] == 0.0:
                continue
            f_obs.append(data.freq[mode])
            c_obs.append(data.phase_velocity[mode])
            m_obs.append(np.full(data.freq[mode].shape, mode, dtype=np.int32))
        return np.concatenate(f_obs), np.concatenate(c_obs), np.concatenate(m_obs)

    def _forward_modes(self, model: np.ndarray, f_obs: np.ndarray, m_obs: np.ndarray) -> np.ndarray:
        """根据模型计算全部观测点对应的理论相速度。"""
        disp = DispersionSolver(model, self.sh)
        out = np.full(f_obs.shape, np.nan, dtype=np.float64)
        for i in range(f_obs.size):
            out[i] = disp.search_mode(float(f_obs[i]), int(m_obs[i]))
        return out

    def _objective_and_gradient(
        self,
        z_model: np.ndarray,
        vs_ref: np.ndarray,
        f_obs: np.ndarray,
        c_obs: np.ndarray,
        m_obs: np.ndarray,
        vs: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """同时计算目标函数、梯度与理论色散。

        输入:
            z_model:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km`
            vs_ref:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km/s`
            f_obs:
                类型: `numpy.ndarray`
                形状: `(nobs,)`
                单位: `Hz`
            c_obs:
                类型: `numpy.ndarray`
                形状: `(nobs,)`
                单位: `km/s`
            m_obs:
                类型: `numpy.ndarray`
                形状: `(nobs,)`
                单位: 无
            vs:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km/s`
        输出:
            `tuple[float, numpy.ndarray, numpy.ndarray]`
                第 1 项: 目标函数值，单位 `km^2/s^2`
                第 2 项: 梯度，形状 `(nl,)`，单位 `(km^2/s^2)/(km/s)`
                第 3 项: 理论相速度，形状 `(nobs,)`，单位 `km/s`
        """
        mat_m = self._regularization_matrix(z_model, vs_ref)
        weight_arr = np.asarray(self.config.weight, dtype=np.float64)
        mode_counts = np.zeros(max(len(weight_arr), int(np.max(m_obs)) + 1), dtype=np.int32)
        for mode in m_obs:
            mode_counts[int(mode)] += 1

        model = self.converter.generate(z_model, vs)
        chain = self.converter.derivative(vs)
        c_syn = self._forward_modes(model, f_obs, m_obs)
        residual = 0.0
        grad = np.zeros(vs.shape, dtype=np.float64)

        for i in range(f_obs.size):
            if np.isnan(c_syn[i]):
                continue
            mode = int(m_obs[i])
            weight = weight_arr[mode] / max(mode_counts[mode], 1)
            misfit = c_syn[i] - c_obs[i]
            residual += weight * misfit**2
            kernel = compute_phase_velocity_kernel(model, float(f_obs[i]), float(c_syn[i]), self.sh)
            grad += (
                2.0
                * weight
                * misfit
                * (
                    kernel.rho * chain["rho"]
                    + kernel.vs * chain["vs"]
                    + kernel.vp * chain["vp"]
                )
            )

        diff = vs - vs_ref
        reg = float((diff.T @ mat_m @ diff) / z_model.size)
        grad += (2.0 / z_model.size) * (mat_m @ diff)
        return float(residual + reg), grad, c_syn

    def run(self) -> dict[str, np.ndarray | list[np.ndarray] | StatisticsResult]:
        """执行多初值反演。

        输出:
            `dict`
                含义: 反演结果集合，可直接存储为 `.npz`。
        """
        if self.config.num_noise > 1 and self.data.raw.shape[1] < 4 and self.config.sigma:
            self.data.add_sigma(self.config.sigma)

        data_noise = [self.data] if self.config.num_noise == 1 else [self.data.resample(self.rng) for _ in range(self.config.num_noise)]
        z_init_list = []
        for _ in range(self.config.num_init):
            if self.config.rand_depth and self.config.num_init > 1:
                z_init = generate_depth_by_layer_ratio(
                    self.data.lmin,
                    self.data.lmax,
                    self.config.r0,
                    self.config.rmin,
                    self.config.rmax,
                    self.config.zmax,
                    self.rng,
                )
            else:
                z_init = self.model_ref[:, 1].copy()
            z_init_list.append(z_init)

        vs_ref_list: list[np.ndarray] = []
        lb_list: list[np.ndarray] = []
        ub_list: list[np.ndarray] = []
        vs_init_list: list[np.ndarray] = []
        for z_init in z_init_list:
            vs_ref, lb, ub = self.converter.get_vs_limits(z_init, self.config.vs_width)
            if self.data.lmax / 2.0 > self.config.zmax and self.data.lmax / 2.0 >= z_init[-1]:
                half_lmax = self.data.lmax / 2.0
                cmax = max(np.max(self.data.phase_velocity[0]), vs_ref[-1])
                z_init = np.append(z_init, half_lmax)
                vs_ref = np.append(vs_ref, cmax * 1.01)
                lb = np.append(lb, cmax)
                ub = np.append(ub, cmax + self.config.vs_width)
            if self.config.rand_vs and self.config.num_init > 1:
                vs_init = lb + (ub - lb) * self.rng.random(lb.size)
            else:
                vs_init = self.converter.interp_vs(z_init)
            vs_ref_list.append(vs_ref)
            lb_list.append(lb)
            ub_list.append(ub)
            vs_init_list.append(vs_init)

        z_inv: list[np.ndarray] = []
        vs_inv: list[np.ndarray] = []
        fitness: list[float] = []
        niter: list[int] = []
        disp_syn: list[np.ndarray] = []
        model_init_dump: list[np.ndarray] = []

        for data_resampled in data_noise:
            f_obs, c_obs, m_obs = self._flatten_observed(data_resampled)
            for z_init, vs_ref, lb, ub, vs_init in zip(z_init_list, vs_ref_list, lb_list, ub_list, vs_init_list):
                cache: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}

                def objective(vs_trial: np.ndarray) -> tuple[float, np.ndarray]:
                    key = np.array2string(vs_trial, precision=12, separator=",")
                    if key not in cache:
                        cache[key] = self._objective_and_gradient(z_init, vs_ref, f_obs, c_obs, m_obs, vs_trial)
                    val, grad, _ = cache[key]
                    return val, grad

                result = minimize(
                    lambda x: objective(x)[0],
                    x0=vs_init,
                    jac=lambda x: objective(x)[1],
                    method="L-BFGS-B",
                    bounds=list(zip(lb, ub)),
                    options={"maxiter": self.config.maxiter, "ftol": 1.0e-9, "gtol": 1.0e-5},
                )
                final_value, _, c_syn = self._objective_and_gradient(z_init, vs_ref, f_obs, c_obs, m_obs, result.x)
                vs_best = result.x.astype(np.float64, copy=True)
                disp_rows = np.column_stack([f_obs[~np.isnan(c_syn)], c_syn[~np.isnan(c_syn)], m_obs[~np.isnan(c_syn)]])
                z_inv.append(z_init.copy())
                vs_inv.append(vs_best)
                fitness.append(float(final_value))
                niter.append(int(result.nit))
                disp_syn.append(disp_rows)
                model_init_dump.append(self.converter.generate(z_init, vs_init))

        fitness_arr = np.asarray(fitness, dtype=np.float64)
        niter_arr = np.asarray(niter, dtype=np.int32)
        if fitness_arr.size > 1:
            outliers = detect_outliers(fitness_arr)
            if outliers.size > 0:
                keep = np.ones(fitness_arr.size, dtype=bool)
                keep[outliers] = False
                fitness_arr = fitness_arr[keep]
                niter_arr = niter_arr[keep]
                z_inv = [item for idx, item in enumerate(z_inv) if keep[idx]]
                vs_inv = [item for idx, item in enumerate(vs_inv) if keep[idx]]
                disp_syn = [item for idx, item in enumerate(disp_syn) if keep[idx]]

        vsmin = min(np.min(v) for v in vs_inv) * 0.95
        vsmax = max(np.max(v) for v in vs_inv) * 1.05
        z_sample, vs_sample, hist = compute_hist2d(
            z_inv, vs_inv, fitness_arr, vsmin, vsmax, self.config.zmax, 100
        )
        stats = compute_statistics(z_sample, vs_sample, hist)
        model_mean = self.converter.generate(stats.z_sample, stats.vs_mean)
        vs_ref_save = self.converter.interp_vs(stats.z_sample)
        mode_used = np.asarray([i for i, w in enumerate(self.config.weight) if w > 0], dtype=np.int32)
        return {
            "fitness": fitness_arr,
            "niter": niter_arr,
            "z_sample": stats.z_sample,
            "vs_sample": stats.vs_sample,
            "vs_hist2d": stats.vs_hist2d,
            "data": self.data.raw,
            "vs_mean": stats.vs_mean,
            "vs_median": stats.vs_median,
            "vs_mode": stats.vs_mode,
            "vs_cred10": stats.vs_cred10,
            "vs_cred90": stats.vs_cred90,
            "model_mean": model_mean,
            "vs_ref": vs_ref_save,
            "num_init": np.asarray([self.config.num_init], dtype=np.int32),
            "mode_used": mode_used,
            "num_valid": np.asarray([fitness_arr.size], dtype=np.int32),
            "disp_syn_list": np.asarray(disp_syn, dtype=object),
            "model_init_list": np.asarray(model_init_dump, dtype=object),
        }
