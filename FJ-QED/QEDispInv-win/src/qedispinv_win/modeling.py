"""模型参数化、经验关系与统计工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def interp1d_constant_edge(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """执行带端点常值外推的一维线性插值。

    输入:
        x_old:
            类型: `numpy.ndarray`
            形状: `(n,)`
            单位: 由调用者决定，通常为 `km`
            含义: 原始自变量采样点，要求严格递增。
        y_old:
            类型: `numpy.ndarray`
            形状: `(n,)`
            单位: 与被插值物理量一致
            含义: 原始函数值。
        x_new:
            类型: `numpy.ndarray`
            形状: `(m,)`
            单位: 与 `x_old` 一致
            含义: 目标插值位置。
    输出:
        `numpy.ndarray`
            类型: `float64`
            形状: `(m,)`
            单位: 与 `y_old` 一致
            含义: 插值结果；超出边界时保持端点常值。
    """
    return np.interp(x_new, x_old, y_old, left=y_old[0], right=y_old[-1])


class Vs2ModelConverter:
    """由 `Vs` 参数恢复完整层状模型的抽象基类。"""

    def __init__(self, model: np.ndarray) -> None:
        """初始化参考模型。

        输入:
            model:
                类型: `numpy.ndarray`
                形状: `(nl, 5)`
                单位: 第 2 列 `km`，第 3 列 `g/cm^3`，第 4/5 列 `km/s`
                含义: 参考层状模型。
        输出:
            无。
        """
        model = np.asarray(model, dtype=np.float64)
        self.z = model[:, 1]
        self.rho = model[:, 2]
        self.vs = model[:, 3]
        self.vp = model[:, 4]

    def z_to_interp_depth(self, z: np.ndarray) -> np.ndarray:
        """将层界面深度转换为插值深度点。

        输入:
            z:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km`
                含义: 层界面深度。
        输出:
            `numpy.ndarray`
                类型: `float64`
                形状: `(nl,)`
                单位: `km`
                含义: 与参考项目一致的层中心/末层插值深度。
        """
        dep = np.zeros_like(z, dtype=np.float64)
        dep[0] = 0.0
        if z.size > 2:
            dep[1:-1] = (z[1:-1] + z[2:]) / 2.0
        dep[-1] = z[-1]
        return dep

    def interp_vs(self, z: np.ndarray) -> np.ndarray:
        """把参考模型的 `Vs` 插值到目标深度节点。

        输入:
            z:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km`
                含义: 目标深度节点。
        输出:
            `numpy.ndarray`
                类型: `float64`
                形状: `(nl,)`
                单位: `km/s`
                含义: 插值得到的参考 `Vs`。
        """
        dep = self.z_to_interp_depth(np.asarray(z, dtype=np.float64))
        return interp1d_constant_edge(self.z, self.vs, dep)

    def get_vs_limits(
        self,
        z: np.ndarray,
        vs_width: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 `Vs` 的参考值、下界与上界。

        输入:
            z:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km`
                含义: 目标深度节点。
            vs_width:
                类型: `float`
                单位: `km/s`
                含义: `Vs` 搜索范围总宽度。
        输出:
            `tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]`
                第 1 项: 参考 `Vs`，单位 `km/s`
                第 2 项: 下界 `lb`，单位 `km/s`
                第 3 项: 上界 `ub`，单位 `km/s`
        """
        vs_ref = self.interp_vs(z)
        lb = np.maximum(0.01, vs_ref - vs_width / 2.0)
        ub = vs_ref + vs_width / 2.0
        return vs_ref, lb, ub

    def generate(self, z: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """根据深度与 `Vs` 构造完整模型。"""
        raise NotImplementedError

    def derivative(self, vs: np.ndarray) -> dict[str, np.ndarray]:
        """给出 `Vp/Vs/Rho` 对反演变量 `Vs` 的导数。

        输入:
            vs:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km/s`
                含义: 当前反演变量。
        输出:
            `dict[str, numpy.ndarray]`
                键包含 `vp`、`vs`、`rho`
                各值形状均为 `(nl,)`
                单位分别为:
                - `vp`: `(km/s)/(km/s)`，无量纲
                - `vs`: `(km/s)/(km/s)`，无量纲
                - `rho`: `(g/cm^3)/(km/s)`
        """
        raise NotImplementedError


class FixVpRhoConverter(Vs2ModelConverter):
    """固定 `Vp` 与密度，仅反演 `Vs`。"""

    def generate(self, z: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """生成完整模型。

        输入:
            z:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km`
            vs:
                类型: `numpy.ndarray`
                形状: `(nl,)`
                单位: `km/s`
        输出:
            `numpy.ndarray`
                类型: `float64`
                形状: `(nl, 5)`
                单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
        """
        z = np.asarray(z, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)
        dep = self.z_to_interp_depth(z)
        model = np.zeros((vs.size, 5), dtype=np.float64)
        model[:, 0] = np.arange(vs.size) + 1.0
        model[:, 1] = z
        model[:, 2] = interp1d_constant_edge(self.z, self.rho, dep)
        model[:, 3] = vs
        model[:, 4] = interp1d_constant_edge(self.z, self.vp, dep)
        return model

    def derivative(self, vs: np.ndarray) -> dict[str, np.ndarray]:
        """返回固定 `Vp/Rho` 模式下的导数。"""
        nl = np.asarray(vs).size
        return {
            "rho": np.zeros(nl, dtype=np.float64),
            "vp": np.zeros(nl, dtype=np.float64),
            "vs": np.ones(nl, dtype=np.float64),
        }

    def get_vs_limits(
        self,
        z: np.ndarray,
        vs_width: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在 `Vs < Vp` 约束下计算上下界。"""
        vs_ref = self.interp_vs(z)
        dep = self.z_to_interp_depth(np.asarray(z, dtype=np.float64))
        vp_ref = interp1d_constant_edge(self.z, self.vp, dep)
        lb = np.maximum(0.01, vs_ref - vs_width / 2.0)
        ub = np.minimum(vp_ref - 1.0e-2, vs_ref + vs_width / 2.0)
        return vs_ref, lb, ub


class Brocher05Converter(Vs2ModelConverter):
    """Brocher (2005) 经验关系。"""

    def generate(self, z: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """按 Brocher 关系构造模型。"""
        z = np.asarray(z, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)
        vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
        rho = 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5
        model = np.zeros((vs.size, 5), dtype=np.float64)
        model[:, 0] = np.arange(vs.size) + 1.0
        model[:, 1] = z
        model[:, 2] = rho
        model[:, 3] = vs
        model[:, 4] = vp
        return model

    def derivative(self, vs: np.ndarray) -> dict[str, np.ndarray]:
        """返回 Brocher 经验关系的链式导数。"""
        vs = np.asarray(vs, dtype=np.float64)
        dvp = 2.0947 - 0.8206 * vs * 2.0 + 0.2683 * vs**2 * 3.0 - 0.0251 * vs**3 * 4.0
        vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
        drho = 1.6612 - 0.4721 * vp * 2.0 + 0.0671 * vp**2 * 3.0 - 0.0043 * vp**3 * 4.0 + 0.000106 * vp**4 * 5.0
        drho = drho * dvp
        return {
            "rho": drho,
            "vp": dvp,
            "vs": np.ones(vs.size, dtype=np.float64),
        }


class GardnerConverter(Vs2ModelConverter):
    """Gardner 经验关系。"""

    def __init__(self, model: np.ndarray, vp2vs: float = 1.7321) -> None:
        """初始化 Gardner 关系参数。"""
        super().__init__(model)
        self.vp2vs = float(vp2vs)
        self.alpha = 0.31
        self.beta = 0.25

    def generate(self, z: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """按 Gardner 关系构造模型。"""
        z = np.asarray(z, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)
        vp = self.vp2vs * vs
        rho = self.alpha * (vp * 1000.0) ** self.beta
        model = np.zeros((vs.size, 5), dtype=np.float64)
        model[:, 0] = np.arange(vs.size) + 1.0
        model[:, 1] = z
        model[:, 2] = rho
        model[:, 3] = vs
        model[:, 4] = vp
        return model

    def derivative(self, vs: np.ndarray) -> dict[str, np.ndarray]:
        """返回 Gardner 经验关系的链式导数。"""
        vs = np.asarray(vs, dtype=np.float64)
        dvs = np.ones(vs.size, dtype=np.float64)
        dvp = self.vp2vs * dvs
        vp = self.vp2vs * vs
        drho = self.alpha * self.beta * (1000.0 * vp) ** (self.beta - 1.0) * 1000.0 * dvp
        return {"rho": drho, "vp": dvp, "vs": dvs}


class NearSurfaceConverter(Vs2ModelConverter):
    """浅层近地表经验关系。"""

    def __init__(self, model: np.ndarray, vp2vs: float) -> None:
        """初始化浅层经验关系参数。"""
        super().__init__(model)
        self.vp2vs = float(vp2vs)
        self.a = -0.22374079
        self.b = 1.32248261
        self.c = 1.54840433

    def generate(self, z: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """按浅层经验关系构造模型。"""
        z = np.asarray(z, dtype=np.float64)
        vs = np.asarray(vs, dtype=np.float64)
        vp = self.vp2vs * vs
        rho = self.a * vs**2 + self.b * vs + self.c
        model = np.zeros((vs.size, 5), dtype=np.float64)
        model[:, 0] = np.arange(vs.size) + 1.0
        model[:, 1] = z
        model[:, 2] = rho
        model[:, 3] = vs
        model[:, 4] = vp
        return model

    def derivative(self, vs: np.ndarray) -> dict[str, np.ndarray]:
        """返回浅层经验关系的链式导数。"""
        vs = np.asarray(vs, dtype=np.float64)
        dvs = np.ones(vs.size, dtype=np.float64)
        dvp = self.vp2vs * dvs
        drho = (2.0 * self.a * vs + self.b) * dvs
        return {"rho": drho, "vp": dvp, "vs": dvs}


def generate_depth_by_layer_ratio(
    lmin: float,
    lmax: float,
    r0: float,
    rmin: float,
    rmax: float,
    zmax: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """按照参考项目的比例分层策略生成深度节点。

    输入:
        lmin:
            类型: `float`
            单位: `km`
            含义: 基阶最短波长。
        lmax:
            类型: `float`
            单位: `km`
            含义: 基阶最长波长。
        r0:
            类型: `float`
            单位: 无
            含义: 首层厚度比例参数。
        rmin, rmax:
            类型: `float`
            单位: 无
            含义: 深层厚度递增比例上下界。
        zmax:
            类型: `float`
            单位: `km`
            含义: 统计或约束深度上限。
        rng:
            类型: `numpy.random.Generator | None`
            单位: 无
            含义: 随机数生成器；为空时自动创建。
    输出:
        `numpy.ndarray`
            类型: `float64`
            形状: `(nl,)`
            单位: `km`
            含义: 生成的深度节点。
    """
    rng = np.random.default_rng() if rng is None else rng
    depmax = max(lmax / 2.0, zmax)
    depth = [0.0, r0 * lmin / 2.0]
    while depth[-1] < depmax:
        ratio = rng.uniform(rmin, rmax)
        depth.append(depth[-1] + ratio * (depth[-1] - depth[-2]))
    depth_arr = np.zeros(len(depth), dtype=np.float64)
    for i in range(len(depth) - 1):
        depth_arr[i + 1] = rng.uniform(depth[i], depth[i + 1])
    return depth_arr


@dataclass
class StatisticsResult:
    """反演统计结果容器。"""

    z_sample: np.ndarray
    vs_sample: np.ndarray
    vs_hist2d: np.ndarray
    vs_mean: np.ndarray
    vs_median: np.ndarray
    vs_mode: np.ndarray
    vs_cred10: np.ndarray
    vs_cred90: np.ndarray


def compute_hist2d(
    z_inv: list[np.ndarray],
    vs_inv: list[np.ndarray],
    fitness: np.ndarray,
    vsmin: float,
    vsmax: float,
    zmax: float,
    num_hist: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建深度-速度二维直方图。

    输入:
        z_inv:
            类型: `list[numpy.ndarray]`
            单位: `km`
            含义: 每次反演的深度节点。
        vs_inv:
            类型: `list[numpy.ndarray]`
            单位: `km/s`
            含义: 每次反演的速度结果。
        fitness:
            类型: `numpy.ndarray`
            单位: `km^2/s^2`
            含义: 各次反演目标函数值。
        vsmin, vsmax:
            类型: `float`
            单位: `km/s`
            含义: 统计速度范围。
        zmax:
            类型: `float`
            单位: `km`
            含义: 统计深度上限。
        num_hist:
            类型: `int`
            单位: 无
            含义: 深度与速度两个方向的采样点数。
    输出:
        `tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]`
            第 1 项: `z_samples`，单位 `km`
            第 2 项: `vs_samples`，单位 `km/s`
            第 3 项: `hist2d`，无量纲
    """
    dz = zmax / (num_hist - 1)
    dvs = (vsmax - vsmin) / (num_hist - 1)
    f_val = 1.0 / np.asarray(fitness, dtype=np.float64)
    f_val /= np.max(f_val)
    z_samples = np.linspace(0.0, zmax, num_hist)
    vs_samples = np.linspace(vsmin, vsmax, num_hist)
    hist2d = np.zeros((num_hist, num_hist), dtype=np.float64)
    for n, z in enumerate(z_inv):
        vs = vs_inv[n]
        i_z = 0
        for i in range(z.size):
            if z[i] > zmax:
                break
            i_v = int(np.floor((vs[i] - vsmin) / dvs))
            i_v = min(max(i_v, 0), num_hist - 1)
            zub = zmax if i == z.size - 1 else min(z[i + 1], zmax)
            while i_z < num_hist and i_z * dz <= zub:
                hist2d[i_z, i_v] += f_val[n]
                i_z += 1
    return z_samples, vs_samples, hist2d


def compute_statistics(
    z: np.ndarray,
    vs: np.ndarray,
    hist: np.ndarray,
) -> StatisticsResult:
    """从二维直方图提取统计量。

    输入:
        z:
            类型: `numpy.ndarray`
            形状: `(nz,)`
            单位: `km`
            含义: 深度采样点。
        vs:
            类型: `numpy.ndarray`
            形状: `(nv,)`
            单位: `km/s`
            含义: 速度采样点。
        hist:
            类型: `numpy.ndarray`
            形状: `(nz, nv)`
            单位: 无量纲
            含义: 深度-速度二维统计图。
    输出:
        `StatisticsResult`
            含义: 均值、中位数、众数、P10、P90 等统计结果。
    """
    n = vs.size
    vs_mean = np.zeros(z.size, dtype=np.float64)
    vs_mode = np.zeros(z.size, dtype=np.float64)
    vs_median = np.zeros(z.size, dtype=np.float64)
    vs_cred10 = np.zeros(z.size, dtype=np.float64)
    vs_cred90 = np.zeros(z.size, dtype=np.float64)
    for i_z in range(z.size):
        hist1d = hist[i_z]
        total_weight = np.sum(hist1d)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive.")
        vs_mean[i_z] = np.sum(vs * hist1d) / total_weight
        vs_mode[i_z] = vs[np.argmax(hist1d)]
        cum_weights = np.cumsum(hist1d)

        def quantile(q: float) -> float:
            target = total_weight * q
            idx = int(np.searchsorted(cum_weights, target, side="left"))
            if idx == 0:
                return float(vs[0])
            if idx >= n:
                return float(vs[-1])
            weight_before = cum_weights[idx - 1]
            weight_here = cum_weights[idx]
            x_before = vs[idx - 1]
            x_here = vs[idx]
            if weight_here <= weight_before:
                return float(x_before)
            t = (target - weight_before) / (weight_here - weight_before)
            return float(x_before + t * (x_here - x_before))

        vs_median[i_z] = quantile(0.5)
        vs_cred10[i_z] = quantile(0.1)
        vs_cred90[i_z] = quantile(0.9)
    return StatisticsResult(
        z_sample=z,
        vs_sample=vs,
        vs_hist2d=hist,
        vs_mean=vs_mean,
        vs_median=vs_median,
        vs_mode=vs_mode,
        vs_cred10=vs_cred10,
        vs_cred90=vs_cred90,
    )


def detect_outliers(
    fitness: np.ndarray,
    multiplier: float = 1.5,
) -> np.ndarray:
    """依据四分位距法检测异常反演结果。

    输入:
        fitness:
            类型: `numpy.ndarray`
            形状: `(n,)`
            单位: `km^2/s^2`
            含义: 各次反演目标函数值。
        multiplier:
            类型: `float`
            单位: 无
            含义: 四分位距倍数阈值。
    输出:
        `numpy.ndarray`
            类型: `int64`
            形状: `(k,)`
            单位: 无
            含义: 异常值在原数组中的索引。
    """
    q1, q3 = np.percentile(fitness, [25.0, 75.0])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return np.where((fitness < lower) | (fitness > upper))[0]
