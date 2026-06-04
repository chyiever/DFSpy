"""色散曲线搜索。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import toms748

from .secfunc import SecularFunction


def _calculate_newton_step(root: float, vp: float, vs: float) -> float:
    p = root ** -1
    p2 = root ** -2
    vs_2 = vs ** -2
    vp_2 = vp ** -2
    xi = math.sqrt(p2 - vs_2)
    eta = math.sqrt(p2 - vp_2)
    func = (vs_2 - 2.0 * p2) ** 2 - 4.0 * xi * eta * p2
    deri = p2 * (
        8.0 * p * (vs_2 - 2.0 * p2) + 8.0 * p * xi * eta + 4.0 * p2 * p * (xi / eta + eta / xi)
    )
    return func / deri


def _find_extremum(x1: float, x2: float, x3: float, f1: float, f2: float, f3: float) -> float:
    b1 = (x2 * x2 - x3 * x3) * f1
    b2 = (x3 * x3 - x1 * x1) * f2
    b3 = (x1 * x1 - x2 * x2) * f3
    c1 = (x2 - x3) * f1
    c2 = (x3 - x1) * f2
    c3 = (x1 - x2) * f3
    d = (x1 - x2) * (x2 - x3) * (x3 - x1)
    b = (b1 + b2 + b3) / d
    a = -(c1 + c2 + c3) / d
    return -b / (2.0 * a)


@dataclass
class DispersionSolver:
    """色散曲线求解器。"""

    model: np.ndarray
    sh: bool = False

    def __post_init__(self) -> None:
        self.model = np.asarray(self.model, dtype=np.float64)
        self.nl = int(self.model.shape[0])
        self.thk = self.model[1:, 1] - self.model[:-1, 1]
        self.vs = self.model[:, 3]
        self.vp = self.model[:, 4]
        self.itop = 1 if self.model[0, 3] == 0 else 0
        self.sf = SecularFunction(self.model, self.sh)
        self.ednn = 0.50
        self.nfine = 2
        self.ctol = 1.0e-5
        self.nbias = 2
        self.niter_ext = 3
        self.vs0 = self.vs[self.itop]
        self.vp0 = self.vp[self.itop]
        self.vs_min = float(np.min(self.vs[self.itop :]))
        if self.itop == 1 and not self.sh:
            self.vs_min = min(self.vs_min, float(self.vp[0]))
        self.vs_max = float(np.max(self.vs))
        self.vs_hf = float(self.vs[-1])
        self.rayv = self.evaluate_rayleigh_velocity()

    def evaluate_rayleigh_velocity(self) -> float:
        root = 0.8 * self.vs0
        count = 0
        while True:
            step = _calculate_newton_step(root, self.vp0, self.vs0)
            root -= step
            if count > 10 or abs(step) < self.ctol:
                break
            count += 1
        return root

    def approx(self, f: float, c: float) -> float:
        """计算近似模态计数函数。

        输入:
            f:
                类型: `float`
                单位: `Hz`
            c:
                类型: `float`
                单位: `km/s`
        输出:
            `float`
                单位: 无量纲
                含义: 近似累积模态指标。
        """
        total = 0.0
        c2 = c ** -2
        for i in range(self.itop, self.nl - 1):
            if c > self.vs[i]:
                total += math.sqrt(self.vs[i] ** -2 - c2) * self.thk[i]
        if not self.sh:
            for i in range(self.nl - 1):
                if c > self.vp[i]:
                    total += math.sqrt(self.vp[i] ** -2 - c2) * self.thk[i]
        return 2.0 * f * total

    def get_samples(self, f: float) -> list[float]:
        """生成给定频率下的初始相速度采样点。"""
        pred: list[float] = []
        nmax = int(math.floor(self.approx(f, self.vs_hf))) + 1
        dc = (self.vs_hf - self.vs_min) / nmax
        c1 = self.vs_min
        e1 = self.approx(f, c1)
        while c1 < self.vs_hf:
            c2 = c1 + dc
            e2 = self.approx(f, c2)
            while abs(e2 - e1) > self.ednn:
                c2 = c1 + (c2 - c1) * 0.618
                e2 = self.approx(f, c2)
            c1 = c2
            e1 = e2
            if c2 < self.vs_hf:
                pred.append(c2)
        pred.extend([0.8 * self.vs_min, self.vs_hf - self.ctol])
        pred.sort()
        for _ in range(self.nfine):
            mids = [(pred[j] + pred[j + 1]) / 2.0 for j in range(len(pred) - 1)]
            pred.extend(mids)
            pred.sort()
        samples = list(pred)
        samples.extend([self.vs_min - self.ctol * 10.0, self.vs_min + self.ctol * 10.0])
        if not self.sh:
            samples.append(self.rayv)
        samples.sort()
        return samples

    def locate_extremum(self, f: float, x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
        """通过二次插值搜索局部极值点。"""
        x_tmp = list(x)
        y_tmp = list(y)
        x_ext: list[float] = []
        y_ext: list[float] = []
        for it in range(self.niter_ext):
            x_ext = []
            y_ext = []
            for i in range(len(x_tmp) - 2):
                xe = _find_extremum(x_tmp[i], x_tmp[i + 1], x_tmp[i + 2], y_tmp[i], y_tmp[i + 1], y_tmp[i + 2])
                if x_tmp[i] < xe < x_tmp[i + 2]:
                    x_ext.append(xe)
                    y_ext.append(self.sf.evaluate(f, xe))
            if it == self.niter_ext - 1 or len(x_ext) < 3:
                break
            pairs = sorted(zip(x_tmp + x_ext, y_tmp + y_ext), key=lambda item: item[0])
            x_tmp = [item[0] for item in pairs]
            y_tmp = [item[1] for item in pairs]
        return x_ext, y_ext

    def find_coarse_intervals(self, f: float, num_mode: int) -> list[tuple[float, float]]:
        """搜索候选根区间。"""
        samples = self.get_samples(f)
        x_sample = [samples[0]]
        y_sample = [self.sf.evaluate(f, samples[0])]
        count_root = 0
        for i in range(1, len(samples)):
            x_val = samples[i]
            y_val = self.sf.evaluate(f, x_val)
            x_sample.append(x_val)
            y_sample.append(y_val)
            if y_sample[i - 1] * y_sample[i] < 0:
                count_root += 1
            if count_root >= num_mode + self.nbias:
                break
        x_ext: list[float] = []
        y_ext: list[float] = []
        if len(x_sample) >= 3:
            x_ext, y_ext = self.locate_extremum(f, x_sample, y_sample)
        pairs = sorted(zip(x_sample + x_ext, y_sample + y_ext), key=lambda item: item[0])
        find_intv: list[tuple[float, float]] = []
        for i in range(len(pairs) - 1):
            if pairs[i][1] * pairs[i + 1][1] < 0:
                find_intv.append((pairs[i][0], pairs[i + 1][0]))
        return find_intv[:num_mode]

    def search(self, f: float, num_mode: int) -> list[float]:
        """计算指定频率的前 `num_mode` 个模态相速度。

        输入:
            f:
                类型: `float`
                单位: `Hz`
            num_mode:
                类型: `int`
                单位: 无
                含义: 需要返回的模态数。
        输出:
            `list[float]`
                单位: `km/s`
                含义: 从基阶开始的相速度列表。
        """
        dedup_tol = 1.0e-7
        found: list[float] = []
        for c1, c2 in self.find_coarse_intervals(f, num_mode):
            try:
                root = toms748(lambda c: self.sf.evaluate(f, c), c1, c2, xtol=1.0e-12, rtol=1.0e-12, maxiter=100)
            except ValueError:
                continue
            if not found or abs(root - found[-1]) > dedup_tol:
                found.append(root)
        return found[:num_mode]

    def search_mode(self, f: float, mode: int) -> float:
        """计算指定模态的相速度。

        输入:
            f:
                类型: `float`
                单位: `Hz`
            mode:
                类型: `int`
                单位: 无
                含义: 模态序号，基阶为 0。
        输出:
            `float`
                单位: `km/s`
                含义: 指定模态相速度；若未找到，返回 `nan`。
        """
        cs = self.search(f, mode + 1)
        if len(cs) < mode + 1:
            return float("nan")
        return float(cs[mode])
