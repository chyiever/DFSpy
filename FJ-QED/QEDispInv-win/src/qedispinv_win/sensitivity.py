"""相速度敏感核统一入口。"""

from __future__ import annotations

from .fortran_kernels import (
    FortranKernelLibrary,
    PhaseVelocityKernel,
    get_fortran_kernel_library,
)


def compute_phase_velocity_kernel(
    model,
    freq: float,
    phase_velocity: float,
    sh: bool = False,
    rel_step: float = 1.0e-5,
) -> PhaseVelocityKernel:
    """计算单个频率-模态点的相速度敏感核。

    输入:
        model:
            类型: `numpy.ndarray`
            形状: `(nl, 5)`
            单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            含义: 层状模型。
        freq:
            类型: `float`
            单位: `Hz`
            含义: 频率。
        phase_velocity:
            类型: `float`
            单位: `km/s`
            含义: 已求得的相速度。
        sh:
            类型: `bool`
            单位: 无
            含义: 是否为 Love 波。
        rel_step:
            类型: `float`
            单位: 无
            含义: 保留接口兼容；Fortran 核路径下不再使用。
    输出:
        `PhaseVelocityKernel`
            含义: `vp/vs/rho/thickness/group_velocity` 敏感核结果。
    """
    _ = rel_step
    lib: FortranKernelLibrary = get_fortran_kernel_library()
    if sh:
        return lib.love_kernel(model, freq, phase_velocity)
    return lib.rayleigh_kernel(model, freq, phase_velocity)
