"""Fortran 核函数动态库包装层。"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from numpy.ctypeslib import ndpointer


@dataclass
class PhaseVelocityKernel:
    """单个频率-模态点的相速度敏感核。"""

    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    thickness: np.ndarray
    group_velocity: float


class FortranKernelLibrary:
    """对 `sregn96_/slegn96_` 的 `ctypes` 包装。"""

    def __init__(self, dll_path: str | Path | None = None) -> None:
        """加载 Fortran 动态库并配置函数签名。

        输入:
            dll_path:
                类型: `str | pathlib.Path | None`
                单位: 无
                含义: 动态库路径；为空时按项目默认位置查找。
        输出:
            无。
        """
        root = Path(__file__).resolve().parents[2]
        self.root = root
        self.dll_path = self.resolve_dll_path(dll_path)
        if not self.dll_path.exists():
            raise FileNotFoundError(f"Fortran 动态库不存在: {self.dll_path}")

        for runtime_dir in self.runtime_search_dirs():
            if runtime_dir.exists():
                os.add_dll_directory(str(runtime_dir))
        self.lib = ctypes.CDLL(str(self.dll_path.resolve()))
        self._configure_signatures()

    def resolve_dll_path(self, dll_path: str | Path | None) -> Path:
        """解析 Fortran 动态库路径。

        输入:
            dll_path:
                类型: `str | pathlib.Path | None`
                单位: 无
                含义: 外部显式指定的动态库路径。
        输出:
            `pathlib.Path`
                单位: 无
                含义: 实际使用的 DLL 路径。
        """
        if dll_path is not None:
            return Path(dll_path)
        env_path = os.environ.get("QEDISPINV_FORTRAN_DLL")
        if env_path:
            return Path(env_path)
        return self.root / "build" / "cpskernels.dll"

    @staticmethod
    def runtime_search_dirs() -> list[Path]:
        """返回 Windows 下可能包含 Fortran 运行时 DLL 的搜索目录。

        输入:
            无。
        输出:
            `list[pathlib.Path]`
                单位: 无
                含义: 需要通过 `os.add_dll_directory` 加入的目录列表。
        """
        dirs: list[Path] = []
        env_dir = os.environ.get("QEDISPINV_MINGW_BIN")
        if env_dir:
            dirs.append(Path(env_dir))

        metadata_path = Path(__file__).resolve().parents[2] / "build" / "cpskernels.runtime.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for item in metadata.get("runtime_dirs", []):
                dirs.append(Path(item))

        python_root = Path(sys.executable).resolve().parent.parent
        dirs.append(python_root / "Library" / "mingw-w64" / "bin")
        dirs.append(python_root / "Library" / "bin")

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_root = Path(conda_prefix)
            dirs.append(conda_root / "Library" / "mingw-w64" / "bin")
            dirs.append(conda_root / "Library" / "bin")

        # 去重并保留原有顺序，避免重复 add_dll_directory。
        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for path in dirs:
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                seen.add(key)
                unique_dirs.append(path)
        return unique_dirs

    def _configure_signatures(self) -> None:
        """为导出函数设置 `ctypes` 参数签名。"""
        float32_1d = ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        float64_1d = ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")

        self.sregn96 = self.lib.sregn96_
        self.sregn96.argtypes = [
            float32_1d,
            float32_1d,
            float32_1d,
            float32_1d,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            ctypes.c_int,
        ]
        self.sregn96.restype = None

        self.slegn96 = self.lib.slegn96_
        self.slegn96.argtypes = [
            float32_1d,
            float32_1d,
            float32_1d,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            float64_1d,
            ctypes.c_int,
        ]
        self.slegn96.restype = None

    @staticmethod
    def model_to_thickness(model: np.ndarray) -> np.ndarray:
        """把深度节点转换为 CPS 风格层厚数组。

        输入:
            model:
                类型: `numpy.ndarray`
                形状: `(nl, 5)`
                单位: 深度 `km`
                含义: 层状模型。
        输出:
            `numpy.ndarray`
                类型: `float32`
                形状: `(nl,)`
                单位: `km`
                含义: 每层厚度，末层为半空间，厚度置 `0`。
        """
        model = np.asarray(model, dtype=np.float64)
        nl = model.shape[0]
        thk = np.zeros(nl, dtype=np.float32)
        if nl > 1:
            thk[:-1] = (model[1:, 1] - model[:-1, 1]).astype(np.float32)
        return thk

    def rayleigh_kernel(
        self,
        model: np.ndarray,
        freq: float,
        phase_velocity: float,
        iflsph: int = 0,
    ) -> PhaseVelocityKernel:
        """调用 `sregn96_` 计算 Rayleigh 波相速度敏感核。

        输入:
            model:
                类型: `numpy.ndarray`
                形状: `(nl, 5)`
                单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            freq:
                类型: `float`
                单位: `Hz`
            phase_velocity:
                类型: `float`
                单位: `km/s`
            iflsph:
                类型: `int`
                单位: 无
                含义: 是否启用球面展平，当前默认 `0`。
        输出:
            `PhaseVelocityKernel`
                `vp/vs/rho/thickness` 均为形状 `(nl,)` 的数组。
        """
        model = np.asarray(model, dtype=np.float64)
        nl = int(model.shape[0])
        thk = self.model_to_thickness(model)
        vp = np.asarray(model[:, 4], dtype=np.float32)
        vs = np.asarray(model[:, 3], dtype=np.float32)
        rho = np.asarray(model[:, 2], dtype=np.float32)
        period = ctypes.c_double(1.0 / float(freq))
        cp = ctypes.c_double(float(phase_velocity))
        cg = ctypes.c_double(float(phase_velocity))
        dispu = np.zeros(nl, dtype=np.float64)
        dispw = np.zeros(nl, dtype=np.float64)
        stressu = np.zeros(nl, dtype=np.float64)
        stressw = np.zeros(nl, dtype=np.float64)
        dc2da = np.zeros(nl, dtype=np.float64)
        dc2db = np.zeros(nl, dtype=np.float64)
        dc2dh = np.zeros(nl, dtype=np.float64)
        dc2dr = np.zeros(nl, dtype=np.float64)
        self.sregn96(
            thk,
            vp,
            vs,
            rho,
            nl,
            ctypes.byref(period),
            ctypes.byref(cp),
            ctypes.byref(cg),
            dispu,
            dispw,
            stressu,
            stressw,
            dc2da,
            dc2db,
            dc2dh,
            dc2dr,
            iflsph,
        )
        return PhaseVelocityKernel(
            vp=dc2da,
            vs=dc2db,
            rho=dc2dr,
            thickness=dc2dh,
            group_velocity=float(cg.value),
        )

    def love_kernel(
        self,
        model: np.ndarray,
        freq: float,
        phase_velocity: float,
        iflsph: int = 0,
    ) -> PhaseVelocityKernel:
        """调用 `slegn96_` 计算 Love 波相速度敏感核。

        输入:
            model:
                类型: `numpy.ndarray`
                形状: `(nl, 5)`
                单位: 深度 `km`，密度 `g/cm^3`，速度 `km/s`
            freq:
                类型: `float`
                单位: `Hz`
            phase_velocity:
                类型: `float`
                单位: `km/s`
            iflsph:
                类型: `int`
                单位: 无
                含义: 是否启用球面展平，当前默认 `0`。
        输出:
            `PhaseVelocityKernel`
                `vp/vs/rho/thickness` 均为形状 `(nl,)` 的数组。
        """
        model = np.asarray(model, dtype=np.float64)
        nl = int(model.shape[0])
        thk = self.model_to_thickness(model)
        vs = np.asarray(model[:, 3], dtype=np.float32)
        rho = np.asarray(model[:, 2], dtype=np.float32)
        period = ctypes.c_double(1.0 / float(freq))
        cp = ctypes.c_double(float(phase_velocity))
        cg = ctypes.c_double(float(phase_velocity))
        disp = np.zeros(nl, dtype=np.float64)
        stress = np.zeros(nl, dtype=np.float64)
        dc2db = np.zeros(nl, dtype=np.float64)
        dc2dh = np.zeros(nl, dtype=np.float64)
        dc2dr = np.zeros(nl, dtype=np.float64)
        self.slegn96(
            thk,
            vs,
            rho,
            nl,
            ctypes.byref(period),
            ctypes.byref(cp),
            ctypes.byref(cg),
            disp,
            stress,
            dc2db,
            dc2dh,
            dc2dr,
            iflsph,
        )
        return PhaseVelocityKernel(
            vp=np.zeros(nl, dtype=np.float64),
            vs=dc2db,
            rho=dc2dr,
            thickness=dc2dh,
            group_velocity=float(cg.value),
        )


_KERNEL_LIB: FortranKernelLibrary | None = None


def get_fortran_kernel_library() -> FortranKernelLibrary:
    """返回全局单例 Fortran 动态库包装器。

    输入:
        无。
    输出:
        `FortranKernelLibrary`
            单位: 无
            含义: 已完成动态库加载与签名配置的包装实例。
    """
    global _KERNEL_LIB
    if _KERNEL_LIB is None:
        _KERNEL_LIB = FortranKernelLibrary()
    return _KERNEL_LIB
