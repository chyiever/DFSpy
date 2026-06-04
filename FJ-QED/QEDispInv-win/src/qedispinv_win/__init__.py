"""QEDispInv Windows 版核心包。"""

from .dispersion import DispersionSolver
from .inversion import InversionConfig, InversionRunner, DataSet
from .modeling import (
    Brocher05Converter,
    FixVpRhoConverter,
    GardnerConverter,
    NearSurfaceConverter,
    generate_depth_by_layer_ratio,
)
from .secfunc import SecularFunction
from .sensitivity import PhaseVelocityKernel, compute_phase_velocity_kernel

__all__ = [
    "Brocher05Converter",
    "DataSet",
    "DispersionSolver",
    "FixVpRhoConverter",
    "GardnerConverter",
    "InversionConfig",
    "InversionRunner",
    "NearSurfaceConverter",
    "PhaseVelocityKernel",
    "SecularFunction",
    "compute_phase_velocity_kernel",
    "generate_depth_by_layer_ratio",
]
