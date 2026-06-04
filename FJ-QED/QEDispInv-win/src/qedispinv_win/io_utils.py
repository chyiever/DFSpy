"""I/O 与配置读取工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        import toml as tomllib  # type: ignore[no-redef]


def loadtxt(path: str | Path) -> np.ndarray:
    """读取文本数值文件。

    输入:
        path:
            类型: `str | pathlib.Path`
            单位: 无
            含义: 待读取文本文件路径。
    输出:
        `numpy.ndarray`
            类型: `float64`
            形状: `(n, m)` 或 `(n,)`
            单位: 与原文件一致
            含义: 文本中的数值矩阵。
    """
    return np.loadtxt(Path(path), dtype=np.float64)


def parse_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 配置文件。

    输入:
        path:
            类型: `str | pathlib.Path`
            单位: 无
            含义: TOML 文件路径。
    输出:
        `dict[str, Any]`
            含义: 解析后的配置字典。
    """
    path = Path(path)
    with path.open("rb") as fh:
        try:
            return tomllib.load(fh)
        except TypeError:
            return tomllib.loads(path.read_text(encoding="utf-8"))


@dataclass
class ForwardConfig:
    """前向计算配置。"""

    file_model: str
    fmin: float
    fmax: float
    nf: int


def resolve_forward_config(config: dict[str, Any]) -> ForwardConfig:
    """提取前向计算配置块。"""
    block = config["forward"]
    return ForwardConfig(
        file_model=str(block["file_model"]),
        fmin=float(block["fmin"]),
        fmax=float(block["fmax"]),
        nf=int(block["nf"]),
    )
