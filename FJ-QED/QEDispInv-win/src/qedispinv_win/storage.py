"""结果存储与兼容读取工具。

本项目使用 `npz` 代替参考项目中的 HDF5。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz_dict(path: str | Path, data: dict[str, Any]) -> None:
    """将字典保存为 `.npz` 文件。

    输入:
        path:
            类型: `str | pathlib.Path`
            单位: 无
            含义: 输出文件路径，建议后缀为 `.npz`。
        data:
            类型: `dict[str, Any]`
            含义: 需要存储的键值对。
    输出:
        无。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_npz_dict(path: str | Path) -> dict[str, Any]:
    """读取 `.npz` 文件并展开为普通字典。"""
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}

