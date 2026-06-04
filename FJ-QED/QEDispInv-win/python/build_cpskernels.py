#!/usr/bin/env python
"""编译参考项目 Fortran 核函数为 Windows DLL。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输入:
        无。
    输出:
        `argparse.Namespace`
            单位: 无
            含义: 构建脚本参数集合。
    """
    root = Path(__file__).resolve().parents[1]
    default_fortran_dir = root.parent / "ref" / "QEDispInv-main" / "fortran"
    default_output = root / "build" / "cpskernels.dll"
    parser = argparse.ArgumentParser(description="编译 sregn96/slegn96 为 Windows DLL")
    parser.add_argument(
        "--gfortran",
        type=str,
        default="gfortran",
        help="gfortran 可执行文件路径。",
    )
    parser.add_argument(
        "--fortran-dir",
        type=str,
        default=str(default_fortran_dir),
        help="参考项目 fortran 源码目录，需包含 sregn96.f90 和 slegn96.f90。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help="输出 DLL 路径。",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="-O3",
        help="Fortran 编译优化参数。",
    )
    return parser.parse_args()


def resolve_gfortran(gfortran: str) -> str:
    """解析 `gfortran` 可执行文件路径。

    输入:
        gfortran:
            类型: `str`
            单位: 无
            含义: 用户传入的编译器命令或绝对路径。
    输出:
        `str`
            单位: 无
            含义: 实际可执行的 `gfortran` 路径。
    """
    found = shutil.which(gfortran)
    if found:
        return found
    path = Path(gfortran)
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"未找到 gfortran: {gfortran}")


def build_dll(gfortran: str, fortran_dir: Path, output: Path, opt: str) -> subprocess.CompletedProcess[str]:
    """执行 DLL 编译命令。

    输入:
        gfortran:
            类型: `str`
            单位: 无
            含义: `gfortran` 可执行路径。
        fortran_dir:
            类型: `pathlib.Path`
            单位: 无
            含义: 参考项目 Fortran 源码目录。
        output:
            类型: `pathlib.Path`
            单位: 无
            含义: 目标 DLL 输出路径。
        opt:
            类型: `str`
            单位: 无
            含义: 优化等级参数，例如 `-O3`。
    输出:
        `subprocess.CompletedProcess[str]`
            单位: 无
            含义: 编译过程返回结果。
    """
    sregn96 = fortran_dir / "sregn96.f90"
    slegn96 = fortran_dir / "slegn96.f90"
    if not sregn96.exists() or not slegn96.exists():
        raise FileNotFoundError(f"Fortran 源码目录缺少 sregn96.f90 或 slegn96.f90: {fortran_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        gfortran,
        "-shared",
        opt,
        "-static-libgfortran",
        "-static-libgcc",
        "-o",
        str(output),
        str(sregn96),
        str(slegn96),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def write_runtime_metadata(gfortran: str, output: Path) -> Path:
    """写出 DLL 运行时依赖元数据。

    输入:
        gfortran:
            类型: `str`
            单位: 无
            含义: 实际使用的 `gfortran` 可执行路径。
        output:
            类型: `pathlib.Path`
            单位: 无
            含义: 已生成的 DLL 路径。
    输出:
        `pathlib.Path`
            单位: 无
            含义: 元数据 JSON 文件路径。
    """
    metadata_path = output.with_suffix(".runtime.json")
    gfortran_path = Path(gfortran).resolve()
    metadata = {
        "dll_path": str(output.resolve()),
        "gfortran": str(gfortran_path),
        "runtime_dirs": [
            str(gfortran_path.parent),
            str(gfortran_path.parent.parent / "bin"),
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def main() -> int:
    """构建入口。

    输入:
        无。
    输出:
        `int`
            单位: 无
            含义: 进程退出码，`0` 表示成功。
    """
    args = parse_args()
    gfortran = resolve_gfortran(args.gfortran)
    result = build_dll(
        gfortran=gfortran,
        fortran_dir=Path(args.fortran_dir).resolve(),
        output=Path(args.output).resolve(),
        opt=args.opt,
    )
    print(f"gfortran: {gfortran}")
    print(f"fortran_dir: {Path(args.fortran_dir).resolve()}")
    print(f"output: {Path(args.output).resolve()}")
    if result.stdout.strip():
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr)
        return result.returncode
    metadata_path = write_runtime_metadata(gfortran, Path(args.output).resolve())
    print("cpskernels.dll 编译完成。")
    print(f"runtime metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
