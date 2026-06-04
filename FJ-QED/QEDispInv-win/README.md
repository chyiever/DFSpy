# QEDispInv-win

`QEDispInv-win` 是参考 `QEDispInv-main` 开发的 Windows 版实现。

本项目保持与参考仓库接近的目录组织方式：

- `src/qedispinv_win/`：核心计算与 I/O 模块
- `bin/`：命令行入口
- `python/`：绘图与辅助脚本
- `demo/`：与参考项目对应的测试目录
- `doc/`：使用说明与开发说明
- `notebooks/`：测试 notebook

项目中的 `.py`、`.md`、`.ipynb` 文本文件均按 `UTF-8` 编码保存。

本项目当前采用“Python 主程序 + NumPy/SciPy/Numba 调度层 + 同源 Fortran 核函数 DLL”的方式重建参考项目功能。
其中参考项目 `fortran/sregn96.f90` 与 `fortran/slegn96.f90` 已在 Windows 下编译为 `build/cpskernels.dll`，
并通过 `ctypes` 直接接入 `--compute_kernel` 和反演梯度链路。涉及 Linux 侧 `ccfj` 能力时，优先复用 `FJpy-win`。

## 当前实现范围

- Rayleigh / Love 波色散曲线前向计算
- 基于二次极值插值的根区间搜索
- 参考模型生成
- 多初值 L-BFGS-B 反演
- 基于 `sregn96/slegn96` 同源 Fortran DLL 的相速度敏感核计算
- 结果存储、绘图与模型导出
- 使用参考项目相同 demo 数据集的 Windows 测试 notebook

## Fortran 核构建

若需要在新环境中重新编译核函数 DLL，可运行：

```powershell
python python\build_cpskernels.py `
  --gfortran D:\anaconda3\envs\LZdataread39\Library\mingw-w64\bin\gfortran.exe
```

默认输出为 `build/cpskernels.dll`。运行时也支持通过环境变量覆盖：

构建脚本会同时写出 `build/cpskernels.runtime.json`，用于记录 `gfortran` 运行时库目录，便于后续在其他 Python 环境中自动定位依赖。

- `QEDISPINV_FORTRAN_DLL`
  - 指定自定义 DLL 路径
- `QEDISPINV_MINGW_BIN`
  - 指定 MinGW/Fortran 运行时 DLL 所在目录

## 环境

建议使用 Python 3.12，依赖见 `requirements.txt`。
