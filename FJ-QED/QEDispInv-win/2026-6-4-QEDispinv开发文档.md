# 2026-6-4 QEDispInv 开发文档

## 1. 目标与约束

本次开发目标是在 `E:\codes\DFSpy\FJ-QED\QEDispInv-win` 下实现一个可在 Windows 中运行的 QEDispInv 版本，尽量保持与参考项目 `ref/QEDispInv-main` 一致的原理、目录组织和数据流。

开发时遇到的关键约束如下：

- 参考项目原生运行环境为 Linux，核函数链路依赖 `sregn96/slegn96` Fortran 程序。
- 当前工作目录仍以 Python 工程为主，不适合整体改造成纯 C++/Fortran 工程。
- 当前环境无 `h5py`，不适合直接复刻参考项目的 HDF5 写出链路。
- 用户要求优先复用已有 Windows 侧能力，涉及 `ccfj` 时优先调用 `FJpy-win`，避免为了迁移而引入 Linux 依赖。

因此本版采取的方案是：

- 保留与参考项目相近的目录结构：`src/`、`bin/`、`python/`、`demo/`、`doc/`、`notebooks/`。
- 用 Python 负责命令行、数据流、参数化、反演组织与结果写出。
- 把参考项目 `fortran/sregn96.f90`、`fortran/slegn96.f90` 直接编译为 Windows DLL，并通过 `ctypes` 包装调用。
- 用 `.npz` 结果文件替代参考项目中的 `.h5`，并同步改写配套绘图脚本。
- 全部新增 `.py/.md/.ipynb` 文本文件统一保存为 `UTF-8` 编码，消除乱码风险。

## 2. 目录结构

- `src/qedispinv_win/__init__.py`
  - 包入口。
- `src/qedispinv_win/io_utils.py`
  - 文本与 TOML 配置读取。
- `src/qedispinv_win/storage.py`
  - `.npz` 结果存储与读取。
- `src/qedispinv_win/secfunc.py`
  - Rayleigh / Love 波世俗函数。
- `src/qedispinv_win/fortran_kernels.py`
  - `sregn96/slegn96` 动态库包装层。
- `src/qedispinv_win/sensitivity.py`
  - 相速度敏感核计算。
- `src/qedispinv_win/dispersion.py`
  - 基于二次极值插值的色散根搜索。
- `src/qedispinv_win/modeling.py`
  - `Vs -> 模型` 参数化、深度分层、统计量计算。
- `src/qedispinv_win/inversion.py`
  - 数据集封装、多初值 L-BFGS-B 反演。
- `bin/forward.py`
  - 前向 CLI。
- `bin/inversion.py`
  - 反演 CLI。
- `python/create_reference_model.py`
  - 参考模型生成。
- `python/plot_disp.py`
  - 色散图绘制。
- `python/plot_inv.py`
  - 反演结果绘制。
- `python/print_model.py`
  - 导出反演模型。
- `python/plot_model.py`
  - 模型剖面绘制。
- `python/plot_kernel.py`
  - 核函数结果绘制。
- `demo/`
  - 从参考项目复制的 3 套 demo 数据，并补充 Windows 快速测试配置。
- `notebooks/QEDispInv-win-test.ipynb`
  - 运行测试 notebook。

## 3. 核心实现说明

### 3.1 `secfunc.py`

该模块直接按参考项目 `src/secfunc.cc` 的 PSV / SH 路径移植，使用 `numba.njit` 编译核心数值函数。

主要函数：

- `SecularFunction.__init__(model, sh=False)`
  - 输入:
    - `model`: `numpy.ndarray`，形状 `(nl, 5)`，单位分别为层号/深度 km/密度 g·cm^-3/Vs km·s^-1/Vp km·s^-1
    - `sh`: `bool`，是否计算 Love 波
  - 输出:
    - 无
  - 功能:
    - 构建世俗函数计算器，识别是否存在水层。

- `SecularFunction.evaluate(f, c)`
  - 输入:
    - `f`: `float`，单位 `Hz`
    - `c`: `float`，单位 `km/s`
  - 输出:
    - `float`，无量纲
  - 功能:
    - 计算给定频率与相速度下的世俗函数值。

### 3.2 `dispersion.py`

该模块按参考项目 `src/disp.cc` 实现：

- 近似模态计数函数
- 采样点生成
- 二次极值点插值
- 候选根区间搜索
- `scipy.optimize.toms748` 精确求根

主要函数：

- `DispersionSolver.__init__(model, sh=False)`
  - 输入:
    - `model`: `numpy.ndarray`，形状 `(nl, 5)`，单位同上
    - `sh`: `bool`
  - 输出:
    - 无
  - 功能:
    - 初始化求解器并计算高频参考速度、Rayleigh 参考速度等派生参数。

- `DispersionSolver.approx(f, c)`
  - 输入:
    - `f`: `float`，单位 `Hz`
    - `c`: `float`，单位 `km/s`
  - 输出:
    - `float`
  - 功能:
    - 给出根数的近似累积指标，用于生成初始采样。

- `DispersionSolver.get_samples(f)`
  - 输入:
    - `f`: `float`，单位 `Hz`
  - 输出:
    - `list[float]`，单位 `km/s`
  - 功能:
    - 为该频率生成初始相速度采样点。

- `DispersionSolver.locate_extremum(f, x, y)`
  - 输入:
    - `f`: `float`，单位 `Hz`
    - `x`: `list[float]`，单位 `km/s`
    - `y`: `list[float]`，无量纲
  - 输出:
    - `tuple[list[float], list[float]]`
  - 功能:
    - 用二次插值找采样点之间的局部极值，提高 mode-kissing 区域的根检出率。

- `DispersionSolver.search(f, num_mode)`
  - 输入:
    - `f`: `float`，单位 `Hz`
    - `num_mode`: `int`
  - 输出:
    - `list[float]`，单位 `km/s`
  - 功能:
    - 返回从基阶开始的前 `num_mode` 个模态相速度。

- `DispersionSolver.search_mode(f, mode)`
  - 输入:
    - `f`: `float`，单位 `Hz`
    - `mode`: `int`
  - 输出:
    - `float`，单位 `km/s`
  - 功能:
    - 返回指定模态相速度；缺失时返回 `nan`。

### 3.3 `modeling.py`

该模块对应参考项目 `src/model.cc` 与 `src/problem.cc` 中的部分统计逻辑。

主要类与函数：

- `FixVpRhoConverter.generate(z, vs)`
- `Brocher05Converter.generate(z, vs)`
- `GardnerConverter.generate(z, vs)`
- `NearSurfaceConverter.generate(z, vs)`

这些函数的共同约定如下：

- 输入:
  - `z`: `numpy.ndarray`，形状 `(nl,)`，单位 `km`
  - `vs`: `numpy.ndarray`，形状 `(nl,)`，单位 `km/s`
- 输出:
  - `numpy.ndarray`，形状 `(nl, 5)`，列为层号/深度/密度/Vs/Vp
- 功能:
  - 根据不同 `vs2model` 关系，把反演变量 `Vs` 恢复为完整层状模型。

其他函数：

- `generate_depth_by_layer_ratio(lmin, lmax, r0, rmin, rmax, zmax, rng=None)`
  - 输入:
    - `lmin`, `lmax`, `zmax`: `float`，单位 `km`
    - `r0`, `rmin`, `rmax`: `float`，无量纲
  - 输出:
    - `numpy.ndarray`，形状 `(nl,)`，单位 `km`
  - 功能:
    - 按参考项目的比例分层方案生成随机深度节点。

- `compute_hist2d(z_inv, vs_inv, fitness, vsmin, vsmax, zmax, num_hist)`
  - 输入:
    - `z_inv`: `list[numpy.ndarray]`，单位 `km`
    - `vs_inv`: `list[numpy.ndarray]`，单位 `km/s`
    - `fitness`: `numpy.ndarray`
  - 输出:
    - `tuple[z_sample, vs_sample, hist2d]`
  - 功能:
    - 将反演结果聚合为深度-速度二维统计图。

- `compute_statistics(z, vs, hist)`
  - 输入:
    - `z`: `numpy.ndarray`，单位 `km`
    - `vs`: `numpy.ndarray`，单位 `km/s`
    - `hist`: `numpy.ndarray`
  - 输出:
    - `StatisticsResult`
  - 功能:
    - 计算均值、中位数、众数、P10、P90。

### 3.4 `inversion.py`

该模块对应参考项目 `src/main_inversion.cc` 与 `src/problem.cc`。

主要类与函数：

- `DataSet(raw)`
  - 输入:
    - `raw`: `numpy.ndarray`，形状 `(n, 3)` 或 `(n, 4)`
    - 第 1 列频率，单位 `Hz`
    - 第 2 列相速度，单位 `km/s`
    - 第 3 列模态号
    - 第 4 列可选标准差，单位 `km/s`
  - 输出:
    - `DataSet`
  - 功能:
    - 将观测色散数据按模态整理，并计算 `lmin/lmax/cmin/cmax`。

- `DataSet.add_sigma(sigma_by_mode)`
  - 输入:
    - `sigma_by_mode`: `list[float]`，单位 `km/s`
  - 输出:
    - 无
  - 功能:
    - 对无误差列数据追加按模态常数标准差。

- `DataSet.resample(rng)`
  - 输入:
    - `rng`: `numpy.random.Generator`
  - 输出:
    - `DataSet`
  - 功能:
    - 按已有标准差进行噪声重采样。

- `InversionRunner(model_ref, data, config, sh=False, seed=20260604)`
  - 输入:
    - `model_ref`: `numpy.ndarray`，形状 `(nl, 5)`，单位同模型文件
    - `data`: `DataSet`
    - `config`: `InversionConfig`
    - `sh`: `bool`
  - 输出:
    - `InversionRunner`

- `InversionRunner.run()`
  - 输出:
    - `dict[str, numpy.ndarray | object]`
  - 功能:
    - 执行多初值反演、异常值剔除、统计结果汇总，并返回可直接写入 `.npz` 的结果字典。

当前实现中，目标函数采用 `scipy.optimize.minimize(method="L-BFGS-B")`，
梯度不再依赖 SciPy 的有限差分，而是基于 `sregn96/slegn96` 输出的相速度敏感核与 `Vs2Model` 链式导数显式构建。

### 3.5 `sensitivity.py`

该模块现在只保留“统一入口”职责，真正的核函数计算已经切换到 `fortran_kernels.py`。

当前实现方式如下：

- `compute_phase_velocity_kernel(...)` 根据 `sh` 参数选择 Rayleigh 或 Love 波。
- Rayleigh 波调用 `FortranKernelLibrary.rayleigh_kernel(...)`，底层直接对应 `sregn96_`。
- Love 波调用 `FortranKernelLibrary.love_kernel(...)`，底层直接对应 `slegn96_`。
- Fortran 输出的 `dc2da / dc2db / dc2dr / dc2dh` 分别映射为 `vp / vs / rho / thickness` 核数组。
- 同时返回 Fortran 更新后的群速度 `group_velocity`。

主要函数：

- `compute_phase_velocity_kernel(model, freq, phase_velocity, sh=False, rel_step=1.0e-5)`
  - 输入:
    - `model`: `numpy.ndarray`，形状 `(nl, 5)`，深度 `km`，密度 `g/cm^3`，速度 `km/s`
    - `freq`: `float`，单位 `Hz`
    - `phase_velocity`: `float`，单位 `km/s`
    - `sh`: `bool`
  - 输出:
    - `PhaseVelocityKernel`
      - `vp`: 形状 `(nl,)`
      - `vs`: 形状 `(nl,)`
      - `rho`: 形状 `(nl,)`
  - 功能:
    - 计算单个频率-模态点的相速度敏感核；当前底层直接调用同源 Fortran DLL。

### 3.6 `fortran_kernels.py`

该模块负责把参考项目 Fortran 核函数接入 Windows 版主流程。

主要类与函数：

- `PhaseVelocityKernel`
  - 核函数数据结构，字段为 `vp`、`vs`、`rho`、`thickness`、`group_velocity`
- `FortranKernelLibrary.resolve_dll_path(dll_path)`
  - 解析 DLL 路径；优先级为显式参数、环境变量 `QEDISPINV_FORTRAN_DLL`、项目默认 `build/cpskernels.dll`
- `FortranKernelLibrary.runtime_search_dirs()`
  - 自动收集 `QEDISPINV_MINGW_BIN`、当前 Python/Conda 环境下常见的 MinGW 运行时目录
- `FortranKernelLibrary.rayleigh_kernel(model, freq, phase_velocity, iflsph=0)`
  - 调用 `sregn96_`
- `FortranKernelLibrary.love_kernel(model, freq, phase_velocity, iflsph=0)`
  - 调用 `slegn96_`
- `get_fortran_kernel_library()`
  - 返回全局单例，避免重复加载 DLL

模型传递时，`model_to_thickness(...)` 会把内部深度节点形式 `(nl, 5)` 转为 CPS 所需的层厚数组，其中最后一层厚度置为 `0` 表示半空间。
### 3.7 `python/build_cpskernels.py`

该脚本用于在新的 Windows 环境中重新编译 Fortran 动态库。

主要参数：

- `--gfortran`
  - 类型: `str`
  - 含义: `gfortran` 可执行路径或命令名
- `--fortran-dir`
  - 类型: `str`
  - 含义: 参考项目 `fortran/` 源码目录
- `--output`
  - 类型: `str`
  - 含义: 输出 DLL 路径，默认 `build/cpskernels.dll`

当前默认编译命令等价于：

```powershell
gfortran -shared -O3 -static-libgfortran -static-libgcc `
  -o build\cpskernels.dll `
  ref\QEDispInv-main\fortran\sregn96.f90 `
  ref\QEDispInv-main\fortran\slegn96.f90
```

编译完成后，脚本还会写出 `build/cpskernels.runtime.json`，用于记录 DLL 对应的 `gfortran` 与运行时库目录，便于后续在不同 Python/Conda 环境中自动补齐 `os.add_dll_directory(...)`。

### 3.8 CLI 与绘图脚本

- `bin/forward.py`
  - 输入:
    - `config.toml`
    - 模型文件
    - 可选目标色散文件
  - 输出:
    - 文本色散文件 `disp*.txt`
    - 可选 `kernel.npz`

- `bin/inversion.py`
  - 输入:
    - `config.toml`
    - 观测色散数据
    - 参考模型
  - 输出:
    - 反演结果 `inv*.npz`

- `python/create_reference_model.py`
  - 功能:
    - 从基阶色散数据估计参考模型。

- `python/plot_inv.py`
  - 功能:
    - 读取 `inv*.npz` 绘制模型统计图、色散拟合图、目标函数图。

- `python/print_model.py`
  - 功能:
    - 导出均值模型，可切换线性/阶梯式。

## 4. 与参考项目的主要差异

### 4.1 已对齐部分

- 目录组织尽量接近参考项目。
- 计算原理保持一致：
  - 世俗函数
  - 二次极值插值
  - `toms748` 根搜索
  - `vs2model` 经验关系
  - 分层比例策略
  - 多初值反演与统计输出

### 4.2 当前未完全等价部分

- 参考项目使用 C++ + Fortran；当前 Windows 版使用 Python 负责调度，并通过 `ctypes + cpskernels.dll` 直接调用同源 `sregn96/slegn96`。
- 当前 `--compute_kernel` 与反演显式梯度已经改为 Fortran 同源核，但前向色散根搜索仍是 Python/Numba 实现，不是参考仓库的原始 C++ 二进制。
- 参考项目输出 HDF5；当前版本输出 `.npz`。
- 当前版本已恢复显式梯度链路，但完整数据集反演速度仍慢于参考项目的 C++/Fortran 版本。

### 4.3 前向色散根搜索差异说明

“前向色散根搜索仍是 Python/Numba 实现”这件事，影响并不只是在计算效率上。

从算法层面看，当前 Windows 版仍然保持了与参考项目一致的核心思路：都基于同一套世俗函数表达式、局部二次极值插值补点策略，以及 `toms748` 类型的区间求根方法。因此，在普通模型、普通频段、模态分支分离较好的情况下，Python 版与原始 C++ 版的相速度结果通常应当非常接近，差异主要体现在浮点舍入误差量级。

但在数值实现层面，二者仍然存在几类可能影响最终结果的差异。第一类差异来自浮点运算顺序。C++ 版通过 `Eigen` 和编译器优化完成矩阵与标量运算，Python 版则通过 `NumPy/Numba` 执行相同公式；即使公式一致，乘加顺序、临时变量提升、平方根与绝对值函数的底层实现也可能不同，这会导致世俗函数在接近零点时出现极小偏差。第二类差异来自根区间管理与异常处理。原始 C++ 代码中的采样点生成、极值点插入顺序、候选区间去重和边界裁剪逻辑，是和它自己的容器、排序与阈值判断耦合在一起的；Python 版虽然仿照实现，但在列表拼接、排序稳定性、`nan` 传播方式和 SciPy 失败回退路径上仍可能略有不同。第三类差异集中出现在低速层、mode-kissing、水层或频散分支贴近的区段，这些场景对“有没有恰好补到临界采样点”极为敏感，因此即便只是一点点数值偏差，也可能造成“某一支模态被更早识别”或者“某个频点恰好多/少一个根”。

因此，当前这部分实现的结论应当是：在大多数正常场景下，它与参考仓库的物理原理和数值流程高度一致，结果通常不会出现系统性偏差；但在最敏感的根搜索边界场景下，仍不能宣称与原始 C++ 可执行程序逐点完全等价。这也是为什么本次工作优先恢复了同源 Fortran 核函数，而把前向 C++ 根搜索保留为下一阶段可继续收敛的对象。

## 5. 测试结果

### 5.1 已执行测试

使用相同 demo 数据目录完成以下测试：

- `demo/lvl-l4`
  - 前向计算完成，输出 `disp_win.txt`
  - 输出行数: `1000`
  - 已完成 `--compute_kernel`，输出 `kernel.npz`

- `demo/syn-nearsurface`
  - 使用完整 `data.txt` 完成前向匹配计算，输出 `disp_forward_win.txt`
  - 输出行数: `80`
  - 使用同一测试集 `data.txt` 完成 Windows quick 反演，输出 `inv_win_full_quick.npz`
  - 有效反演数量: `1`
  - 最小目标函数值: `0.004769627469602607`

- `demo/syn-crustmantle`
  - 使用完整 `data.txt` 完成前向匹配计算，输出 `disp_forward_win.txt`
  - 输出行数: `125`
  - 使用同一测试集 `data.txt` 完成 Windows quick 反演，输出 `inv_win_full_quick.npz`
  - 有效反演数量: `1`
  - 最小目标函数值: `0.009462248409277587`

### 5.2 notebook 验证

已执行:

- `notebooks/QEDispInv-win-test.ipynb`

执行方式：

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks\QEDispInv-win-test.ipynb
```

该 notebook 已成功执行并回写结果。

## 6. 后续可继续完善的点

- 可继续把参考项目其余 C++/Fortran 组件逐步包装到 Windows 入口中，进一步缩小与原始 Linux 版本的实现差异。
- 可继续优化完整数据、完整初值场景下的性能，例如减少重复根搜索、缓存核函数与并行多初值任务。
- 可增加更多绘图参数，与参考项目 `python/` 目录对齐得更彻底。

## 7. 本次补充文档与清单

为便于后续维护与迁移，本次除主开发文档和使用文档外，还额外补充了以下文件：

- `2026-6-4-CPS的sregn96与slegn96说明文档.md`
  - 详细说明 CPS Fortran 核函数的理论背景、输入输出、DLL 编译方法和 Python 调用方式。
- `2026-6-4-QEDispInv-win需要的库列表.txt`
  - 汇总当前已验证的 Python 依赖、notebook 依赖、Fortran 编译依赖和运行时依赖。

这些文件的目的不是重复已有 README，而是把“为什么这样实现、如何在新环境里复现”单独沉淀下来，减少后续排查编码、DLL 与依赖路径问题的时间成本。
