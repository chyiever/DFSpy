# 2026-6-4 QEDispInv 使用文档

## 1. 运行环境

建议环境：

- Windows
- Python 3.12

依赖见：

- `requirements.txt`
- `2026-6-4-QEDispInv-win需要的库列表.txt`

当前代码已经兼容较老的 Python 解释器读取 TOML，但仍建议优先使用 Python 3.12。

项目中的 `.py`、`.md`、`.ipynb` 文件均使用 `UTF-8` 编码保存。

Fortran 核函数当前通过 `build/cpskernels.dll` 提供，默认由
`ref/QEDispInv-main/fortran/sregn96.f90` 与 `slegn96.f90` 编译而来。
若需要在新环境重新编译，可运行：

```powershell
python python\build_cpskernels.py `
  --gfortran D:\anaconda3\envs\LZdataread39\Library\mingw-w64\bin\gfortran.exe
```

该脚本会同时生成：

- `build/cpskernels.dll`
- `build/cpskernels.runtime.json`
  - 含义: 记录 DLL 使用的 `gfortran` 运行时目录，供程序启动时自动加载依赖

运行时支持以下环境变量：

- `QEDISPINV_FORTRAN_DLL`
  - 含义: 自定义 `cpskernels.dll` 路径
- `QEDISPINV_MINGW_BIN`
  - 含义: MinGW/Fortran 运行时 DLL 所在目录

如果需要了解 `sregn96/slegn96` 的理论背景、DLL 构建与 Python 调用方式，可同时阅读
`2026-6-4-CPS的sregn96与slegn96说明文档.md`。

## 2. 项目结构

- `bin/`
  - 命令行主入口
- `python/`
  - 绘图和辅助工具
- `demo/`
  - 示例数据
- `notebooks/`
  - 测试 notebook

## 3. 输入文件格式

### 3.1 模型文件

文本文件共 5 列：

1. 层号，类型 `int`，无单位
2. 深度，类型 `float`，单位 `km`
3. 密度，类型 `float`，单位 `g/cm^3`
4. `Vs`，类型 `float`，单位 `km/s`
5. `Vp`，类型 `float`，单位 `km/s`

示例：

```text
1    0.00000    1.90000    0.40000    0.70000
2    0.00200    1.70000    0.20000    0.30000
```

### 3.2 色散数据文件

文本文件共 3 列或 4 列：

1. 频率，类型 `float`，单位 `Hz`
2. 相速度，类型 `float`，单位 `km/s`
3. 模态号，类型 `int`，无单位
4. 可选标准差，类型 `float`，单位 `km/s`

## 4. 配置文件格式

项目使用 TOML 配置，基本与参考项目一致。

### 4.1 `[forward]`

- `file_model`
  - 类型: `str`
  - 含义: 模型文件路径
- `fmin`
  - 类型: `float`
  - 单位: `Hz`
  - 含义: 最小频率
- `fmax`
  - 类型: `float`
  - 单位: `Hz`
  - 含义: 最大频率
- `nf`
  - 类型: `int`
  - 含义: 频率采样点数

### 4.2 `[inversion]`

- `vs2model`
  - 类型: `str`
  - 含义: `FixVpRho` / `brocher05` / `gardner` / `nearsurface`
- `model_ref`
  - 类型: `str`
  - 含义: 参考模型路径
- `vs_width`
  - 类型: `float`
  - 单位: `km/s`
  - 含义: Vs 搜索范围宽度
- `lambda`
  - 类型: `float`
  - 含义: 正则化系数
- `reg_type`
  - 类型: `int`
  - 含义: 1 为一阶差分，2 为自适应一阶差分
- `num_init`
  - 类型: `int`
  - 含义: 初始模型数
- `num_noise`
  - 类型: `int`
  - 含义: 噪声重采样次数
- `rand_depth`
  - 类型: `bool`
  - 含义: 是否随机深度分层
- `rand_vs`
  - 类型: `bool`
  - 含义: 是否随机初始化 Vs
- `zmax`
  - 类型: `float`
  - 单位: `km`
  - 含义: 统计深度上限
- `r0`, `rmin`, `rmax`
  - 类型: `float`
  - 含义: 分层比例控制参数
- `weight`
  - 类型: `list[float]`
  - 含义: 各模态权重
- `maxiter`
  - 类型: `int`
  - 含义: L-BFGS-B 最大迭代次数，Windows 快速测试可设小值

## 5. 常用命令

### 5.1 前向计算

以 `lvl-l4` 为例：

```powershell
python bin\forward.py -c demo\lvl-l4\config.toml -m 0 -o demo\lvl-l4\disp_win.txt
```

输出：

- `demo\lvl-l4\disp_win.txt`

### 5.2 按已有色散点重新计算相速度

以近地表数据为例：

```powershell
python bin\forward.py `
  -c demo\syn-nearsurface\config_win_quick.toml `
  --disp demo\syn-nearsurface\data.txt `
  -o demo\syn-nearsurface\disp_forward_win.txt
```

适用场景：

- 与已有观测频点逐点对比
- 为反演前快速检查前向精度

### 5.2.1 同时输出敏感核

```powershell
python bin\forward.py `
  -c demo\lvl-l4\config.toml `
  -m 0 `
  --compute_kernel `
  -o demo\lvl-l4\disp_win.txt
```

输出：

- `disp_win.txt`
- `kernel.npz`

### 5.3 反演

近地表快速测试：

```powershell
python bin\inversion.py `
  -c demo\syn-nearsurface\config_win_quick.toml `
  -d demo\syn-nearsurface\data.txt `
  -o demo\syn-nearsurface\inv_win_full_quick.npz
```

输出：

- `inv_win_full_quick.npz`

注意：

- 当前 Windows 版反演为纯 Python 路径。
- 建议先使用 `config_win_quick.toml` 做完整测试集的 quick 反演验证。
- 若把 `num_init`、`maxiter` 调大到接近参考项目默认值，耗时会明显增加。

### 5.4 生成参考模型

```powershell
python python\create_reference_model.py demo\syn-nearsurface\data.txt -o demo\syn-nearsurface\mref_new.txt
```

常用参数：

- `--vp2vs`
- `-s/--smooth`
- `--dmodel`
- `--zmax`

### 5.5 绘制色散曲线

```powershell
python python\plot_disp.py demo\lvl-l4\disp_win.txt
```

### 5.6 绘制反演结果

```powershell
python python\plot_inv.py demo\syn-nearsurface\inv_win_full_quick.npz --plot_model
```

或：

```powershell
python python\plot_inv.py demo\syn-nearsurface\inv_win_full_quick.npz --plot_disp
```

### 5.7 导出模型

线性模型：

```powershell
python python\print_model.py demo\syn-nearsurface\inv_win_full_quick.npz -o demo\syn-nearsurface\model_mean.txt
```

阶梯模型：

```powershell
python python\print_model.py demo\syn-nearsurface\inv_win_full_quick.npz --step -o demo\syn-nearsurface\model_step.txt
```

### 5.8 绘制模型剖面

```powershell
python python\plot_model.py demo\syn-nearsurface\model_mean.txt --linear
```

## 6. notebook 测试

项目提供：

- `notebooks/QEDispInv-win-test.ipynb`

执行命令：

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks\QEDispInv-win-test.ipynb
```

该 notebook 会完成：

- `lvl-l4` 前向测试
- `syn-nearsurface` 前向测试
- `syn-nearsurface` 完整测试集 quick 反演测试
- `syn-crustmantle` 前向测试
- `syn-crustmantle` 完整测试集 quick 反演测试

## 7. 结果文件说明

### 7.1 `disp*.txt`

3 列文本：

1. 频率 `Hz`
2. 相速度 `km/s`
3. 模态号

### 7.2 `inv*.npz`

主要键如下：

- `fitness`
  - 类型: `numpy.ndarray`
  - 单位: `km^2/s^2`
  - 含义: 每次有效反演目标函数值
- `niter`
  - 类型: `numpy.ndarray`
  - 含义: 每次反演迭代次数
- `z_sample`
  - 类型: `numpy.ndarray`
  - 单位: `km`
- `vs_sample`
  - 类型: `numpy.ndarray`
  - 单位: `km/s`
- `vs_hist2d`
  - 类型: `numpy.ndarray`
  - 含义: 深度-速度二维统计图
- `vs_mean`
  - 类型: `numpy.ndarray`
  - 单位: `km/s`
- `vs_median`
  - 类型: `numpy.ndarray`
  - 单位: `km/s`
- `vs_mode`
  - 类型: `numpy.ndarray`
  - 单位: `km/s`
- `vs_cred10`
  - 类型: `numpy.ndarray`
  - 单位: `km/s`
- `vs_cred90`
  - 类型: `numpy.ndarray`
  - 单位: `km/s`
- `model_mean`
  - 类型: `numpy.ndarray`
  - 形状: `(nl, 5)`
  - 含义: 由均值速度曲线恢复的完整模型
- `disp_syn_list`
  - 类型: `object array`
  - 含义: 每个有效反演对应的理论色散曲线

## 8. 当前限制

- `--compute_kernel` 与反演显式梯度当前已经直接调用同源 `sregn96/slegn96` Fortran DLL；如果 DLL 或其运行时依赖缺失，相关功能会直接报错。
- 完整数据、完整初值数量下的反演速度明显慢于参考项目 C++/Fortran 版本。
- 当前前向色散根搜索仍是 Python/Numba 实现，不是参考仓库原始 C++ 可执行程序。
