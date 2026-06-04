# FJpy-win

`FJpy-win` 是参考项目 `CC-FJpy-master` 的 Windows 纯 Python 版本实现，目标是在不依赖 Cython、CUDA、FFTW 的前提下，尽量保持原项目的函数接口、数据流和结果解释方式一致，方便在 Windows 环境直接开展互相关与频散成像测试。

当前目录主要包含三部分内容：

- 核心接口文件：`ccfj.py`
- 核心算法实现：`src/ccfj_core.py`
- 示例 notebook 与测试脚本：`examples/`

## 目录说明

### 核心代码

- `ccfj.py`
  - 对外入口模块
  - 作用是保持与原项目一致的 `import ccfj` 使用方式

- `src/ccfj_core.py`
  - 互相关、F-J、F-H、MWFJ 等核心计算函数实现
  - 包含 Windows 版本的 `numba` 并行优化

### 示例与脚本

- `examples/example_CC.ipynb`
  - 三台站互相关示例
  - 输出频域互相关谱图和时域互相关剖面图

- `examples/example_noise.ipynb`
  - 环境噪声互相关结果的 F-J / F-H 成像示例

- `examples/example_EQ.ipynb`
  - 原始地震波形的多窗 F-J 成像示例

- `examples/run_cc_windows.py`
  - `example_CC.ipynb` 的脚本版

- `examples/run_fj_noise_windows.py`
  - `example_noise.ipynb` 的脚本版

- `examples/run_mwfj_windows.py`
  - `example_EQ.ipynb` 的脚本版

## 关键函数调用说明

### 1. `ccfj.GetStationPairs(nsta)`

#### 功能

根据台站数生成上三角台站对索引，顺序与原项目保持一致。

#### 使用方式

```python
pairs = ccfj.GetStationPairs(nsta)
```

#### 输入

- `nsta`
  - 类型：`int`
  - 含义：台站数量

#### 输出

- 返回值 `pairs`
  - 类型：`numpy.ndarray`
  - dtype：`int32`
  - 形状：`(2 * nPairs,)`
  - 含义：按 `[sta0, sta1, sta0, sta2, ...]` 方式展开的台站对索引

### 2. `ccfj.CC(...)`

#### 功能

对多台站波形执行频域互相关累积，输出复数互相关频谱。

#### 使用方式

```python
ncfs = ccfj.CC(
    npts,
    nsta,
    nf,
    fftlen,
    pairs,
    startend,
    data,
    overlaprate=0.5,
    nThreads=4,
    fstride=1,
    ifonebit=0,
    ifspecwhittenning=1,
)
```

#### 输入

- `npts`
  - 每个台站使用的采样点数
- `nsta`
  - 台站数
- `nf`
  - 输出频点数
- `fftlen`
  - 每段 FFT 长度
- `pairs`
  - `GetStationPairs(nsta)` 的结果
- `startend`
  - 每个台站有效时间窗的起止索引，长度为 `2 * nsta`
- `data`
  - 按台站拼接的一维波形数组，布局为 `[sta0, sta1, ...]`
- `overlaprate`
  - 窗口重叠比例
- `nThreads`
  - 为兼容原接口保留，当前 Windows 纯 Python 版不直接控制线程数
- `fstride`
  - 频点步长
- `ifonebit`
  - 是否进行 one-bit 归一化
- `ifspecwhittenning`
  - 是否进行频谱白化

#### 输出

- 返回值 `ncfs`
  - 类型：复数二维数组
  - 形状：`(nPairs, nf)`
  - 含义：每个台站对对应的复数互相关频谱

### 3. `ccfj.fj_noise(uf, r, c, f, ...)`

#### 功能

对环境噪声互相关结果执行 F-J 或 F-H 型频散成像。

#### 使用方式

```python
ds = ccfj.fj_noise(np.real(ncfs), r, c, f, fstride=1, itype=1, func=0)
```

#### 输入

- `uf`
  - 互相关频谱实部或其他待成像二维输入
  - 形状通常为 `(nPairs, nf)`
- `r`
  - 台站对距离，单位应与相速度 `c` 匹配
  - 当前示例统一使用米
- `c`
  - 相速度网格，单位 `m/s`
- `f`
  - 频率采样数组，单位 `Hz`
- `fstride`
  - 频点步长
- `itype`
  - `0` 表示梯形积分
  - `1` 表示线性近似积分
- `func`
  - `0` 表示 Bessel 分支
  - `1` 表示 Hankel 分支

#### 输出

- 返回值 `ds`
  - 类型：二维实数数组
  - 形状：`(len(c), len(f))`
  - 含义：归一化后的频散能量图

### 4. `ccfj.fj_earthquake(u, r, c, f, ...)`

#### 功能

对原始地震波形直接执行 F-J / F-H 成像。

#### 使用方式

```python
out = ccfj.fj_earthquake(u0, r, c, f, fstride=1, itype=1, func=0)
```

#### 输入

- `u`
  - 原始波形二维数组
  - 形状：`(nsta, npts)`
- `r`
  - 台站距离，单位通常为米
- `c`
  - 相速度网格，单位 `m/s`
- `f`
  - 频率数组

#### 输出

- 返回值 `out`
  - 类型：二维数组
  - 形状：`(len(c), len(f))`
  - 含义：归一化后的频散成像结果

### 5. `ccfj.MWFJ(u, r, c, f, Fs, nwin, winl, winr, ...)`

#### 功能

对原始地震波形执行多窗 F-J 成像。

#### 使用方式

```python
out = ccfj.MWFJ(u0, r, c, f, Fs, nwin, winl, winr, taper=0.9, fstride=1, itype=1, func=0)
```

#### 输入

- `u`
  - 原始波形二维数组
- `r`
  - 台站距离，单位米
- `c`
  - 相速度网格
- `f`
  - 频率数组
- `Fs`
  - 采样率
- `nwin`
  - 时间窗数量
- `winl`
  - 每个窗、每个台站的起始时间
- `winr`
  - 每个窗、每个台站的结束时间
- `taper`
  - 窗函数边缘 taper 比例
- `func`
  - `0` 表示 Bessel 组合
  - `1` 表示 Hankel 组合

#### 输出

- 返回值 `out`
  - 类型：三维数组
  - 形状：`(nwin, len(c), len(f))`
  - 含义：每个时间窗对应一幅频散成像图

## 示例运行方式

在 `FJpy-win` 目录下执行：

```bash
python examples/run_cc_windows.py
python examples/run_fj_noise_windows.py
python examples/run_mwfj_windows.py
```

## 注意事项

- notebook 与脚本中的中文内容统一通过 UTF-8 写入
- `example_noise.ipynb` 当前使用经过验证的较小规模输入，便于 Windows 版本快速复核
- `example_EQ.ipynb` 已恢复为原项目同规模参数设置
- `noise` 路径输入的是互相关结果
- `EQ` 路径输入的是原始地震波形
