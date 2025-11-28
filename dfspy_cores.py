"""
DFSPy 核心功能模块

本模块从原 GUI 子模块提取出以下功能，供脚本/Notebook 直接调用：
- 数据读取：txt 数组读取、ObsPy Stream 读取
- 绘图：二维数组诸道绘图、ObsPy Stream 诸道绘图
- 格式转换：txt <-> SAC/MSEED/SEGY（当前实现 txt->SAC/MSEED 以及 SAC/MSEED 互转；SEGY 预留）
- 降噪：带通滤波（bandpass）
- 参量转换：应变 -> 速度（按原程序约定）
- 压缩/解压：
    - 小波系数压缩（FWT，适用于 txt 数据），以及解压重构
    - GZip 通用文件压缩/解压（附加）

重要约定与注意事项：
1) 所有“会产生文件输出”的函数，默认会在“输入文件所在目录”下新建独立的输出子目录，例如：
   <输入文件目录>/DFSPy_<任务名>_outputs/...
   以避免污染项目根目录。
2) 示例头文件 exampledata/headfile.txt 的格式为 key: value 的逐行配置；
   本模块提供 read_headfile() 解析工具，并在 txt -> SAC/MSEED 转换时使用。
3) 函数均提供中文文档与参数说明，并在常见错误时抛出带中文信息的异常。

依赖：numpy、matplotlib、obspy、pywt（可选，仅 FWT 压缩需要）


**************************************************************************************
软  件  名  称 : 分布式光纤传感数据处理软件 [简称： DFSPy] V1.0
著  作  权  人 : 中国科学院半导体研究所
软件著作权登记号: 2025SR0353448
联  系  邮  箱 : qi.gh@outlook.com
开 发 者 主 页 : https://github.com/chyiever
开  发  语  言 : Python 3.9+
软  件  简  介：DFSPy 是一款专为科研人员设计的分布式光纤传感数据处理专业工具，致力于提供高效、
可靠的数据处理解决方案。该软件集成了多种先进的数据处理算法，支持分布式光纤传感系统采集数据的
预处理、分析与可视化，助力科研人员深入挖掘数据价值，加速研究进程。
使  用  声  明：
  1. 本软件为中国科学院半导体研究所开发的开源科研工具，仅供学术研究与非商业用途。
  2. 用户使用本软件时，须遵守国家相关法律法规及科研道德规范，不得用于任何商业活动或非法用途。
  3. 软件以 "现状" 提供，开发者不对其适用性、完整性或准确性做出任何明示或暗示的保证。
  4. 用户应自行评估并承担使用本软件所产生的一切风险，开发者不对因使用本软件而导致的任何直接或
     间接损失承担责任。
  5. 如需引用本软件进行学术成果发表，请注明软件来源及开发者信息。
***************************************************************************************
"""

from __future__ import annotations

import os
import re
import gzip
import json
import math
import pickle
import shutil
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Iterable

import numpy as np

# 可选依赖：绘图与地震数据格式处理
try:
    import matplotlib.pyplot as plt
except Exception as _:
    plt = None  # 允许在无 matplotlib 环境下导入

try:
    from obspy import read as obspy_read
    from obspy import Stream, Trace, UTCDateTime
except Exception as _:
    obspy_read = None
    Stream = None
    Trace = None
    UTCDateTime = None

try:
    import pywt
except Exception as _:
    pywt = None  # 仅 FWT 压缩需要


# ----------------------------- 基础设施与工具函数 -----------------------------

def _ensure_output_dir(input_path: str, task_name: str) -> str:
    """
    基于输入路径创建/获取任务专属输出目录，避免层层嵌套。

    规则：
    - 如果输入位于某个 DFSPy_*_outputs 目录内，则向上回溯直到离开该输出目录，
      在“首个非 DFSPy_*_outputs 目录”下创建/复用 DFSPy_<task>_outputs。
    - 否则，直接在输入文件所在目录下创建 DFSPy_<task>_outputs。

    参数
    - input_path: 输入文件或目录（绝对或相对）
    - task_name: 任务名（如 'format', 'denoise', 'paraconv', 'compress', 'decompress'）

    返回
    - 输出目录绝对路径（若不存在则新建）
    """
    if not input_path:
        raise ValueError("input_path 不能为空")
    abs_in = os.path.abspath(input_path)
    base_dir = abs_in if os.path.isdir(abs_in) else os.path.dirname(abs_in)

    # 若当前位于 DFSPy_*_outputs 中，则回溯到其上一层，避免嵌套
    def is_dfs_outputs(dirname: str) -> bool:
        name = os.path.basename(dirname)
        return bool(re.match(r'^DFSPy_.*_outputs$', name))

    parent = base_dir
    while parent and is_dfs_outputs(parent):
        parent = os.path.dirname(parent)
    if not parent:
        parent = base_dir  # 兜底，理论上不会发生

    out_dir = os.path.join(parent, f"DFSPy_{task_name}_outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def read_headfile(headfile_path: str) -> Dict[str, str]:
    """
    读取头文件（key: value 逐行），返回字典。

    参数
    - headfile_path: 头文件路径

    返回
    - dict，键值均为 str。常见键示例：starttime、delta、network、station。

    可能异常
    - FileNotFoundError: 文件不存在
    - ValueError: 文件为空或解析失败
    """
    if not os.path.isfile(headfile_path):
        raise FileNotFoundError(f"未找到头文件: {headfile_path}")
    data: Dict[str, str] = {}
    with open(headfile_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' not in line:
                # 允许使用 等号 作为分隔，也兼容 key value
                if '=' in line:
                    k, v = line.split('=', 1)
                else:
                    parts = re.split(r"\s+", line, maxsplit=1)
                    if len(parts) != 2:
                        continue
                    k, v = parts
            else:
                k, v = line.split(':', 1)
            data[k.strip()] = v.strip()
    if not data:
        raise ValueError(f"头文件为空或格式不正确: {headfile_path}")
    return data


def read_txt_array(txt_path: str) -> np.ndarray:
    """
    读取 txt 文本中的二维数据为 ndarray。

    要求：
    - 每列代表一条道（trace），每行代表一个采样点（与原 GUI 一致）。

    参数
    - txt_path: 文本数据路径

    返回
    - ndarray，形状为 (n_samples, n_traces)

    可能异常
    - FileNotFoundError: 文件不存在
    - ValueError: 数据为空或维度异常
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"未找到数据文件: {txt_path}")
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("txt 数据必须是二维矩阵 (samples x traces)")
    return arr


def read_stream(file_path: str):
    """
    使用 ObsPy 读取地震格式文件，返回 Stream。

    参数
    - file_path: 输入文件路径（SAC/MSEED/SEED/SEGY 等）

    返回
    - obspy.Stream 对象

    可能异常
    - ImportError: 未安装 obspy
    - FileNotFoundError: 文件不存在
    - Exception: ObsPy 读取失败
    """
    if obspy_read is None:
        raise ImportError("需要安装 obspy 才能读取地震数据格式。pip install obspy")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"未找到数据文件: {file_path}")
    return obspy_read(file_path)


def array_to_stream_from_head(data: np.ndarray, head: Dict[str, str]):
    """
    根据头文件信息，将二维数组 data 转为 ObsPy Stream（逐列为一条道）。

    头文件常用字段：
    - starttime: 起始时间（可解析为 UTCDateTime 的字符串，如 "2023-08-08T01:24:14.732"）
    - delta: 采样间隔（秒）或采样率（Hz），优先解析为采样间隔
    - network, station: 网络和台站代码（可选）

    参数
    - data: ndarray，形状 (n_samples, n_traces)
    - head: 头字段字典

    返回
    - obspy.Stream
    """
    if Stream is None or Trace is None:
        raise ImportError("需要安装 obspy 才能构建 Stream。pip install obspy")
    if data.ndim != 2:
        raise ValueError("data 必须是二维 (samples x traces)")

    st = Stream()
    # 解析 starttime
    stime_raw = head.get('starttime') or head.get('start_time') or head.get('StartTime')
    if stime_raw is None:
        # 没有 starttime，允许为空，但建议告知
        start_time = None
    else:
        if UTCDateTime is None:
            raise ImportError("需要安装 obspy 以解析 starttime 为 UTCDateTime")
        start_time = UTCDateTime(str(stime_raw))

    # 解析 delta：优先 delta（秒），若仅给出 samplerate 则转换
    delta = None
    if 'delta' in head:
        try:
            delta = float(head['delta'])
        except Exception:
            raise ValueError(f"头文件 delta 无法解析为浮点数: {head.get('delta')}")
    elif 'samplerate' in head:
        try:
            fs = float(head['samplerate'])
            delta = 1.0 / fs if fs > 0 else None
        except Exception:
            raise ValueError(f"头文件 samplerate 无法解析为浮点数: {head.get('samplerate')}")

    network = head.get('network', '')
    station = head.get('station', '')

    n_samples, n_traces = data.shape
    for i in range(n_traces):
        tr = Trace()
        tr.data = data[:, i].astype(np.float32, copy=False)
        if start_time is not None:
            tr.stats.starttime = start_time
        if delta is not None:
            tr.stats.delta = float(delta)
        if network:
            tr.stats.network = network
        if station:
            tr.stats.station = station
        tr.stats.channel = f"{i + 1:02d}"
        tr.stats.npts = n_samples
        st += tr
    return st


# ----------------------------- 绘图 -----------------------------

def plot_array(data: np.ndarray, title: str = None, ax=None):
    """
    诸道二维数组绘图（按原 GUI 习惯：列为道，行为采样），x 轴为道序，y 轴为样点并倒轴。

    参数
    - data: ndarray，形状 (n_samples, n_traces)
    - title: 图标题
    - ax: 可选，matplotlib Axes

    返回
    - ax 对象（便于在 Notebook 中继续美化）

    可能异常
    - ImportError: 未安装 matplotlib
    - ValueError: 维度不符
    """
    if plt is None:
        raise ImportError("需要安装 matplotlib 以使用绘图。pip install matplotlib")
    if data.ndim != 2:
        raise ValueError("data 必须是二维 (samples x traces)")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.cla()
    n_samples, n_traces = data.shape
    for i in range(n_traces):
        y = data[:, i]
        denom = np.max(np.abs(y)) or 1.0
        y = y / denom + i + 1
        x = np.arange(len(y))
        ax.plot(y, x, lw=0.8)
    ax.set_xlabel('Trace')
    ax.set_ylabel('Samples')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    return ax


def plot_stream(st, title: str = None, ax=None):
    """
    诸道 ObsPy Stream 绘图（与 GUI 一致：x 为道序，y 为样点并倒轴）。

    参数
    - st: obspy.Stream 或兼容对象（包含若干 Trace）
    - title: 图标题
    - ax: 可选，matplotlib Axes

    返回
    - ax 对象
    """
    if plt is None:
        raise ImportError("需要安装 matplotlib 以使用绘图。pip install matplotlib")
    if Stream is None:
        raise ImportError("需要安装 obspy 以处理 Stream。pip install obspy")
    if not isinstance(st, Stream):
        raise TypeError("st 必须是 obspy.Stream")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.cla()
    for i in range(len(st)):
        tr = st.select(channel=f"{i + 1:02d}")[0] if st.select(channel=f"{i + 1:02d}") else st[i]
        y = tr.data
        denom = np.max(np.abs(y)) or 1.0
        y = y / denom + i + 1
        x = np.arange(len(y))
        ax.plot(y, x, lw=0.8)
    ax.set_xlabel('Trace')
    ax.set_ylabel('Samples')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    return ax


# ----------------------------- 格式转换 -----------------------------

def convert_format(input_path: str, output_format: str, headfile_path: Optional[str] = None) -> str:
    """
    通用格式转换：
    - txt -> SAC/MSEED：需要头文件以构造时间与采样信息
    - SAC/MSEED -> 互转 或 -> txt

    参数
    - input_path: 输入文件
    - output_format: 目标格式（不区分大小写，支持 'SAC'/'MSEED'/'TXT'）
    - headfile_path: 当输入为 txt 且目标为 SAC/MSEED 时必需

    返回
    - 输出文件路径

    可能异常
    - ImportError/ValueError/FileNotFoundError
    """
    if output_format is None:
        raise ValueError("缺少目标格式 output_format")
    output_format = output_format.strip().upper()
    if output_format not in {"SAC", "MSEED", "TXT"}:
        raise ValueError("当前仅支持输出为 SAC/MSEED/TXT")

    ext = os.path.splitext(input_path)[1].lower()
    out_dir = _ensure_output_dir(input_path, 'format')

    if ext == '.txt':
        if output_format == 'TXT':
            raise ValueError("输入已是 TXT，无需转换")
        if headfile_path is None:
            raise ValueError("txt 转为 SAC/MSEED 时需要提供 headfile_path")
        head = read_headfile(headfile_path)
        data = read_txt_array(input_path)
        st = array_to_stream_from_head(data, head)
        base = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(out_dir, f"{base}.{output_format.lower()}")
        st.write(out_path, format=output_format)
        return out_path

    # ObsPy 可读的格式（SAC/MSEED 等）
    if obspy_read is None:
        raise ImportError("需要安装 obspy 才能读取/写入地震格式。pip install obspy")
    st = read_stream(input_path)

    base = os.path.splitext(os.path.basename(input_path))[0]
    if output_format == 'TXT':
        # 导出为 txt（按列为道）
        # 将各道长度对齐（取最短长度）
        min_len = min(int(tr.stats.npts) for tr in st)
        data = np.column_stack([tr.data[:min_len] for tr in st]).astype(np.float32)
        out_path = os.path.join(out_dir, f"{base}.txt")
        np.savetxt(out_path, data, fmt='%.7e')
        return out_path

    # SAC <-> MSEED 等互转
    out_path = os.path.join(out_dir, f"{base}.{output_format.lower()}")
    st.write(out_path, format=output_format)
    return out_path


# ----------------------------- 降噪 -----------------------------

def bandpass_denoise(input_path_or_stream: Union[str, object], freqmin: float, freqmax: float) -> str:
    """
    带通滤波降噪（对 ObsPy Stream）：

    参数
    - input_path_or_stream: 输入文件路径（SAC/MSEED 等）或 Stream
    - freqmin, freqmax: 带通频带（Hz）

    返回
    - 输出文件路径（与输入同目录的 DFSPy_denoise_outputs 子目录）

    可能异常
    - ImportError/ValueError/FileNotFoundError
    """
    if obspy_read is None:
        raise ImportError("需要安装 obspy 才能进行滤波。pip install obspy")
    if not (freqmin > 0 and freqmax > 0 and freqmax > freqmin):
        raise ValueError("freqmin/freqmax 必须为正，且 freqmax > freqmin")

    if isinstance(input_path_or_stream, str):
        st = read_stream(input_path_or_stream)
        in_path = input_path_or_stream
    elif isinstance(input_path_or_stream, Stream):
        st = input_path_or_stream
        in_path = st[0].stats.get('source', 'stream') if len(st) > 0 else 'stream'
    else:
        raise TypeError("input_path_or_stream 必须是 str 或 Stream")

    st_f = st.copy()
    st_f.filter('bandpass', freqmin=freqmin, freqmax=freqmax)

    out_dir = _ensure_output_dir(in_path, 'denoise')
    # 继承原扩展名
    in_base = os.path.basename(in_path)
    base, ext = os.path.splitext(in_base)
    if ext:
        out_path = os.path.join(out_dir, f"{base}_Bandpass{ext}")
    else:
        out_path = os.path.join(out_dir, f"{base}_Bandpass.mseed")
    # 猜测格式
    fmt = ext[1:].upper() if ext else 'MSEED'
    st_f.write(out_path, format=fmt)
    return out_path


# ----------------------------- 参量转换 -----------------------------

def filter_stream(
    input_path_or_stream: Union[str, object],
    mode: str = 'bandpass',
    kind: str = 'butterworth',
    freqs: Optional[Tuple[float, float]] = None,
    order: int = 4,
    zerophase: bool = True,
) -> str:
    """
    通用滤波器封装，支持：
    - mode: 'bandpass' | 'lowpass' | 'highpass' | 'bandstop'
    - kind: 'butterworth'（当前走 ObsPy 内置 SciPy IIR）| 'cheby1'（占位，后续扩展）
    - freqs: 频率参数（Hz）。
        - bandpass/bandstop: (fmin, fmax)
        - lowpass/highpass: (fcut, None)
    - order: 滤波器阶次
    - zerophase: 是否零相位滤波（filtfilt）

    返回输出文件路径，位于非嵌套的 DFSPy_denoise_outputs 目录。
    """
    if obspy_read is None:
        raise ImportError("需要安装 obspy 才能进行滤波。pip install obspy")

    valid_modes = {"bandpass", "lowpass", "highpass", "bandstop"}
    mode = mode.lower()
    if mode not in valid_modes:
        raise ValueError(f"不支持的滤波类型 mode: {mode}")

    kind = (kind or 'butterworth').lower()
    if kind not in {"butterworth", "cheby1"}:
        raise ValueError(f"不支持的滤波器：{kind}")

    if isinstance(input_path_or_stream, str):
        st = read_stream(input_path_or_stream)
        in_path = input_path_or_stream
    elif isinstance(input_path_or_stream, Stream):
        st = input_path_or_stream
        in_path = st[0].stats.get('source', 'stream') if len(st) > 0 else 'stream'
    else:
        raise TypeError("input_path_or_stream 必须是 str 或 Stream")

    # 解析频率参数
    if mode in {"bandpass", "bandstop"}:
        if not freqs or len(freqs) != 2 or not all(isinstance(x, (int, float)) for x in freqs):
            raise ValueError("bandpass/bandstop 模式需要 freqs=(fmin, fmax)")
        fmin, fmax = float(freqs[0]), float(freqs[1])
        if not (fmin > 0 and fmax > 0 and fmax > fmin):
            raise ValueError("频率范围必须满足 0 < fmin < fmax")
    else:
        if not freqs or not isinstance(freqs[0], (int, float)):
            raise ValueError("lowpass/highpass 模式需要 freqs=(fcut, None)")
        fmin, fmax = float(freqs[0]), None
        if not (fmin > 0):
            raise ValueError("截止频率必须为正")

    st_f = st.copy()
    # ObsPy 的 filter 支持 type='butterworth', 'cheby1' 等（取决于 SciPy）。
    if mode == 'bandpass':
        st_f.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=int(order), zerophase=zerophase)
    elif mode == 'bandstop':
        st_f.filter('bandstop', freqmin=fmin, freqmax=fmax, corners=int(order), zerophase=zerophase)
    elif mode == 'lowpass':
        st_f.filter('lowpass', freq=fmin, corners=int(order), zerophase=zerophase)
    elif mode == 'highpass':
        st_f.filter('highpass', freq=fmin, corners=int(order), zerophase=zerophase)

    out_dir = _ensure_output_dir(in_path, 'denoise')
    base, ext = os.path.splitext(os.path.basename(in_path))
    suffix = f"_{mode}_o{order}"
    if mode in {"bandpass", "bandstop"}:
        suffix += f"_{fmin:g}-{fmax:g}Hz"
    else:
        suffix += f"_{fmin:g}Hz"
    out_path = os.path.join(out_dir, f"{base}{suffix}{ext or '.mseed'}")
    fmt = (ext[1:].upper()) if ext else 'MSEED'
    st_f.write(out_path, format=fmt)
    return out_path


def advanced_denoise(
    input_path_or_stream: Union[str, object],
    method: str = 'wavelet',
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold: Optional[float] = None,
    thr_mode: str = 'soft',
) -> str:
    """
    高级降噪：默认采用小波阈值（逐道处理）。

    参数
    - method: 'wavelet'（当前支持）
    - wavelet: 小波基名称（如 'db4', 'haar'）
    - level: 分解层数（None 则由 pywt 依据数据长度与小波决定最大可行层）
    - threshold: 阈值（None 则采用 VisuShrink: sigma*sqrt(2*log(n))，sigma 由 MAD 估算）
    - thr_mode: 'soft' 或 'hard'

    返回
    - 输出文件路径，位于 DFSPy_denoise_outputs
    """
    if obspy_read is None:
        raise ImportError("需要安装 obspy 才能处理地震数据。pip install obspy")
    if method != 'wavelet':
        raise ValueError("当前仅支持 method='wavelet'")
    if pywt is None:
        raise ImportError("需要安装 pywt 才能使用小波降噪。pip install PyWavelets")

    if isinstance(input_path_or_stream, str):
        st = read_stream(input_path_or_stream)
        in_path = input_path_or_stream
    elif isinstance(input_path_or_stream, Stream):
        st = input_path_or_stream
        in_path = st[0].stats.get('source', 'stream') if len(st) > 0 else 'stream'
    else:
        raise TypeError("input_path_or_stream 必须是 str 或 Stream")

    st_out = st.copy()
    for tr in st_out:
        x = tr.data.astype(np.float64)
        # 估计噪声 sigma via MAD of detail coeff at highest level
        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        L = level if (level is not None and level > 0) else max_level
        coeffs = pywt.wavedec(x, wavelet, level=L)
        detail = coeffs[1] if len(coeffs) > 1 else coeffs[0]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
        thr = threshold if (threshold is not None) else sigma * math.sqrt(2 * math.log(len(x) + 1))
        coeffs_thr = [coeffs[0]] + [pywt.threshold(c, thr, mode=thr_mode) for c in coeffs[1:]]
        x_rec = pywt.waverec(coeffs_thr, wavelet)
        # 对齐长度
        n = min(len(x_rec), len(x))
        tr.data = x_rec[:n].astype(np.float32)

    out_dir = _ensure_output_dir(in_path, 'denoise')
    base, ext = os.path.splitext(os.path.basename(in_path))
    out_path = os.path.join(out_dir, f"{base}_wd_{wavelet}{('_L'+str(level)) if level else ''}{ext or '.mseed'}")
    fmt = (ext[1:].upper()) if ext else 'MSEED'
    st_out.write(out_path, format=fmt)
    return out_path

def correlation_denoise(
    input_path_or_stream: Union[str, object],
    window_size: int = 1024,
    step_size: int = 512,
    corr_threshold: float = 0.5,
) -> str:
    """
    基于多道信号相关性的降噪（滑动窗口 + 通道间平均相关性阈值）。

    参考 Notebook 中的示例算法：
    - 对每个窗口计算通道间相关性矩阵，取每道的平均绝对相关性作为“信号置信度”。
    - 选出 >= corr_threshold 的“信号道”，以其均值作为信号模型；
      对各道进行噪声抑制：x_deno = x - (1 - |corr(x, model)|)*(x - model)。

    参数
    - input_path_or_stream: 输入文件路径（SAC/MSEED 等）或 Stream
    - window_size: 窗口长度（采样点）
    - step_size: 窗口步长（采样点）
    - corr_threshold: 平均相关性阈值（0~1）

    返回
    - 输出文件路径（DFSPy_denoise_outputs 下）
    """
    if obspy_read is None or Stream is None or Trace is None:
        raise ImportError("需要安装 obspy 才能处理地震数据。pip install obspy")

    if isinstance(input_path_or_stream, str):
        st_in = read_stream(input_path_or_stream)
        in_path = input_path_or_stream
    elif isinstance(input_path_or_stream, Stream):
        st_in = input_path_or_stream
        in_path = st_in[0].stats.get('source', 'stream') if len(st_in) > 0 else 'stream'
    else:
        raise TypeError("input_path_or_stream 必须是 str 或 Stream")

    if len(st_in) < 2:
        raise ValueError("至少需要 2 道数据用于相关性降噪")

    # 采样率与长度一致性检查
    fs = float(st_in[0].stats.sampling_rate) if hasattr(st_in[0].stats, 'sampling_rate') else None
    npts = int(st_in[0].stats.npts)
    for tr in st_in[1:]:
        if fs is not None and hasattr(tr.stats, 'sampling_rate'):
            if not np.isclose(float(tr.stats.sampling_rate), fs):
                raise ValueError("所有道的采样率必须一致")
        if int(tr.stats.npts) != npts:
            raise ValueError("所有道的样点数必须一致")

    if not (isinstance(window_size, int) and isinstance(step_size, int)):
        raise ValueError("window_size/step_size 必须为整数")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size/step_size 必须为正数")
    if window_size > npts:
        raise ValueError("window_size 不应大于数据长度")
    if not (0 <= corr_threshold <= 1):
        raise ValueError("corr_threshold 应位于 [0,1] 区间")

    # 数据堆叠与去趋势
    data = np.vstack([tr.data for tr in st_in]).astype(np.float64)  # (n_channels, n_samples)
    from scipy.signal import detrend as _detrend
    data = np.vstack([_detrend(ch) for ch in data])
    n_channels, n_samples = data.shape

    deno = np.zeros_like(data)
    n_windows = (n_samples - window_size) // step_size + 1
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        if end > n_samples:
            end = n_samples
            start = end - window_size

        win = data[:, start:end]
        # 相关性矩阵与平均相关性
        if win.shape[1] < 2:
            deno[:, start:end] = win
            continue
        corr = np.corrcoef(win)
        mean_corr = np.mean(np.abs(corr), axis=0)
        signal_mask = mean_corr >= corr_threshold
        n_sig = int(np.sum(signal_mask))

        if n_sig >= 1:
            model = np.mean(win[signal_mask, :], axis=0)
            # 遍历各道，按与 model 的相关性进行抑制
            for ch in range(n_channels):
                ch_corr = np.corrcoef(win[ch, :], model)[0, 1]
                ch_corr = 0.0 if np.isnan(ch_corr) else ch_corr
                deno_win = win[ch, :] - (1 - abs(ch_corr)) * (win[ch, :] - model)
                deno[ch, start:end] = deno_win
        else:
            deno[:, start:end] = win

    # 组装输出 Stream
    st_out = Stream()
    for i, tr in enumerate(st_in):
        new_tr = Trace(data=deno[i, :].astype(np.float32))
        new_tr.stats = tr.stats.copy()
        st_out.append(new_tr)

    out_dir = _ensure_output_dir(in_path, 'denoise')
    base, ext = os.path.splitext(os.path.basename(in_path))
    out_path = os.path.join(out_dir, f"{base}_corr_w{window_size}_s{step_size}_t{corr_threshold:.2f}{ext or '.mseed'}")
    fmt = (ext[1:].upper()) if ext else 'MSEED'
    st_out.write(out_path, format=fmt)
    return out_path

def spectral_subtraction_denoise(
    input_path_or_stream: Union[str, object],
    frame_length: int = 1024,
    hop_length: int = 512,
    window: str = 'hann',
    noise_frames: int = 10,
    alpha: float = 1.0,
    beta: float = 0.02,
) -> str:
    """
    谱减法降噪（逐道 STFT / ISTFT）：

    - 估计噪声谱幅度 |N(f)|（默认取首 `noise_frames` 帧幅度的均值）。
    - 逐帧逐频执行谱减：|Y| = max(|X| - alpha*|N|, beta*|N|)，相位保持。
    - 典型参数：alpha≈1.0（补偿系数），beta≈0.01~0.05（谱地板比例）。

    参数
    - input_path_or_stream: 输入文件路径（SAC/MSEED 等）或 Stream
    - frame_length: STFT 窗口长度（点数）
    - hop_length: 帧移（点数），通常 <= frame_length/2
    - window: 窗口类型（scipy.signal.get_window 支持的名称）
    - noise_frames: 用于估计噪声的前置帧数，若不足则退化为全局最小值估计
    - alpha: 谱减系数（>0）
    - beta: 地板系数（>=0），避免音乐噪声

    返回
    - 输出文件路径（DFSPy_denoise_outputs 下）
    """
    if obspy_read is None or Stream is None or Trace is None:
        raise ImportError("需要安装 obspy 才能处理地震数据。pip install obspy")
    try:
        from scipy.signal import stft as _stft, istft as _istft, get_window as _get_window
    except Exception:
        raise ImportError("需要安装 scipy 才能使用谱减法。pip install scipy")

    if isinstance(input_path_or_stream, str):
        st_in = read_stream(input_path_or_stream)
        in_path = input_path_or_stream
    elif isinstance(input_path_or_stream, Stream):
        st_in = input_path_or_stream
        in_path = st_in[0].stats.get('source', 'stream') if len(st_in) > 0 else 'stream'
    else:
        raise TypeError("input_path_or_stream 必须是 str 或 Stream")

    if len(st_in) == 0:
        raise ValueError("输入 Stream 为空")

    # 输出容器
    st_out = Stream()

    for tr in st_in:
        x = tr.data.astype(np.float64, copy=False)
        n_samples = len(x)
        if n_samples == 0:
            new_tr = Trace(data=x.astype(np.float32))
            new_tr.stats = tr.stats.copy()
            st_out.append(new_tr)
            continue

        nperseg = int(max(2, min(frame_length, n_samples)))
        noverlap = int(max(0, min(nperseg - 1, nperseg - hop_length)))
        try:
            win = _get_window(window, nperseg, fftbins=True)
        except Exception:
            win = _get_window('hann', nperseg, fftbins=True)

        fs = float(getattr(tr.stats, 'sampling_rate', 1.0) or 1.0)

        # STFT
        f, t, Zxx = _stft(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, boundary='zeros', padded=True, return_onesided=True)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 噪声谱估计
        if noise_frames is not None and noise_frames > 0 and mag.shape[1] >= 1:
            nf = int(min(noise_frames, mag.shape[1]))
            noise_mag = np.mean(mag[:, :nf], axis=1)
        else:
            # 退化：使用每个频点的最小幅度作为噪声估计
            noise_mag = np.min(mag, axis=1) if mag.size else np.array([], dtype=np.float64)

        # 避免除零
        eps = 1e-12
        noise_mag = np.maximum(noise_mag, eps)

        # 谱减（向量化）
        floor = beta * noise_mag[:, None]
        mag_hat = np.maximum(mag - alpha * noise_mag[:, None], floor)

        # 重构复谱并 iSTFT
        Y = mag_hat * np.exp(1j * phase)
        _, y = _istft(Y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=True)

        # 对齐长度
        if len(y) < n_samples:
            y = np.pad(y, (0, n_samples - len(y)))
        elif len(y) > n_samples:
            y = y[:n_samples]

        new_tr = Trace(data=y.astype(np.float32))
        new_tr.stats = tr.stats.copy()
        st_out.append(new_tr)

    out_dir = _ensure_output_dir(in_path, 'denoise')
    base, ext = os.path.splitext(os.path.basename(in_path))
    suffix = f"_specsub_n{int(noise_frames)}_a{alpha:g}_b{beta:g}_w{str(window)}_fl{int(frame_length)}_hl{int(hop_length)}"
    out_path = os.path.join(out_dir, f"{base}{suffix}{ext or '.mseed'}")
    fmt = (ext[1:].upper()) if ext else 'MSEED'
    st_out.write(out_path, format=fmt)
    return out_path

def strain_to_velocity(input_path_or_stream: Union[str, object], apparent_velocity: float, normalize_divisor: float = 5000.0) -> str:
    """
    应变 -> 速度 的简单转换（按原程序约定）：
    - 先除以 normalize_divisor（默认 5000，用于从工程量换算到“标准应变”）
    - 再乘以 apparent_velocity（沿光缆视速度，单位 m/s）

    参数
    - input_path_or_stream: 输入文件路径（SAC/MSEED 等）或 Stream
    - apparent_velocity: 视速度 (m/s)，必须为正
    - normalize_divisor: 规范化除数，>0

    返回
    - 输出文件路径（与输入同目录的 DFSPy_paraconv_outputs 子目录）
    """
    if obspy_read is None:
        raise ImportError("需要安装 obspy 才能进行参量转换。pip install obspy")
    if not (apparent_velocity and apparent_velocity > 0):
        raise ValueError("apparent_velocity 必须为正值 (m/s)")
    if not (normalize_divisor and normalize_divisor > 0):
        raise ValueError("normalize_divisor 必须为正值")

    if isinstance(input_path_or_stream, str):
        st = read_stream(input_path_or_stream)
        in_path = input_path_or_stream
    elif isinstance(input_path_or_stream, Stream):
        st = input_path_or_stream
        in_path = st[0].stats.get('source', 'stream') if len(st) > 0 else 'stream'
    else:
        raise TypeError("input_path_or_stream 必须是 str 或 Stream")

    st_v = st.copy()
    for tr in st_v:
        tr.data = (tr.data.astype(np.float64) / normalize_divisor) * apparent_velocity
        tr.data = tr.data.astype(np.float32)

    out_dir = _ensure_output_dir(in_path, 'paraconv')
    in_base = os.path.basename(in_path)
    base, ext = os.path.splitext(in_base)
    ext = ext or '.mseed'
    out_path = os.path.join(out_dir, f"{base}_velocity{ext}")
    fmt = ext[1:].upper()
    st_v.write(out_path, format=fmt)
    return out_path


# ----------------------------- 压缩 / 解压 -----------------------------

def compress_fwt_txt(txt_path: str, wavelet: str = 'haar') -> Tuple[str, float]:
    """
    使用离散小波（FWT）对 txt 矩阵数据进行系数压缩，并保存为 .pkl。

    参数
    - txt_path: 输入 txt 数据文件（samples x traces）
    - wavelet: 小波基名称（默认 'haar'）

    返回
    - (输出 .pkl 路径, 压缩比百分数)，压缩比=压缩文件大小/原文件大小*100

    可能异常
    - ImportError: 未安装 pywt
    - FileNotFoundError/ValueError
    """
    if pywt is None:
        raise ImportError("需要安装 pywt 才能进行 FWT 压缩。pip install PyWavelets")
    data = read_txt_array(txt_path)
    out_dir = _ensure_output_dir(txt_path, 'compress')

    size_src = os.path.getsize(txt_path)
    coeffs: Dict[str, object] = {}
    for i in range(data.shape[1]):
        coeffs[f'coefficients_{i}'] = pywt.wavedec(data[:, i], wavelet)

    base = os.path.splitext(os.path.basename(txt_path))[0]
    out_pkl = os.path.join(out_dir, f"{base}-coefficients.pkl")
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'wavelet': wavelet,
            'shape': data.shape,
            'coeffs': coeffs,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_cmp = os.path.getsize(out_pkl)
    ratio = (size_cmp / size_src * 100.0) if size_src > 0 else math.nan
    return out_pkl, ratio


#----------------


import os
import math
import pickle
from typing import Tuple, Dict, List
import numpy as np
import pywt
from scipy.fftpack import fft
# from sklearn.cluster import KMeans

def _estimate_signal_features(signal: np.ndarray) -> Dict:
    """估算一维信号的特征（用于自适应小波选择）"""
    # 计算频谱特征
    freq = np.abs(fft(signal))[:len(signal)//2]
    dominant_freq = np.argmax(freq) / len(signal)  # 主导频率归一化
    # 计算信号突变程度（一阶差分的方差）
    diff_var = np.var(np.diff(signal))
    return {
        "dominant_freq": dominant_freq,
        "diff_var": diff_var  # 值越大，信号突变越剧烈
    }

def _select_optimal_wavelet(features: Dict) -> Tuple[str, int]:
    """根据信号特征选择最优小波基和分解层数"""
    # 小波基候选集（针对地震信号特性筛选）
    wavelet_candidates = {
        "db4": (0.1, 0.3),   # 适合低频平稳信号（主导频率0.1-0.3）
        "sym5": (0.3, 0.6),  # 适合中高频瞬态信号
        "coif3": (0.6, 1.0)  # 适合高频噪声较多的信号
    }
    # 匹配主导频率对应的小波基
    df = features["dominant_freq"]
    selected_wavelet = "haar"  # 默认值
    for wav, (low, high) in wavelet_candidates.items():
        if low <= df < high:
            selected_wavelet = wav
            break
    # 根据信号复杂度动态确定分解层数（3-5层）
    max_level = 3 if features["diff_var"] < 1e-4 else 5
    return selected_wavelet, max_level

# def _2d_sparse_compress(coeffs_matrix: np.ndarray, threshold_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
#     """对系数矩阵进行二维稀疏压缩（利用列间相关性），返回压缩后的矩阵和聚类标签"""
#     # 对系数矩阵进行K-means聚类，相似列合并压缩
#     k = max(2, coeffs_matrix.shape[1] // 10)  # 聚类数（每10列聚为一类）
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(coeffs_matrix.T)
#     compressed = []
#     for cluster_id in range(k):
#         cluster_cols = coeffs_matrix[:, kmeans.labels_ == cluster_id]
#         # 对聚类内的列取均值作为代表，保留关键信息
#         cluster_mean = np.mean(cluster_cols, axis=1)
#         # 阈值过滤：保留能量前(1-threshold_ratio)的系数
#         energy = np.sum(cluster_mean **2)
#         sorted_coeffs = np.sort(np.abs(cluster_mean))[::-1]
#         cumulative_energy = np.cumsum(sorted_coeffs** 2) / energy
#         threshold_idx = np.argmax(cumulative_energy >= (1 - threshold_ratio))
#         cluster_mean[np.abs(cluster_mean) < sorted_coeffs[threshold_idx]] = 0
#         compressed.append(cluster_mean)
#     return np.column_stack(compressed), kmeans.labels_  # 返回压缩矩阵和标签

# def improved_compress_fwt(txt_path: str) -> Tuple[str, float, float]:
#     """
#     改进的FWT压缩算法：自适应小波选择+二维稀疏压缩+去噪一体化
    
#     返回：(输出.pkl路径, 压缩比%, 信号信噪比SNR)
#     """
#     try:
#         import pywt
#     except ImportError:
#         raise ImportError("需要安装 pywt：pip install PyWavelets")
    
#     # 读取原始数据（samples x traces 二维矩阵）
#     data = np.loadtxt(txt_path)  # 假设txt为空格分隔的矩阵
#     if len(data.shape) != 2:
#         raise ValueError("输入数据必须是二维矩阵（samples x traces）")
#     size_src = os.path.getsize(txt_path)
#     out_dir = os.path.join(os.path.dirname(txt_path), "compressed_improved")
#     os.makedirs(out_dir, exist_ok=True)
    
#     # 逐列进行自适应小波分解
#     all_coeffs = []
#     wavelet_info = []  # 记录每列使用的小波基和层数
#     for i in range(data.shape[1]):
#         signal = data[:, i]
#         # 1. 特征估算与小波自适应选择
#         features = _estimate_signal_features(signal)
#         wavelet, level = _select_optimal_wavelet(features)
#         wavelet_info.append({"wavelet": wavelet, "level": level})
#         # 2. 小波分解与去噪（贝叶斯阈值）
#         coeffs = pywt.wavedec(signal, wavelet, level=level)
#         # 对细节系数进行去噪（保留近似系数，对高频细节施加阈值）
#         denoised_coeffs = [coeffs[0]]  # 近似系数保留
#         for cD in coeffs[1:]:
#             # 贝叶斯阈值计算（基于噪声估计）
#             sigma = np.median(np.abs(cD)) / 0.6745  # 噪声标准差估计
#             threshold = sigma * np.sqrt(2 * np.log(len(cD)))
#             denoised_coeffs.append(pywt.threshold(cD, threshold, mode="soft"))
#         all_coeffs.append(denoised_coeffs)
    
#     # 3. 二维稀疏压缩（利用列间相关性）
#     # 将系数展平为矩阵（便于二维处理）
#     max_len = max(len(np.concatenate(coeffs)) for coeffs in all_coeffs)
#     coeffs_matrix = np.zeros((max_len, data.shape[1]))
#     for i, coeffs in enumerate(all_coeffs):
#         flat_coeffs = np.concatenate(coeffs)
#         coeffs_matrix[:len(flat_coeffs), i] = flat_coeffs
#     # 二维稀疏压缩（获取压缩矩阵和聚类标签）
#     compressed_coeffs, cluster_labels = _2d_sparse_compress(coeffs_matrix)
    
#     # 保存压缩结果
#     base = os.path.splitext(os.path.basename(txt_path))[0]
#     out_pkl = os.path.join(out_dir, f"{base}-improved-coeffs.pkl")
#     with open(out_pkl, 'wb') as f:
#         pickle.dump({
#             "wavelet_info": wavelet_info,
#             "original_shape": data.shape,
#             "compressed_coeffs": compressed_coeffs,
#             "cluster_labels": cluster_labels  # 使用从函数返回的聚类标签
#         }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#     # 计算压缩比和信噪比（SNR）
#     size_cmp = os.path.getsize(out_pkl)
#     ratio = (size_cmp / size_src * 100.0) if size_src > 0 else math.nan
#     # 估算信噪比（假设去噪后的信号能量/噪声能量）
#     signal_energy = np.sum(data **2)
#     noise_energy = signal_energy - np.sum(compressed_coeffs** 2)
#     snr = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else math.inf
    
#     return out_pkl, ratio, snr
    
# ------------------


def decompress_fwt_to_txt(pkl_path: str, out_txt_name: Optional[str] = None) -> str:
    """
    从 FWT 系数 .pkl 重构数据并导出为 txt。

    参数
    - pkl_path: 压缩得到的 .pkl
    - out_txt_name: 可选，输出 txt 文件名（不含路径）

    返回
    - 输出 txt 路径
    """
    if pywt is None:
        raise ImportError("需要安装 pywt 才能进行 FWT 解压。pip install PyWavelets")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"未找到 pkl 文件: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    wavelet = payload.get('wavelet', 'haar')
    shape = tuple(payload.get('shape', ()))
    coeffs: Dict[str, object] = payload['coeffs']
    if not shape or len(shape) != 2:
        raise ValueError("pkl 中缺少有效的原始形状信息")

    n_samples, n_traces = shape
    data = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        c = coeffs.get(f'coefficients_{i}')
        if c is None:
            raise ValueError(f"pkl 不包含 coefficients_{i}")
        rec = pywt.waverec(c, wavelet)
        # 对齐长度
        L = min(len(rec), n_samples)
        data[:L, i] = rec[:L]

    out_dir = _ensure_output_dir(pkl_path, 'decompress')
    base = os.path.splitext(os.path.basename(pkl_path))[0]
    out_name = out_txt_name or f"{base}-reconstructed.txt"
    out_path = os.path.join(out_dir, out_name)
    np.savetxt(out_path, data, fmt='%.7e')
    return out_path


def gzip_compress(file_path: str) -> str:
    """
    使用 gzip 对任意单个文件进行压缩，输出 .gz。

    返回输出 .gz 路径。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    out_dir = _ensure_output_dir(file_path, 'compress')
    base = os.path.basename(file_path)
    out_path = os.path.join(out_dir, f"{base}.gz")
    with open(file_path, 'rb') as fin, gzip.open(out_path, 'wb') as fout:
        shutil.copyfileobj(fin, fout)
    return out_path


def gzip_decompress(gz_path: str, out_name: Optional[str] = None) -> str:
    """
    解压 gzip 文件，输出到同级 DFSPy_decompress_outputs 目录。

    参数
    - gz_path: .gz 文件
    - out_name: 可选，输出文件名（不含路径）

    返回
    - 输出文件路径
    """
    if not os.path.isfile(gz_path):
        raise FileNotFoundError(f"未找到文件: {gz_path}")
    out_dir = _ensure_output_dir(gz_path, 'decompress')
    base = os.path.basename(gz_path)
    if base.lower().endswith('.gz'):
        base = base[:-3]
    out_path = os.path.join(out_dir, out_name or base)
    with gzip.open(gz_path, 'rb') as fin, open(out_path, 'wb') as fout:
        shutil.copyfileobj(fin, fout)
    return out_path


# ----------------------------- 简易自检 -----------------------------

def _self_check_dependencies() -> Dict[str, bool]:
    """返回依赖可用性，便于在 Notebook 中快速诊断。"""
    return {
        'matplotlib': plt is not None,
        'obspy': obspy_read is not None and Stream is not None,
        'pywt': pywt is not None,
        'numpy': True,
    }


__all__ = [
    # IO
    'read_headfile', 'read_txt_array', 'read_stream', 'array_to_stream_from_head',
    # plotting
    'plot_array', 'plot_stream',
    # convert
    'convert_format',
    # denoise
    'bandpass_denoise', 'filter_stream', 'advanced_denoise', 'correlation_denoise', 'spectral_subtraction_denoise',
    # parameter conversion
    'strain_to_velocity',
    # compression
    'compress_fwt_txt', 'decompress_fwt_to_txt', 'gzip_compress', 'gzip_decompress',
    # misc
    '_self_check_dependencies',
]

'''
*************************************************************************************************************
Software Name         : Distributed Fiber Optic Sensing Data Processing Software [Abbreviation: DFSPy] V1.0
Copyright Holder      : Institute of Semiconductors, Chinese Academy of Sciences
Software Copyright Registration Number: 2025SR0353448
Contact Email         : qi.gh@outlook.com
Developer Homepage    : https://github.com/chyiever
Creation Date         : 2025-09-25
Software Introduction :DFSPy is a professional data processing tool designed for researchers working with
    distributed fiber optic sensing systems. It aims to provide efficient and reliable data processing solutions, 
    integrating various advanced algorithms to support preprocessing, analysis, and visualization of data 
    collected by distributed fiber optic sensing systems. This tool helps researchers explore data value and
    accelerate research progress.
Usage Statement:
  1. This software is an open-source research tool developed by the Institute of Semiconductors, CAS, 
     intended solely for academic research and non-commercial use.
  2. Users must comply with relevant national laws, regulations, and research ethics when using this 
     software, and shall not use it for any commercial activities or illegal purposes.
  3. The software is provided "as is" without any express or implied warranties regarding its applicability,
     completeness, or accuracy.
  4. Users shall independently evaluate and assume all risks arising from the use of this software. The 
      developers shall not be liable for any direct or indirect losses resulting from the use of this software.
  5. When citing this software in academic publications, users are requested to appropriately acknowledge the
     software source and developer information.
*************************************************************************************************************
'''
