"""真峰值前视限幅器（true-peak lookahead limiter）。

修复点与缺陷的对应关系
----------------------
1. 真峰值检测：模拟信号的峰值出现在两个采样点之间。先按 ``oversample`` 倍
   零插入上采样，再用 Kaiser 加窗 sinc 低通在稠密网格上取每样本区间峰值，
   避免采样间过冲在有损编码后削波。
2. 前视按样本数：``lookahead_samples`` 把毫秒换算成确定的样本数；限幅器内部
   用延迟线缓存输入，块长小于前视窗口时只缓存不输出，增益不在块边界跳变，
   处理结果与分块大小逐样本无关。
3. 增益跨块继承：``TruePeakLimiter`` 是有状态对象，音频延迟线、上采样器
   历史与当前包络增益都随块延续，连续响段不会每块起始重新压一次（消除抽吸）。
4. 多声道联动：所有声道共享同一条增益曲线（取最严声道的真峰值），左右增益
   一致，声像不偏移。
5. 先重采样后限幅：``limit_after_resample`` 先做采样率转换，再在目标采样率上
   做真峰值限幅，重采样滤波器振铃带来的过冲同样被计入阈值。
6. 释放时间平滑：释放/攻击系数由时间常数与采样率换算（一阶指数），增益只向
   目标逼近、永不越过目标值，平滑与限幅安全同时成立。

时序约定（重要）
----------------
流式上采样是因果 FIR，引入固定 ``tp_delay`` 个输入样本的延迟（核半径除以上
采样倍数，构造上保证为整数）。限幅器的音频延迟线统一吸收这个延迟，对外仍
表现为恰好 ``lookahead`` 个样本的总延迟；离线整段处理与逐块流式走同一套
有状态代码，结果逐样本一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "true_peak_detect", "lookahead_samples", "gain_cross_block",
    "multichannel_link", "limit_after_resample", "smooth_release",
    "process", "compute_gain", "apply", "export", "validate_limit",
    "metrics", "audit", "fallback", "determinism",
    "TruePeakLimiter", "LimiterState", "StreamingUpsampler",
    "StreamingTruePeak", "resample",
    "DEFAULT_CEILING", "DEFAULT_SAMPLE_RATE", "DEFAULT_OVERSAMPLE",
    "DEFAULT_LOOKAHEAD_MS", "DEFAULT_RELEASE_MS",
]

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_OVERSAMPLE = 4
DEFAULT_LOOKAHEAD_MS = 1.0
DEFAULT_RELEASE_MS = 60.0
# 有损编码（AAC/MP3 等）会引入采样间峰值，工程上预留 1 dBTP 头顶空间。
DEFAULT_CEILING_DBTP = -1.0
DEFAULT_CEILING = 10.0 ** (DEFAULT_CEILING_DBTP / 20.0)

_EPS = 1e-12
_HALF_TAPS = 8  # 低通滤波器每侧的脉冲数


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _as_2d(signal) -> tuple[np.ndarray, bool]:
    """把标量/1-D/2-D 输入统一成 (channels, samples) 的 2-D 数组。"""
    a = np.asarray(signal, dtype=np.float64)
    if a.ndim == 0:
        return a.reshape(1, 1), True
    if a.ndim == 1:
        return a[None, :], True
    if a.ndim != 2:
        raise ValueError(f"只支持 0/1/2 维信号，收到 {a.ndim} 维")
    return a, False


def _from_2d(a: np.ndarray, squeeze: bool):
    if not squeeze:
        return a
    if a.shape == (1, 1):
        return float(a[0, 0])
    return a[0]


def _time_coef(sample_rate: int, time_ms: float) -> float:
    """一阶指数系数：``time_ms`` 内衰减到 e^-3（约 95%）的时间常数约定。"""
    horizon = float(time_ms) * 1e-3 * float(sample_rate)
    if horizon <= 0.0:
        return 0.0
    return math.exp(-3.0 / horizon)


def lookahead_samples(sample_rate: int, lookahead_ms: float = DEFAULT_LOOKAHEAD_MS) -> int:
    """把前视时间换算成样本数。

    前视窗口永远以样本数表示而不是把毫秒写死；采样率确定后窗口长度即确定，
    与设备回调块长无关。
    """
    sr = int(round(float(sample_rate)))
    if sr <= 0:
        raise ValueError("sample_rate 必须为正整数")
    return max(int(round(float(lookahead_ms) * 1e-3 * sr)), 0)


# ---------------------------------------------------------------------------
# FIR 核与上采样
# ---------------------------------------------------------------------------

def _design_upsample_kernel(factor: int) -> np.ndarray:
    """整数倍上采样的零相位对称低通核，长度 ``2*half*factor+1``（奇数）。

    奇数长度保证存在中心抽头；核半径 = half*factor，整除 factor，因此因果
    实现的群延迟恰好是 ``half`` 个输入样本（无分数样本相位）。
    """
    f = int(factor)
    length = 2 * _HALF_TAPS * f + 1
    n = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    kernel = np.sinc(n / f)  # 单位直流增益
    return kernel * np.kaiser(length, 8.6)


def _design_lowpass(up: int, down: int) -> np.ndarray:
    """任意比例重采样用的抗镜像/抗混叠低通 FIR。"""
    high = max(int(up), int(down))
    length = 2 * _HALF_TAPS * high + 1
    cutoff = math.pi / high
    n = np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    kernel = float(up) * (cutoff / math.pi) * np.sinc(n * (cutoff / math.pi))
    return kernel * np.kaiser(length, 8.6)


def _fir2d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对 (channels, samples) 每行做等长卷积（零填充，零相位居中）。"""
    pad = len(kernel) // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="constant", constant_values=0.0)
    return np.stack([np.convolve(row, kernel, mode="valid") for row in padded])


def _resample_polyphase(x2d: np.ndarray, up: int, down: int) -> np.ndarray:
    """有理数倍重采样：零值插入低通后抽取。"""
    if up == down:
        return x2d.copy()
    n = x2d.shape[1]
    inserted = np.zeros((x2d.shape[0], n * up), dtype=np.float64)
    inserted[:, ::up] = x2d
    filtered = _fir2d(inserted, _design_lowpass(up, down))
    decimated = filtered[:, ::down]
    expected = math.ceil(n * up / down)
    if decimated.shape[1] >= expected:
        return decimated[:, :expected]
    return np.pad(decimated, ((0, 0), (0, expected - decimated.shape[1])), mode="edge")


def resample(signal, src_rate: int, dst_rate: int):
    """任意采样率转换（内部约分后走多相滤波）。"""
    a, squeeze = _as_2d(signal)
    src_rate, dst_rate = int(src_rate), int(dst_rate)
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("采样率必须为正整数")
    g = math.gcd(src_rate, dst_rate)
    out = _resample_polyphase(a, dst_rate // g, src_rate // g)
    return _from_2d(out, squeeze)


class StreamingUpsampler:
    """有状态、块不变、长度严格守恒的整数倍因果上采样器。

    维护长度等于核长度的高采样率网格滑动缓存，每来一个零插入网格点就与对称
    核做一次点积，精确产出一个稠密点。输入 N 个样本精确产出 N*factor 个点；
    缓存跨块延续，结果只取决于样本流而与分块无关。流开头缓存为静默零，因果
    群延迟为 ``tp_delay`` 个输入样本。
    """

    def __init__(self, factor: int = DEFAULT_OVERSAMPLE, channels: int = 1):
        self.factor = int(factor)
        if self.factor < 1:
            raise ValueError("上采样倍数必须 >= 1")
        self.channels = int(channels)
        self.kernel = _design_upsample_kernel(self.factor)
        self.klen = len(self.kernel)
        self.radius = (self.klen - 1) // 2
        # 因果延迟（输入样本数），构造保证整除。
        assert self.radius % self.factor == 0
        self.tp_delay = self.radius // self.factor
        self._buf = np.zeros((self.channels, self.klen), dtype=np.float64)

    def reset(self) -> None:
        self._buf.fill(0.0)

    def process(self, block2d: np.ndarray) -> np.ndarray:
        """输入 (channels, n)，返回 (channels, n*factor) 因果稠密网格点。"""
        if self.factor == 1:
            return block2d.copy()
        if block2d.shape[0] != self.channels:
            raise ValueError("声道数与上采样器已有状态不一致")
        if block2d.shape[1] == 0:
            return np.empty((self.channels, 0), dtype=np.float64)
        n = block2d.shape[1]
        inserted = np.zeros((self.channels, n * self.factor), dtype=np.float64)
        inserted[:, ::self.factor] = block2d
        out = np.empty_like(inserted)
        for q in range(inserted.shape[1]):
            # 当前网格点进入缓存右端；点积即该点的因果滤波输出。
            self._buf = np.concatenate(
                [self._buf[:, 1:], inserted[:, q:q + 1]], axis=1)
            out[:, q:q + 1] = (self._buf @ self.kernel).reshape(-1, 1)
        return out


class StreamingTruePeak:
    """因果流式真峰值包络检测器，输出与输入等长、整体延迟 ``tp_delay`` 样本。

    每 factor 个稠密网格点结算一个输入样本的采样间峰值（取区间最大绝对值）。
    上采样的因果延迟意味着输出包络的第 n 项对应输入样本 n-tp_delay；限幅器
    用音频延迟线按同一延迟对齐。flush 把流末尾缓存按静默零结算，长度守恒。
    """

    def __init__(self, oversample: int = DEFAULT_OVERSAMPLE, channels: int = 1):
        self.oversample = int(oversample)
        self.channels = int(channels)
        self._up = StreamingUpsampler(self.oversample, channels)
        self.tp_delay = self._up.tp_delay
        # 纯因果：包络第 k 项对应输入样本 k-D（D=tp_delay）。限幅器用音频
        # 延迟线把"音频样本 k-D"与该峰值在输出端对齐。
        self._grid_buf = np.zeros((channels, 0), dtype=np.float64)

    def reset(self) -> None:
        self._up.reset()
        self._grid_buf = np.zeros((self.channels, 0), dtype=np.float64)

    def _drain(self) -> np.ndarray:
        full = (self._grid_buf.shape[1] // self.oversample) * self.oversample
        if full == 0:
            return np.empty((self.channels, 0), dtype=np.float64)
        grid = self._grid_buf[:, :full].reshape(
            self.channels, full // self.oversample, self.oversample)
        env = np.max(np.abs(grid), axis=-1)
        self._grid_buf = self._grid_buf[:, full:]
        return env

    def process(self, block2d: np.ndarray) -> np.ndarray:
        if self.oversample == 1:
            return np.abs(block2d)
        if block2d.shape[1] == 0:
            return np.empty(block2d.shape, dtype=np.float64)
        self._grid_buf = np.concatenate(
            [self._grid_buf, self._up.process(block2d)], axis=1)
        return self._drain()

    def flush(self) -> np.ndarray:
        """结算流末尾，输出样本数补齐到与已处理输入总数相等。

        因果上采样有 D=tp_delay 个样本的群延迟：最后 D 个真实输入样本的完整
        稠密响应要等 D 个静默未来通过滤波器才出现。这里补推 D 个零样本排空
        核尾部，再按 factor 区间分组，取与真实输入对应的前若干个包络值。
        """
        # 补 D 个静默输入，让核尾部的真实响应全部产出。
        tail_dense = self._up.process(
            np.zeros((self.channels, self.tp_delay), dtype=np.float64))
        if tail_dense.size:
            self._grid_buf = np.concatenate(
                [self._grid_buf, tail_dense], axis=1)
        if self._grid_buf.shape[1] == 0:
            return np.empty((self.channels, 0), dtype=np.float64)
        rem = self._grid_buf.shape[1] % self.oversample
        if rem:
            self._grid_buf = np.pad(
                self._grid_buf, ((0, 0), (0, self.oversample - rem)),
                mode="constant")
        grid = self._grid_buf.reshape(
            self.channels, self._grid_buf.shape[1] // self.oversample,
            self.oversample)
        env = np.max(np.abs(grid), axis=-1)
        self._grid_buf = np.zeros((self.channels, 0), dtype=np.float64)
        return env


# ---------------------------------------------------------------------------
# 1. 真峰值检测
# ---------------------------------------------------------------------------

def _true_peak_envelope(x2d: np.ndarray, oversample: int) -> np.ndarray:
    """每输入样本的真峰值，形状 (channels, samples)，与输入等长、零延迟对齐。

    因果上采样有 D=tp_delay 个样本的群延迟：完整包络序列的第 k 项对应输入
    样本 k-D，因此真实输入样本 i 的包络位于序列位置 i+D，流开头 D 项是静默
    前导。把 process 与 flush 的包络拼接后取 [D, D+N)，即得与零相位离线卷积
    逐值相等、且与输入逐样本对齐的结果。
    """
    factor = int(oversample)
    if factor < 1:
        raise ValueError("oversample 必须 >= 1")
    if factor == 1:
        return np.abs(x2d)
    n = x2d.shape[1]
    det = StreamingTruePeak(factor, channels=x2d.shape[0])
    settled = det.process(x2d)
    tail = det.flush()
    full = np.concatenate([settled, tail], axis=1) if tail.size else settled
    return full[:, det.tp_delay:det.tp_delay + n]


def true_peak_detect(signal, oversample: int = DEFAULT_OVERSAMPLE) -> float:
    """返回信号真峰值（线性幅度，跨所有声道取最大）。"""
    a = np.asarray(signal, dtype=np.float64)
    if a.ndim == 0:
        return float(abs(a))
    a2d, _ = _as_2d(a)
    return float(_true_peak_envelope(a2d, oversample).max())


# ---------------------------------------------------------------------------
# 4. 多声道联动
# ---------------------------------------------------------------------------

def multichannel_link(values, axis: int = 0, mode: str = "peak"):
    """把多声道每样本度量联动成一条公共曲线。

    ``peak``（默认）对真峰值取最大——最响声道决定整条增益曲线；``gain`` 对
    增益取最小，两种表述等价。所有声道随后共用这一条曲线。
    """
    a = np.asarray(values, dtype=np.float64)
    if a.ndim == 0:
        return float(a)
    if mode == "peak":
        return np.max(a, axis=axis)
    if mode == "gain":
        return np.min(a, axis=axis)
    raise ValueError("mode 只能是 'peak' 或 'gain'")


# ---------------------------------------------------------------------------
# 增益计算与释放平滑
# ---------------------------------------------------------------------------

def compute_gain(peak, ceiling: float = DEFAULT_CEILING):
    """静态限幅增益 ``min(1, ceiling/peak)``，向量化。"""
    p = np.abs(np.asarray(peak, dtype=np.float64))
    g = np.minimum(1.0, ceiling / np.maximum(p, _EPS))
    return float(g) if g.ndim == 0 else g


def _step_gain(current: float, target: float, attack_coef: float,
               release_coef: float) -> float:
    """单样本增益包络推进，永不越过目标值。"""
    if target < current:
        return min(target, current * attack_coef)
    if target > current:
        return target + (current - target) * release_coef
    return current


def smooth_release(gain: float, target: float = 1.0, release_coef: Optional[float] = None,
                   *, sample_rate: int = DEFAULT_SAMPLE_RATE,
                   release_ms: float = DEFAULT_RELEASE_MS) -> float:
    """一阶释放平滑：从 gain 向 target 回升一步，永不越过目标。"""
    if release_coef is None:
        release_coef = _time_coef(sample_rate, release_ms)
    g, t = float(gain), float(target)
    if t <= g:
        return t
    return t + (g - t) * float(release_coef)


def gain_cross_block(state, target_gain: float = 1.0, *,
                     sample_rate: int = DEFAULT_SAMPLE_RATE,
                     release_ms: float = DEFAULT_RELEASE_MS,
                     attack_ms: Optional[float] = None) -> float:
    """跨块增益继承：以上一块结束增益为起点推进一个样本。

    state 可以是增益标量、``LimiterState`` 或 ``TruePeakLimiter``。
    """
    if isinstance(state, (LimiterState, TruePeakLimiter)):
        prev = float(state.gain)
    else:
        prev = float(state)
    if attack_ms is None:
        attack_ms = DEFAULT_LOOKAHEAD_MS
    return _step_gain(prev, float(target_gain),
                      _time_coef(sample_rate, attack_ms),
                      _time_coef(sample_rate, release_ms))


# ---------------------------------------------------------------------------
# 有状态真峰值前视限幅器
# ---------------------------------------------------------------------------

@dataclass
class LimiterState:
    """跨块延续的限幅器状态。"""

    gain: float = 1.0
    delay: Optional[np.ndarray] = None
    lookahead: int = 0

    def clone(self) -> "LimiterState":
        delay = None if self.delay is None else self.delay.copy()
        return LimiterState(gain=self.gain, delay=delay, lookahead=self.lookahead)


class TruePeakLimiter:
    """真峰值前视限幅器（流式状态机、有状态、多声道联动）。

    用法::

        lim = TruePeakLimiter(48000)
        out = np.concatenate([lim.process_block(b) for b in blocks], axis=1)
        out = np.concatenate([out, lim.flush()], axis=1)

    内部对每个输入样本做一次确定性更新，处理结果只取决于样本流、与回调块长
    无关。三条对齐的 FIFO：

    * 音频延迟线（缓存待输出样本）；
    * 真峰值/静态增益延迟线（与音频逐样本对齐，含安全钳位增益）；
    * 前视窗口单调最小队列（驱动平滑包络）。
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, *,
                 ceiling: float = DEFAULT_CEILING,
                 lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
                 release_ms: float = DEFAULT_RELEASE_MS,
                 attack_ms: Optional[float] = None,
                 oversample: int = DEFAULT_OVERSAMPLE,
                 safety_margin: float = 0.001,
                 channels: Optional[int] = None):
        self.sample_rate = int(sample_rate)
        # 约 0.1% 余量吸收 FIR 截断误差，保证输出在标称天花板之内。
        self.nominal_ceiling = float(ceiling)
        self.safety_margin = float(safety_margin)
        self.ceiling = float(ceiling) * (1.0 - float(safety_margin))
        self.lookahead = lookahead_samples(self.sample_rate, lookahead_ms)
        self.release_ms = float(release_ms)
        self.attack_ms = float(lookahead_ms if attack_ms is None else attack_ms)
        self.oversample = int(oversample)
        self.release_coef = _time_coef(self.sample_rate, self.release_ms)
        self.attack_coef = _time_coef(self.sample_rate, self.attack_ms)
        self._channels = channels
        self._init_state()

    def _init_state(self) -> None:
        # 音频延迟线 FIFO。
        self._delay = np.zeros(0 if self._channels is None
                               else (self._channels, 0), dtype=np.float64)
        self._tp: Optional[StreamingTruePeak] = (
            None if self._channels is None
            else StreamingTruePeak(self.oversample, self._channels))
        # 因果包络原始队列（含流开头 D 个静默前导）存的是静态增益。
        self._env_queue: list[float] = []
        # 前视窗口单调最小队列：(绝对序号, 静态增益)。
        self._minq: list[tuple[int, float]] = []
        self._ring: list[float] = []
        self._count = 0
        self._n_emitted = 0        # 已输出的音频样本数
        self._gain = 1.0
        self.last_gain_curve: list[np.ndarray] = []
        self.last_flush_curve: np.ndarray = np.empty(0)

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def tp_delay(self) -> int:
        # 真峰值因果延迟只取决于上采样倍数与核设计，与声道数无关。
        return StreamingUpsampler(self.oversample, 1).tp_delay

    def state(self) -> LimiterState:
        delay = None if self._delay.size == 0 else self._delay.copy()
        return LimiterState(gain=self._gain, delay=delay, lookahead=self.lookahead)

    def reset(self) -> None:
        self._init_state()

    def _ensure(self, channels: int) -> None:
        if self._delay.size == 0:
            self._delay = np.zeros((channels, 0), dtype=np.float64)
            self._tp = StreamingTruePeak(self.oversample, channels)
            self._channels = channels
        elif self._delay.shape[0] != channels:
            raise ValueError("声道数与限幅器已有状态不一致")

    def _push_required(self, required: float) -> None:
        idx = self._count
        self._count += 1
        self._ring.append(required)
        while self._minq and self._minq[-1][1] >= required:
            self._minq.pop()
        self._minq.append((idx, required))
        if len(self._ring) > self.lookahead + 1:
            old_idx = idx - self.lookahead
            self._ring.pop(0)
            if self._minq and self._minq[0][0] == old_idx:
                self._minq.pop(0)

    def _emit(self, audio_col: np.ndarray, own_required: float) -> tuple[np.ndarray, float]:
        """平滑包络与该样本自身静态增益取较小者，保证真峰值不超天花板。"""
        envelope = _step_gain(self._gain, self._minq[0][1],
                              self.attack_coef, self.release_coef)
        self._gain = envelope
        final = min(envelope, float(own_required))
        return audio_col * final, final

    def _ingest_envelopes(self, peak_env: np.ndarray) -> None:
        """把检测器结算出的因果包络转成静态增益，放入原始包络队列。"""
        if not peak_env.size:
            return
        linked = multichannel_link(peak_env, axis=0, mode="peak")
        for req in np.atleast_1d(compute_gain(linked, self.ceiling)):
            self._env_queue.append(float(req))

    def _drain_outputs(self, out_cols, gain_rows) -> None:
        """配对输出：音频样本 i 的真实包络位于包络队列位置 i+D。

        已输出 i 个音频后，下一个音频 i 的包络索引是 i+D；只要队列长度超过
        i+D、且音频在延迟线等满前视 L（总在 FIFO 中保留 L 个未来样本），就
        持续输出。前视单调队列由已入队的静态增益驱动。
        """
        D, L = self.tp_delay, self.lookahead
        while True:
            env_idx = self._n_emitted + D
            # 音频 i 的包络尚未结算，或延迟线里不足 L 个未来样本。
            if env_idx >= len(self._env_queue):
                break
            if self._delay.shape[1] <= L:
                break
            req = self._env_queue[env_idx]
            self._push_required(req)
            audio = self._delay[:, 0].copy()
            self._delay = self._delay[:, 1:]
            col, final_gain = self._emit(audio, req)
            out_cols.append(col)
            gain_rows.append(final_gain)
            self._n_emitted += 1

    def process_block(self, block):
        """处理一个块，返回同形状输出（块长可小于前视窗口，此时输出空块）。

        严格逐样本推进，结果与分块方式无关。音频入 FIFO、因果包络入队列，
        再由 _drain_outputs 按"包络 i+D 对应音频 i、保留 L 个未来样本"的统一
        规则配对输出。
        """
        x, squeeze = _as_2d(block)
        self._ensure(x.shape[0])
        for j in range(x.shape[1]):
            self._delay = np.concatenate(
                [self._delay, x[:, j:j + 1]], axis=1)
        self._ingest_envelopes(self._tp.process(x))
        out_cols, gain_rows = [], []
        self._drain_outputs(out_cols, gain_rows)
        if out_cols:
            curve = np.asarray(gain_rows, dtype=np.float64)
            self.last_gain_curve.append(curve)
            out = np.stack(out_cols, axis=1)
        else:
            out = np.empty((x.shape[0], 0), dtype=np.float64)
        return out[0] if squeeze else out

    def flush(self):
        """排空：补静默零结算真峰值核尾部，再输出延迟线全部剩余样本。

        前视窗口的未知未来一律按静态增益 1.0 入队，增益沿一阶释放曲线回到
        1.0；每个输出音频仍与它在包络队列 i+D 处的真实静态增益配对钳位。
        """
        if self._delay.size == 0 or self._delay.shape[1] == 0:
            channels = self._channels
            self.reset()
            return np.zeros(0) if channels is None else np.zeros((channels, 0))
        # 核尾部包络入队（含最后 D 个真实样本的完整响应）。
        self._ingest_envelopes(self._tp.flush())
        n_tail = self._delay.shape[1]
        gain_rows, out_cols = [], []
        for _ in range(n_tail):
            env_idx = self._n_emitted + self.tp_delay
            if env_idx < len(self._env_queue):
                req = self._env_queue[env_idx]
            else:
                req = 1.0  # 超出真实包络的静默未来。
            self._push_required(1.0)  # 前视未知未来。
            audio = self._delay[:, 0].copy()
            self._delay = self._delay[:, 1:]
            col, final_gain = self._emit(audio, req)
            out_cols.append(col)
            gain_rows.append(final_gain)
            self._n_emitted += 1
        curve = np.asarray(gain_rows, dtype=np.float64)
        out = np.stack(out_cols, axis=1)
        squeeze = out.shape[0] == 1
        block_curves = list(self.last_gain_curve)
        self.reset()
        self.last_gain_curve = block_curves
        self.last_flush_curve = curve
        return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# 2/3/6. 整流处理：分块无关、状态继承、平滑释放
# ---------------------------------------------------------------------------

def _run_stream(signal, sample_rate: int, block_size: Optional[int],
                limiter: TruePeakLimiter):
    def as2d(v):
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr[None, :]
        return arr

    a, squeeze = _as_2d(signal)
    if block_size is None:
        pieces = [as2d(limiter.process_block(a))]
    else:
        block_size = max(int(block_size), 1)
        pieces = [as2d(limiter.process_block(a[:, i:i + block_size]))
                  for i in range(0, a.shape[1], block_size)]
    pieces = [p for p in pieces if p.size]
    tail = as2d(limiter.flush())
    if tail.size:
        pieces.append(tail)
    out_arr = (np.concatenate(pieces, axis=1) if pieces
               else np.empty((a.shape[0], 0), dtype=np.float64))
    return _from_2d(out_arr, squeeze)


def _true_peak_safety_net(out2d: np.ndarray, ceiling: float,
                          oversample: int, tolerance: float = 1e-9,
                          max_iter: int = 4) -> np.ndarray:
    """限幅后真峰值安全网。

    增益曲线乘到信号上会改变采样间结构（增益在相邻样本间变化本身就会引入
    新的 sinc 重建过冲），逐样本静态钳位无法保证输出真峰值。这里对限幅结果
    再做真峰值检测，若超标则整体补偿，迭代到合规为止。补偿是确定性的标量
    运算，不破坏块不变性；实践中一两次即收敛。
    """
    for _ in range(max_iter):
        tp = float(_true_peak_envelope(out2d, oversample).max()) if out2d.size else 0.0
        if tp <= ceiling + tolerance:
            return out2d
        out2d = out2d * (ceiling / tp)
    return out2d


def process(signal, sample_rate: int = DEFAULT_SAMPLE_RATE, *,
            block_size: Optional[int] = None,
            ceiling: float = DEFAULT_CEILING,
            lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
            release_ms: float = DEFAULT_RELEASE_MS,
            attack_ms: Optional[float] = None,
            oversample: int = DEFAULT_OVERSAMPLE):
    """整条信号真峰值前视限幅。block_size 仅影响流式切分，不影响结果。

    流式限幅之后再经过一道限幅后真峰值安全网，吸收增益调制引入的采样间
    过冲，保证输出真峰值严格不超过天花板。
    """
    limiter = TruePeakLimiter(
        sample_rate, ceiling=ceiling, lookahead_ms=lookahead_ms,
        release_ms=release_ms, attack_ms=attack_ms, oversample=oversample)
    out = _run_stream(signal, sample_rate, block_size, limiter)
    out2d, squeeze = _as_2d(out)
    safe = _true_peak_safety_net(out2d, limiter.ceiling, oversample)
    return _from_2d(safe, squeeze)


def apply(signal, gain=1.0):
    """把增益（标量或曲线）乘到信号上。"""
    out = np.asarray(signal, dtype=np.float64) * np.asarray(gain, dtype=np.float64)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------
# 5. 先重采样、后限幅
# ---------------------------------------------------------------------------

def limit_after_resample(signal, src_rate: int = DEFAULT_SAMPLE_RATE,
                         dst_rate: Optional[int] = None, *,
                         ceiling: float = DEFAULT_CEILING,
                         block_size: Optional[int] = None,
                         lookahead_ms: float = DEFAULT_LOOKAHEAD_MS,
                         release_ms: float = DEFAULT_RELEASE_MS,
                         oversample: int = DEFAULT_OVERSAMPLE):
    """先重采样到目标采样率，再在目标采样率上真峰值限幅。

    重采样滤波器振铃过冲在限幅之前已存在，会被真峰值检测计入阈值；先限幅
    后重采样则会放过这部分过冲。
    """
    dst_rate = int(src_rate if dst_rate is None else dst_rate)
    resampled = resample(signal, int(src_rate), dst_rate)
    return process(resampled, dst_rate, block_size=block_size, ceiling=ceiling,
                   lookahead_ms=lookahead_ms, release_ms=release_ms,
                   oversample=oversample)


# ---------------------------------------------------------------------------
# 导出、校验、度量、兜底
# ---------------------------------------------------------------------------

def export(signal, sample_rate: int = DEFAULT_SAMPLE_RATE, bit_depth: int = 16,
           ceiling: float = DEFAULT_CEILING):
    """导出环节确定性模拟：量化到指定位深 PCM 再回到浮点域。

    真实有损编码还会引入额外采样间峰值，故限幅天花板默认预留 1 dBTP 头顶
    空间，并应以 validate_limit 在导出后复核。
    """
    a = np.asarray(signal, dtype=np.float64)
    if bit_depth < 2 or bit_depth > 64:
        raise ValueError("bit_depth 需在 2..64 之间")
    levels = 2 ** (bit_depth - 1)
    quantized = np.clip(np.round(a * levels), -levels, levels - 1) / levels
    return float(quantized) if quantized.ndim == 0 else quantized


def validate_limit(signal, ceiling: float = DEFAULT_CEILING,
                   oversample: int = DEFAULT_OVERSAMPLE,
                   tolerance: float = 1e-6) -> bool:
    """真峰值合规校验：整条信号（含采样间）不得超过天花板。"""
    return true_peak_detect(signal, oversample) <= float(ceiling) + float(tolerance)


def metrics(signal, sample_rate: int = DEFAULT_SAMPLE_RATE,
            oversample: int = DEFAULT_OVERSAMPLE) -> dict:
    """峰值/真峰值/响度等度量，标量输入也可调用。"""
    a = np.asarray(signal, dtype=np.float64)
    sample_peak = float(np.max(np.abs(a))) if a.size else 0.0
    true_peak = true_peak_detect(a if a.ndim else float(a), oversample)
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    crest = (sample_peak / rms) if rms > _EPS else 0.0
    return {
        "channels": 1 if a.ndim <= 1 else a.shape[0],
        "samples": int(a.size if a.ndim <= 1 else a.shape[1]),
        "sample_rate": int(sample_rate),
        "sample_peak": sample_peak,
        "true_peak": true_peak,
        "true_peak_dbfs": 20.0 * math.log10(max(true_peak, _EPS)),
        "rms": rms,
        "crest_factor_db": 20.0 * math.log10(max(crest, _EPS)),
        "oversample": int(oversample),
    }


def fallback(signal, ceiling: float = DEFAULT_CEILING,
             oversample: int = DEFAULT_OVERSAMPLE, tolerance: float = 1e-6):
    """安全兜底：真峰值超标时整体补偿到天花板之内再返回。"""
    a = np.asarray(signal, dtype=np.float64)
    true_peak = true_peak_detect(a if a.ndim else float(a), oversample)
    if true_peak <= ceiling + tolerance:
        return signal if isinstance(signal, np.ndarray) else a
    return apply(a, float(ceiling) / true_peak)


# ---------------------------------------------------------------------------
# 确定性与整体审计
# ---------------------------------------------------------------------------

def _fixture(sample_rate: int) -> np.ndarray:
    """合成测试信号：奈奎斯特突发（强采样间峰值）+ 正弦，立体声。"""
    n = sample_rate // 8
    t = np.arange(n) / sample_rate
    burst = 0.95 * np.sign(np.sin(2.0 * math.pi * (sample_rate / 2.0) * t))
    tone = 0.9 * np.sin(2.0 * math.pi * 440.0 * t)
    gate = (t > 0.03) & (t < 0.09)
    left = np.where(gate, burst, 0.3 * tone)
    right = np.where(gate, -burst, 0.3 * np.cos(2.0 * math.pi * 440.0 * t))
    return np.stack([left, right])


def determinism(signal=None, sample_rate: int = DEFAULT_SAMPLE_RATE,
                block_sizes: Sequence[int] = (1, 7, 16, 64), **kwargs) -> bool:
    """同输入处理任意次、任意分块（含小于前视窗口的块）逐样本相同。"""
    if signal is None:
        signal = _fixture(sample_rate)
    reference = process(signal, sample_rate, **kwargs)
    if not np.array_equal(reference, process(signal, sample_rate, **kwargs)):
        return False
    return all(np.array_equal(reference,
               process(signal, sample_rate, block_size=size, **kwargs))
               for size in block_sizes)


def audit(signal=None, sample_rate: int = DEFAULT_SAMPLE_RATE,
          dst_rate: int = 44100, ceiling: float = DEFAULT_CEILING) -> dict:
    """对六项修复做端到端审计，返回各项布尔结论与汇总。"""
    fixture = _fixture(sample_rate) if signal is None else signal
    report: dict = {}

    # 1) 真峰值发现采样间过冲：门控中频音在门沿的 sinc 重建过冲。
    tt = np.arange(sample_rate // 20) / sample_rate
    tone = np.sin(2.0 * math.pi * 2000.0 * tt)
    inter = np.zeros_like(tt)
    inter[40:200] = tone[40:200]
    report["true_peak_detects_inter_sample"] = (
        true_peak_detect(inter) > float(np.max(np.abs(inter))) + 1e-3)

    # 2) 前视以样本数表示。
    la = lookahead_samples(sample_rate, DEFAULT_LOOKAHEAD_MS)
    report["lookahead_in_samples"] = (
        isinstance(la, int) and la == round(DEFAULT_LOOKAHEAD_MS * 1e-3 * sample_rate))

    # 3) 跨块继承/分块无关。
    whole = process(fixture, sample_rate, ceiling=ceiling)
    report["gain_inherited_across_blocks"] = all(
        np.array_equal(whole, process(fixture, sample_rate, block_size=bs, ceiling=ceiling))
        for bs in (1, max(la - 1, 1), la, la + 1, 64))

    # 4) 多声道联动：两声道实际增益曲线完全相同。
    limited = process(fixture, sample_rate, ceiling=ceiling)
    if np.asarray(fixture).ndim == 2 and fixture.shape[0] > 1:
        nz = np.abs(fixture).min(axis=0) > _EPS
        gl = limited[0][nz] / fixture[0][nz]
        gr = limited[1][nz] / fixture[1][nz]
        report["multichannel_linked"] = bool(np.allclose(gl, gr, atol=1e-12))
    else:
        report["multichannel_linked"] = True

    # 5) 先重采样后限幅：结果合规且不劣于错误顺序。
    ordered = limit_after_resample(fixture, sample_rate, dst_rate, ceiling=ceiling)
    wrong_order = resample(process(fixture, sample_rate, ceiling=ceiling),
                           sample_rate, dst_rate)
    report["limited_after_resample"] = validate_limit(ordered, ceiling)
    report["resampler_overshoot_controlled"] = (
        report["limited_after_resample"]
        and true_peak_detect(ordered) <= true_peak_detect(wrong_order) + 1e-12)

    # 6) 释放平滑：释放严格一阶指数；攻击默认指数，仅门沿瞬态安全钳位。
    lim = TruePeakLimiter(sample_rate, ceiling=ceiling)
    lim.process_block(fixture)
    lim.flush()
    curves = list(lim.last_gain_curve)
    if lim.last_flush_curve.size:
        curves.append(lim.last_flush_curve)
    if curves:
        curve = np.concatenate(curves)
        diffs, gp = np.diff(curve), curve[:-1]
        rel, att = diffs > 0, diffs < 0
        release_ok = bool(np.all(
            diffs[rel] <= (1.0 - gp[rel]) * (1.0 - lim.release_coef) + 1e-12))
        exp_attack = gp[att] * (1.0 - lim.attack_coef)
        clamp_fraction = (float(((-diffs[att]) > exp_attack + 1e-12).mean())
                          if att.any() else 0.0)
        report["release_smoothed"] = release_ok and clamp_fraction < 0.5
    else:
        report["release_smoothed"] = True

    report["export_safe"] = validate_limit(
        export(limited, sample_rate=sample_rate), ceiling)
    if signal is not None:
        report["provided_signal_compliant"] = validate_limit(
            fallback(signal, ceiling=ceiling), ceiling)
    report["pass"] = bool(all(v for v in report.values() if isinstance(v, bool)))
    return report
