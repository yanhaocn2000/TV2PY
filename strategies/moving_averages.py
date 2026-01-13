"""
Moving Averages Collection - Python Conversion

包含多种移动平均指标:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double EMA)
- TEMA (Triple EMA)
- HMA (Hull Moving Average)
- VWMA (Volume Weighted MA)
"""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd


class SMA:
    """
    SMA (Simple Moving Average) - 简单移动平均

    公式: SMA = sum(close, period) / period
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 SMA"""
        data = np.asarray(data, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(self.period - 1, len(data)):
            result[i] = np.mean(data[i - self.period + 1:i + 1])
        return result


class EMA:
    """
    EMA (Exponential Moving Average) - 指数移动平均

    TradingView 公式:
        alpha = 2 / (period + 1)
        ema[period-1] = sma(data[:period])  # SMA 作为种子
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 EMA"""
        data = np.asarray(data, dtype=float)
        alpha = 2.0 / (self.period + 1)
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < self.period:
            return result

        # 使用 SMA 作为种子
        result[self.period - 1] = np.mean(data[:self.period])

        # 递归计算
        for i in range(self.period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result


class WMA:
    """
    WMA (Weighted Moving Average) - 加权移动平均

    公式: WMA = sum(weight[i] * data[i]) / sum(weights)
    其中 weight[i] = i + 1 (最近的权重最大)
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 WMA"""
        data = np.asarray(data, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        weights = np.arange(1, self.period + 1)
        weight_sum = np.sum(weights)

        for i in range(self.period - 1, len(data)):
            window = data[i - self.period + 1:i + 1]
            result[i] = np.sum(weights * window) / weight_sum

        return result


class DEMA:
    """
    DEMA (Double Exponential Moving Average) - 双重指数移动平均

    公式: DEMA = 2 * EMA(data) - EMA(EMA(data))

    DEMA 比 EMA 反应更快，滞后更少
    """

    def __init__(self, period: int = 20):
        self.period = period
        self._ema = EMA(period)

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 DEMA"""
        ema1 = self._ema.calculate(data)
        ema2 = self._ema.calculate(ema1)
        return 2 * ema1 - ema2


class TEMA:
    """
    TEMA (Triple Exponential Moving Average) - 三重指数移动平均

    公式: TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))

    TEMA 比 DEMA 反应更快
    """

    def __init__(self, period: int = 20):
        self.period = period
        self._ema = EMA(period)

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 TEMA"""
        ema1 = self._ema.calculate(data)
        ema2 = self._ema.calculate(ema1)
        ema3 = self._ema.calculate(ema2)
        return 3 * ema1 - 3 * ema2 + ema3


class HMA:
    """
    HMA (Hull Moving Average) - Hull 移动平均

    由 Alan Hull 发明，特点是极低的滞后。

    公式:
        wma1 = WMA(data, period/2)
        wma2 = WMA(data, period)
        raw_hma = 2 * wma1 - wma2
        HMA = WMA(raw_hma, sqrt(period))
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 HMA"""
        half_period = max(1, int(self.period / 2))
        sqrt_period = max(1, int(np.sqrt(self.period)))

        wma_half = WMA(half_period).calculate(data)
        wma_full = WMA(self.period).calculate(data)

        raw_hma = 2 * wma_half - wma_full

        return WMA(sqrt_period).calculate(raw_hma)


class VWMA:
    """
    VWMA (Volume Weighted Moving Average) - 成交量加权移动平均

    公式: VWMA = sum(close * volume, period) / sum(volume, period)
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """计算 VWMA"""
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        result = np.full_like(close, np.nan, dtype=float)

        for i in range(self.period - 1, len(close)):
            c = close[i - self.period + 1:i + 1]
            v = volume[i - self.period + 1:i + 1]
            vol_sum = np.sum(v)
            if vol_sum > 0:
                result[i] = np.sum(c * v) / vol_sum
            else:
                result[i] = np.mean(c)

        return result


class RMA:
    """
    RMA (Wilder's Smoothed Moving Average)

    也叫 SMMA 或 Wilder's smoothing
    TradingView 的 ta.rma() 和 ta.atr() 使用这个方法

    公式:
        alpha = 1 / period
        rma[period-1] = sma(data[:period])
        rma[i] = alpha * data[i] + (1 - alpha) * rma[i-1]
    """

    def __init__(self, period: int = 14):
        self.period = period

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """计算 RMA"""
        data = np.asarray(data, dtype=float)
        alpha = 1.0 / self.period
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < self.period:
            return result

        # 使用 SMA 作为种子
        result[self.period - 1] = np.mean(data[:self.period])

        # 递归计算
        for i in range(self.period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result


@dataclass
class MAResult:
    """移动平均结果"""
    values: np.ndarray
    ma_type: str
    period: int


class MovingAverageFactory:
    """移动平均工厂类"""

    @staticmethod
    def create(ma_type: str, period: int):
        """创建指定类型的移动平均"""
        ma_types = {
            "sma": SMA,
            "ema": EMA,
            "wma": WMA,
            "dema": DEMA,
            "tema": TEMA,
            "hma": HMA,
            "rma": RMA,
        }

        ma_class = ma_types.get(ma_type.lower())
        if ma_class is None:
            raise ValueError(f"Unknown MA type: {ma_type}")

        return ma_class(period)


class MovingAveragePyneCore:
    """PyneCore 兼容的移动平均实现"""

    def __init__(self, ma_type: str = "ema", period: int = 20):
        self.ma = MovingAverageFactory.create(ma_type, period)
        self.ma_type = ma_type
        self.period = period

    def __call__(self, data: pd.Series) -> pd.Series:
        """计算并返回 Series 格式的结果"""
        result = self.ma.calculate(data.values)
        return pd.Series(result, index=data.index, name=f"{self.ma_type}_{self.period}")


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_MA = '''
//@version=5
indicator("Moving Averages", overlay=true)

// 参数
length = input.int(20, "Length")
src = input.source(close, "Source")

// 各种移动平均
sma_val = ta.sma(src, length)
ema_val = ta.ema(src, length)
wma_val = ta.wma(src, length)
rma_val = ta.rma(src, length)

// DEMA
ema1 = ta.ema(src, length)
ema2 = ta.ema(ema1, length)
dema_val = 2 * ema1 - ema2

// TEMA
ema3 = ta.ema(ema2, length)
tema_val = 3 * ema1 - 3 * ema2 + ema3

// HMA
half_length = math.floor(length / 2)
sqrt_length = math.floor(math.sqrt(length))
hma_val = ta.wma(2 * ta.wma(src, half_length) - ta.wma(src, length), sqrt_length)

// 绘图
plot(sma_val, "SMA", color=color.red)
plot(ema_val, "EMA", color=color.blue)
plot(wma_val, "WMA", color=color.green)
plot(hma_val, "HMA", color=color.purple)
'''


def main():
    """测试移动平均指标"""
    print("=" * 60)
    print("Moving Averages - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 50

    # 模拟价格数据
    base_price = 100.0
    returns = np.random.randn(n) * 0.01
    close = base_price * np.cumprod(1 + returns)

    period = 10

    # 计算各种移动平均
    sma = SMA(period).calculate(close)
    ema = EMA(period).calculate(close)
    wma = WMA(period).calculate(close)
    dema = DEMA(period).calculate(close)
    tema = TEMA(period).calculate(close)
    hma = HMA(period).calculate(close)
    rma = RMA(period).calculate(close)

    # 显示结果
    print(f"\n周期: {period}")
    print(f"数据点数: {n}")
    print()

    print("最后 10 个数据点:")
    print("-" * 100)
    print(f"{'Bar':<5} {'Close':<8} {'SMA':<8} {'EMA':<8} {'WMA':<8} {'DEMA':<8} {'TEMA':<8} {'HMA':<8} {'RMA':<8}")
    print("-" * 100)

    for i in range(n - 10, n):
        def fmt(v):
            return f"{v:.2f}" if not np.isnan(v) else "NaN"

        print(f"{i:<5} {close[i]:<8.2f} {fmt(sma[i]):<8} {fmt(ema[i]):<8} "
              f"{fmt(wma[i]):<8} {fmt(dema[i]):<8} {fmt(tema[i]):<8} "
              f"{fmt(hma[i]):<8} {fmt(rma[i]):<8}")

    print()
    print("滞后分析 (最后值与最近收盘价的差异):")
    last_close = close[-1]
    print(f"  Close: {last_close:.2f}")
    print(f"  SMA:   {sma[-1]:.2f} (diff: {sma[-1] - last_close:+.2f})")
    print(f"  EMA:   {ema[-1]:.2f} (diff: {ema[-1] - last_close:+.2f})")
    print(f"  HMA:   {hma[-1]:.2f} (diff: {hma[-1] - last_close:+.2f})  <- 最小滞后")
    print(f"  TEMA:  {tema[-1]:.2f} (diff: {tema[-1] - last_close:+.2f})")


if __name__ == "__main__":
    main()
