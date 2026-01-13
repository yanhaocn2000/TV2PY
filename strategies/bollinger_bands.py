"""
Bollinger Bands - Python Conversion

TradingView 内置指标: ta.bb()

核心算法:
    basis = sma(close, period)
    dev = mult * stdev(close, period)  # population stdev, ddof=0
    upper = basis + dev
    lower = basis - dev

信号:
    - 价格触及上轨: 可能超买
    - 价格触及下轨: 可能超卖
    - 带宽收窄: 即将爆发 (squeeze)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BollingerBandsResult:
    """布林带计算结果"""
    upper: np.ndarray          # 上轨
    middle: np.ndarray         # 中轨 (SMA)
    lower: np.ndarray          # 下轨
    bandwidth: np.ndarray      # 带宽
    percent_b: np.ndarray      # %B 指标
    stdev: np.ndarray          # 标准差


class BollingerBands:
    """
    Bollinger Bands (布林带)

    由 John Bollinger 发明，是最常用的波动率指标之一。

    Parameters:
        period: SMA 周期 - 默认 20
        mult: 标准差乘数 - 默认 2.0
    """

    def __init__(
        self,
        period: int = 20,
        mult: float = 2.0,
    ):
        self.period = period
        self.mult = mult

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def _stdev(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算标准差 (population, ddof=0)

        TradingView 使用 population standard deviation
        """
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.std(data[i - period + 1:i + 1], ddof=0)
        return result

    def calculate(self, close: np.ndarray) -> BollingerBandsResult:
        """
        计算布林带

        算法步骤:
        1. 计算中轨 (SMA)
        2. 计算标准差
        3. 计算上轨和下轨
        4. 计算带宽和 %B

        Returns:
            BollingerBandsResult 包含所有计算结果
        """
        close = np.asarray(close, dtype=float)

        # Step 1: 计算中轨
        middle = self._sma(close, self.period)

        # Step 2: 计算标准差
        stdev = self._stdev(close, self.period)

        # Step 3: 计算上下轨
        dev = self.mult * stdev
        upper = middle + dev
        lower = middle - dev

        # Step 4: 计算带宽
        # Bandwidth = (Upper - Lower) / Middle * 100
        bandwidth = np.where(middle != 0, (upper - lower) / middle * 100, 0)

        # Step 5: 计算 %B
        # %B = (Price - Lower) / (Upper - Lower)
        band_width = upper - lower
        percent_b = np.where(band_width != 0, (close - lower) / band_width, 0.5)

        return BollingerBandsResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
            stdev=stdev,
        )

    def get_signals(self, close: np.ndarray, result: BollingerBandsResult) -> dict:
        """
        获取交易信号

        - touch_upper: 价格触及上轨
        - touch_lower: 价格触及下轨
        - squeeze: 带宽收窄 (低于平均带宽的 50%)
        """
        n = len(close)

        # 触及上下轨
        touch_upper = close >= result.upper
        touch_lower = close <= result.lower

        # Squeeze 检测 (带宽低于移动平均的 50%)
        bw_ma = self._sma(result.bandwidth, self.period)
        squeeze = result.bandwidth < (bw_ma * 0.5)

        return {
            "touch_upper": touch_upper,
            "touch_lower": touch_lower,
            "squeeze": squeeze,
            "overbought": result.percent_b > 1.0,
            "oversold": result.percent_b < 0.0,
        }


class BollingerBandsStrategy:
    """
    布林带交易策略

    策略1: 均值回归
        - 买入: 价格触及下轨后反弹
        - 卖出: 价格触及上轨后回落

    策略2: 突破
        - 买入: 价格突破上轨 (带 squeeze)
        - 卖出: 价格跌破下轨 (带 squeeze)
    """

    def __init__(
        self,
        period: int = 20,
        mult: float = 2.0,
        strategy_type: str = "mean_reversion",
    ):
        self.indicator = BollingerBands(period=period, mult=mult)
        self.strategy_type = strategy_type

    def generate_signals(self, close: np.ndarray) -> pd.DataFrame:
        """生成交易信号 DataFrame"""
        result = self.indicator.calculate(close)
        signals = self.indicator.get_signals(close, result)

        n = len(close)
        signal = np.zeros(n, dtype=int)

        if self.strategy_type == "mean_reversion":
            # 均值回归策略
            for i in range(1, n):
                if signals["touch_lower"][i-1] and close[i] > close[i-1]:
                    signal[i] = 1  # 买入
                elif signals["touch_upper"][i-1] and close[i] < close[i-1]:
                    signal[i] = -1  # 卖出
        else:
            # 突破策略
            for i in range(1, n):
                if signals["squeeze"][i-1] and close[i] > result.upper[i-1]:
                    signal[i] = 1  # 买入
                elif signals["squeeze"][i-1] and close[i] < result.lower[i-1]:
                    signal[i] = -1  # 卖出

        return pd.DataFrame({
            "upper": result.upper,
            "middle": result.middle,
            "lower": result.lower,
            "bandwidth": result.bandwidth,
            "percent_b": result.percent_b,
            "signal": signal,
        })


class BollingerBandsPyneCore:
    """PyneCore 兼容的布林带实现"""

    def __init__(
        self,
        period: int = 20,
        mult: float = 2.0,
    ):
        self.indicator = BollingerBands(period=period, mult=mult)

    def __call__(self, close: pd.Series) -> pd.DataFrame:
        """计算并返回 DataFrame 格式的结果"""
        result = self.indicator.calculate(close.values)

        return pd.DataFrame({
            "upper": result.upper,
            "middle": result.middle,
            "lower": result.lower,
            "bandwidth": result.bandwidth,
            "percent_b": result.percent_b,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_BOLLINGER = '''
//@version=5
indicator("Bollinger Bands", overlay=true)

// 参数
length = input.int(20, "Length")
mult = input.float(2.0, "StdDev")
src = input.source(close, "Source")

// 计算
basis = ta.sma(src, length)
dev = mult * ta.stdev(src, length)
upper = basis + dev
lower = basis - dev

// 带宽和 %B
bandwidth = (upper - lower) / basis * 100
percentB = (src - lower) / (upper - lower)

// 绘图
plot(basis, "Basis", color=color.orange)
p1 = plot(upper, "Upper", color=color.blue)
p2 = plot(lower, "Lower", color=color.blue)
fill(p1, p2, color=color.new(color.blue, 90))
'''


def main():
    """测试布林带指标"""
    print("=" * 60)
    print("Bollinger Bands - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟价格数据
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)

    # 计算布林带
    indicator = BollingerBands()
    result = indicator.calculate(close)

    # 显示结果
    print(f"\n参数: period={indicator.period}, mult={indicator.mult}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'Upper':<10} {'Middle':<10} {'Lower':<10} {'%B':<8}")
    print("-" * 80)

    for i in range(n - 15, n):
        upper_str = f"{result.upper[i]:.2f}" if not np.isnan(result.upper[i]) else "NaN"
        mid_str = f"{result.middle[i]:.2f}" if not np.isnan(result.middle[i]) else "NaN"
        lower_str = f"{result.lower[i]:.2f}" if not np.isnan(result.lower[i]) else "NaN"
        pb_str = f"{result.percent_b[i]:.2f}" if not np.isnan(result.percent_b[i]) else "NaN"

        print(f"{i:<5} {close[i]:<10.2f} {upper_str:<10} {mid_str:<10} {lower_str:<10} {pb_str:<8}")

    print()
    signals = indicator.get_signals(close, result)
    print("信号统计:")
    print(f"  触及上轨: {np.sum(signals['touch_upper'])}")
    print(f"  触及下轨: {np.sum(signals['touch_lower'])}")
    print(f"  Squeeze: {np.sum(signals['squeeze'])}")


if __name__ == "__main__":
    main()
