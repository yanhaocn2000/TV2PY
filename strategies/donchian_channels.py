"""
Donchian Channels - Python Conversion

TradingView 内置指标

核心算法:
    upper = highest(high, period)
    lower = lowest(low, period)
    basis = (upper + lower) / 2

信号:
    - 价格突破上轨: 可能的突破买入信号
    - 价格跌破下轨: 可能的突破卖出信号
    - 价格在通道内: 区间震荡

用途:
    - 趋势突破系统 (海龟交易法则)
    - 波动率衡量
    - 支撑/阻力位
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DonchianResult:
    """Donchian Channels 计算结果"""
    upper: np.ndarray          # 上轨
    lower: np.ndarray          # 下轨
    basis: np.ndarray          # 中轨
    width: np.ndarray          # 通道宽度
    width_percent: np.ndarray  # 通道宽度百分比


class DonchianChannels:
    """
    Donchian Channels (唐奇安通道)

    由 Richard Donchian 发明，是最早的趋势跟踪系统之一。
    海龟交易法则就是基于 Donchian 通道。

    Parameters:
        period: 回望周期 - 默认 20
    """

    def __init__(self, period: int = 20):
        self.period = period

    def _highest(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算滚动最高值"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.max(data[i - period + 1:i + 1])
        return result

    def _lowest(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算滚动最低值"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.min(data[i - period + 1:i + 1])
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> DonchianResult:
        """
        计算 Donchian Channels

        Returns:
            DonchianResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # 计算上下轨
        upper = self._highest(high, self.period)
        lower = self._lowest(low, self.period)

        # 计算中轨
        basis = (upper + lower) / 2

        # 计算通道宽度
        width = upper - lower
        width_percent = np.where(basis != 0, width / basis * 100, 0)

        return DonchianResult(
            upper=upper,
            lower=lower,
            basis=basis,
            width=width,
            width_percent=width_percent,
        )

    def get_signals(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        result: DonchianResult,
    ) -> dict:
        """
        获取交易信号

        海龟交易法则信号:
        - 买入: 价格突破 20 日上轨
        - 卖出: 价格跌破 20 日下轨
        - 止损: 价格跌破 10 日下轨 (多头) / 突破 10 日上轨 (空头)
        """
        n = len(close)

        # 突破上轨
        breakout_up = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(result.upper[i-1]):
                breakout_up[i] = high[i] > result.upper[i-1]

        # 跌破下轨
        breakout_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(result.lower[i-1]):
                breakout_down[i] = low[i] < result.lower[i-1]

        # 价格位置
        above_upper = close > result.upper
        below_lower = close < result.lower

        return {
            "breakout_up": breakout_up,
            "breakout_down": breakout_down,
            "above_upper": above_upper,
            "below_lower": below_lower,
        }


class TurtleTradingSystem:
    """
    海龟交易法则 - 基于 Donchian Channels

    系统1 (短期):
        - 入场: 20 日突破
        - 出场: 10 日反向突破

    系统2 (长期):
        - 入场: 55 日突破
        - 出场: 20 日反向突破
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
    ):
        self.entry_channel = DonchianChannels(entry_period)
        self.exit_channel = DonchianChannels(exit_period)

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> dict:
        """计算海龟交易信号"""
        entry_result = self.entry_channel.calculate(high, low, close)
        exit_result = self.exit_channel.calculate(high, low, close)

        n = len(close)

        # 入场信号
        long_entry = np.zeros(n, dtype=bool)
        short_entry = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(entry_result.upper[i-1]):
                long_entry[i] = high[i] > entry_result.upper[i-1]
            if not np.isnan(entry_result.lower[i-1]):
                short_entry[i] = low[i] < entry_result.lower[i-1]

        # 出场信号
        long_exit = np.zeros(n, dtype=bool)
        short_exit = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(exit_result.lower[i-1]):
                long_exit[i] = low[i] < exit_result.lower[i-1]
            if not np.isnan(exit_result.upper[i-1]):
                short_exit[i] = high[i] > exit_result.upper[i-1]

        return {
            "entry_upper": entry_result.upper,
            "entry_lower": entry_result.lower,
            "exit_upper": exit_result.upper,
            "exit_lower": exit_result.lower,
            "long_entry": long_entry,
            "short_entry": short_entry,
            "long_exit": long_exit,
            "short_exit": short_exit,
        }


class DonchianPyneCore:
    """PyneCore 兼容的 Donchian Channels 实现"""

    def __init__(self, period: int = 20):
        self.indicator = DonchianChannels(period=period)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(
            high.values, low.values, close.values
        )
        return pd.DataFrame({
            "upper": result.upper,
            "basis": result.basis,
            "lower": result.lower,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_DONCHIAN = '''
//@version=5
indicator("Donchian Channels", overlay=true)

// 参数
length = input.int(20, "Period")

// 计算
upper = ta.highest(high, length)
lower = ta.lowest(low, length)
basis = math.avg(upper, lower)

// 绘图
plot(upper, "Upper", color=color.blue)
plot(basis, "Basis", color=color.orange)
plot(lower, "Lower", color=color.blue)
fill(plot(upper), plot(lower), color=color.new(color.blue, 90))
'''


def main():
    """测试 Donchian Channels 指标"""
    print("=" * 60)
    print("Donchian Channels - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟价格数据
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

    # 计算 Donchian Channels
    indicator = DonchianChannels()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: period={indicator.period}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'Upper':<10} {'Basis':<10} {'Lower':<10} {'Width%':<10}")
    print("-" * 80)

    for i in range(n - 15, n):
        upper_str = f"{result.upper[i]:.2f}" if not np.isnan(result.upper[i]) else "NaN"
        basis_str = f"{result.basis[i]:.2f}" if not np.isnan(result.basis[i]) else "NaN"
        lower_str = f"{result.lower[i]:.2f}" if not np.isnan(result.lower[i]) else "NaN"
        width_str = f"{result.width_percent[i]:.2f}%" if not np.isnan(result.width_percent[i]) else "NaN"

        print(f"{i:<5} {close[i]:<10.2f} {upper_str:<10} {basis_str:<10} {lower_str:<10} {width_str:<10}")

    print()
    signals = indicator.get_signals(high, low, close, result)
    print("信号统计:")
    print(f"  向上突破: {np.sum(signals['breakout_up'])}")
    print(f"  向下突破: {np.sum(signals['breakout_down'])}")

    print()
    print("海龟交易系统测试:")
    turtle = TurtleTradingSystem(entry_period=20, exit_period=10)
    turtle_signals = turtle.calculate(high, low, close)
    print(f"  多头入场信号: {np.sum(turtle_signals['long_entry'])}")
    print(f"  空头入场信号: {np.sum(turtle_signals['short_entry'])}")
    print(f"  多头出场信号: {np.sum(turtle_signals['long_exit'])}")
    print(f"  空头出场信号: {np.sum(turtle_signals['short_exit'])}")


if __name__ == "__main__":
    main()
