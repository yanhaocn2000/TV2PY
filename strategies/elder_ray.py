"""
Elder Ray Index - Python Conversion

由 Alexander Elder 发明

核心算法:
    Bull Power = High - EMA(close, period)
    Bear Power = Low - EMA(close, period)

信号:
    - Bull Power > 0 且上升: 看涨
    - Bear Power < 0 且下降: 看跌
    - 结合趋势使用 (EMA 方向)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ElderRayResult:
    """Elder Ray 计算结果"""
    bull_power: np.ndarray     # 牛市力量
    bear_power: np.ndarray     # 熊市力量
    ema: np.ndarray            # EMA 值
    trend: np.ndarray          # 趋势方向


class ElderRayIndicator:
    """
    Elder Ray Index

    由 Alexander Elder 发明，用于衡量买卖双方力量。

    Parameters:
        period: EMA 周期 - 默认 13
    """

    def __init__(self, period: int = 13):
        self.period = period

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> ElderRayResult:
        """计算 Elder Ray Index"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 计算 EMA
        ema = self._ema(close, self.period)

        # 计算 Bull Power 和 Bear Power
        bull_power = high - ema
        bear_power = low - ema

        # 趋势方向
        trend = np.zeros(n, dtype=int)
        for i in range(1, n):
            if not np.isnan(ema[i]) and not np.isnan(ema[i-1]):
                if ema[i] > ema[i-1]:
                    trend[i] = 1
                elif ema[i] < ema[i-1]:
                    trend[i] = -1

        return ElderRayResult(
            bull_power=bull_power,
            bear_power=bear_power,
            ema=ema,
            trend=trend,
        )

    def get_signals(self, result: ElderRayResult) -> dict:
        """
        获取交易信号

        买入条件:
        1. EMA 上升 (趋势向上)
        2. Bear Power < 0 但在上升
        3. Bull Power 的最近高点高于前一个高点

        卖出条件:
        1. EMA 下降 (趋势向下)
        2. Bull Power > 0 但在下降
        """
        n = len(result.bull_power)
        buy_signal = np.zeros(n, dtype=bool)
        sell_signal = np.zeros(n, dtype=bool)

        for i in range(2, n):
            # 买入: 趋势向上 + Bear Power 上升
            if (result.trend[i] == 1 and
                result.bear_power[i] < 0 and
                result.bear_power[i] > result.bear_power[i-1]):
                buy_signal[i] = True

            # 卖出: 趋势向下 + Bull Power 下降
            if (result.trend[i] == -1 and
                result.bull_power[i] > 0 and
                result.bull_power[i] < result.bull_power[i-1]):
                sell_signal[i] = True

        return {
            "buy": buy_signal,
            "sell": sell_signal,
        }


class ElderRayPyneCore:
    """PyneCore 兼容的 Elder Ray 实现"""

    def __init__(self, period: int = 13):
        self.indicator = ElderRayIndicator(period=period)

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
            "bull_power": result.bull_power,
            "bear_power": result.bear_power,
            "ema": result.ema,
        }, index=close.index)


PINE_SCRIPT_ELDER_RAY = '''
//@version=5
indicator("Elder Ray Index", overlay=false)

length = input.int(13, "EMA Length")

ema_val = ta.ema(close, length)
bull_power = high - ema_val
bear_power = low - ema_val

plot(bull_power, "Bull Power", color=color.green, style=plot.style_histogram)
plot(bear_power, "Bear Power", color=color.red, style=plot.style_histogram)
hline(0, "Zero", color=color.gray)
'''


def main():
    print("=" * 60)
    print("Elder Ray Index - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 50

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

    indicator = ElderRayIndicator()
    result = indicator.calculate(high, low, close)

    print(f"\n参数: period={indicator.period}")
    print(f"\n最后 10 个数据点:")
    print("-" * 60)
    print(f"{'Bar':<5} {'Bull':<12} {'Bear':<12} {'Trend':<8}")
    print("-" * 60)

    for i in range(n - 10, n):
        bull_str = f"{result.bull_power[i]:.4f}" if not np.isnan(result.bull_power[i]) else "NaN"
        bear_str = f"{result.bear_power[i]:.4f}" if not np.isnan(result.bear_power[i]) else "NaN"
        trend_str = "UP" if result.trend[i] == 1 else "DOWN" if result.trend[i] == -1 else "-"
        print(f"{i:<5} {bull_str:<12} {bear_str:<12} {trend_str:<8}")


if __name__ == "__main__":
    main()
