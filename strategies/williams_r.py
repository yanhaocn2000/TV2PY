"""
Williams %R - Python Conversion

TradingView 内置指标: ta.wpr()

核心算法:
    %R = (highest_high - close) / (highest_high - lowest_low) * -100

特点:
    - 范围: -100 到 0
    - 超买: > -20
    - 超卖: < -80
    - 与 Stochastic 类似但方向相反
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class WilliamsRResult:
    """Williams %R 计算结果"""
    wpr: np.ndarray            # Williams %R 值
    overbought: np.ndarray     # 超买
    oversold: np.ndarray       # 超卖


class WilliamsRIndicator:
    """
    Williams %R (Williams Percent Range)

    由 Larry Williams 发明，是一个动量指标。

    Parameters:
        period: 回望周期 - 默认 14
        overbought: 超买阈值 - 默认 -20
        oversold: 超卖阈值 - 默认 -80
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = -20,
        oversold: float = -80,
    ):
        self.period = period
        self.overbought_level = overbought
        self.oversold_level = oversold

    def _highest(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.max(data[i - period + 1:i + 1])
        return result

    def _lowest(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.min(data[i - period + 1:i + 1])
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> WilliamsRResult:
        """计算 Williams %R"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        highest_high = self._highest(high, self.period)
        lowest_low = self._lowest(low, self.period)

        range_hl = highest_high - lowest_low
        wpr = np.where(
            range_hl != 0,
            (highest_high - close) / range_hl * -100,
            -50
        )

        overbought = wpr > self.overbought_level
        oversold = wpr < self.oversold_level

        return WilliamsRResult(
            wpr=wpr,
            overbought=overbought,
            oversold=oversold,
        )


class WilliamsRPyneCore:
    """PyneCore 兼容的 Williams %R 实现"""

    def __init__(self, period: int = 14):
        self.indicator = WilliamsRIndicator(period=period)

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
            "wpr": result.wpr,
            "overbought": result.overbought,
            "oversold": result.oversold,
        }, index=close.index)


PINE_SCRIPT_WILLIAMS_R = '''
//@version=5
indicator("Williams %R", overlay=false)

length = input.int(14, "Period")
wpr = ta.wpr(length)

plot(wpr, "Williams %R", color=color.blue)
hline(-20, "Overbought", color=color.red)
hline(-80, "Oversold", color=color.green)
'''


def main():
    print("=" * 60)
    print("Williams %R - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 50

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

    indicator = WilliamsRIndicator()
    result = indicator.calculate(high, low, close)

    print(f"\n参数: period={indicator.period}")
    print(f"\n最后 10 个数据点:")
    print("-" * 50)
    print(f"{'Bar':<5} {'Close':<10} {'%R':<10} {'Zone':<12}")
    print("-" * 50)

    for i in range(n - 10, n):
        wpr_str = f"{result.wpr[i]:.2f}" if not np.isnan(result.wpr[i]) else "NaN"
        if result.overbought[i]:
            zone = "Overbought"
        elif result.oversold[i]:
            zone = "Oversold"
        else:
            zone = "Neutral"
        print(f"{i:<5} {close[i]:<10.2f} {wpr_str:<10} {zone:<12}")


if __name__ == "__main__":
    main()
