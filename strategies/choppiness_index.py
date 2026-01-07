"""
Choppiness Index - Python Conversion

核心算法:
    CI = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)

范围: 0 - 100
    - CI > 61.8: 市场横盘震荡 (choppy)
    - CI < 38.2: 市场趋势明确 (trending)

用途:
    - 识别震荡市场 vs 趋势市场
    - 帮助选择策略类型 (趋势跟踪 vs 区间交易)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ChoppinessResult:
    """Choppiness Index 计算结果"""
    chop: np.ndarray           # Choppiness Index 值
    is_choppy: np.ndarray      # 震荡市场
    is_trending: np.ndarray    # 趋势市场


class ChoppinessIndex:
    """
    Choppiness Index

    用于判断市场是处于震荡还是趋势状态。

    Parameters:
        period: 周期 - 默认 14
        choppy_threshold: 震荡阈值 - 默认 61.8
        trending_threshold: 趋势阈值 - 默认 38.2
    """

    def __init__(
        self,
        period: int = 14,
        choppy_threshold: float = 61.8,
        trending_threshold: float = 38.2,
    ):
        self.period = period
        self.choppy_threshold = choppy_threshold
        self.trending_threshold = trending_threshold

    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """计算 ATR"""
        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # 使用 RMA
        alpha = 1.0 / period
        atr = np.full(n, np.nan)
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return atr

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> ChoppinessResult:
        """计算 Choppiness Index"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 计算 True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # 计算 Choppiness Index
        chop = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            # Sum of TR
            tr_sum = np.sum(tr[i - self.period + 1:i + 1])

            # Highest High - Lowest Low
            highest_high = np.max(high[i - self.period + 1:i + 1])
            lowest_low = np.min(low[i - self.period + 1:i + 1])
            hl_range = highest_high - lowest_low

            if hl_range > 0:
                chop[i] = 100 * np.log10(tr_sum / hl_range) / np.log10(self.period)

        # 市场状态判断
        is_choppy = chop > self.choppy_threshold
        is_trending = chop < self.trending_threshold

        return ChoppinessResult(
            chop=chop,
            is_choppy=is_choppy,
            is_trending=is_trending,
        )


class ChoppinessPyneCore:
    """PyneCore 兼容的 Choppiness Index 实现"""

    def __init__(self, period: int = 14):
        self.indicator = ChoppinessIndex(period=period)

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
            "chop": result.chop,
            "is_choppy": result.is_choppy,
            "is_trending": result.is_trending,
        }, index=close.index)


PINE_SCRIPT_CHOPPINESS = '''
//@version=5
indicator("Choppiness Index", overlay=false)

length = input.int(14, "Period")

// True Range
tr = ta.tr(true)

// Choppiness Index
atr_sum = math.sum(tr, length)
highest = ta.highest(high, length)
lowest = ta.lowest(low, length)
chop = 100 * math.log10(atr_sum / (highest - lowest)) / math.log10(length)

plot(chop, "CHOP", color=color.blue)
hline(61.8, "Choppy", color=color.red)
hline(38.2, "Trending", color=color.green)

bgcolor(chop > 61.8 ? color.new(color.red, 90) : chop < 38.2 ? color.new(color.green, 90) : na)
'''


def main():
    print("=" * 60)
    print("Choppiness Index - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 50

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

    indicator = ChoppinessIndex()
    result = indicator.calculate(high, low, close)

    print(f"\n参数: period={indicator.period}")
    print(f"\n最后 10 个数据点:")
    print("-" * 50)
    print(f"{'Bar':<5} {'CHOP':<12} {'Market':<15}")
    print("-" * 50)

    for i in range(n - 10, n):
        chop_str = f"{result.chop[i]:.2f}" if not np.isnan(result.chop[i]) else "NaN"
        if result.is_choppy[i]:
            market = "Choppy"
        elif result.is_trending[i]:
            market = "Trending"
        else:
            market = "Neutral"
        print(f"{i:<5} {chop_str:<12} {market:<15}")


if __name__ == "__main__":
    main()
