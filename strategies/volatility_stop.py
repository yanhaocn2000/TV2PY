"""
Volatility Stop - Python Conversion

核心算法:
    使用 ATR 计算动态止损位

    上涨趋势:
        Stop = max(Stop[1], close - mult * ATR)
    下跌趋势:
        Stop = min(Stop[1], close + mult * ATR)

    趋势反转:
        close > Stop (下跌趋势) → 反转为上涨
        close < Stop (上涨趋势) → 反转为下跌

类似于: Chandelier Exit, ATR Trailing Stop
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VolatilityStopResult:
    """Volatility Stop 计算结果"""
    vstop: np.ndarray          # Volatility Stop 值
    trend: np.ndarray          # 趋势方向 (1=上涨, -1=下跌)
    atr: np.ndarray            # ATR 值
    buy_signal: np.ndarray     # 买入信号
    sell_signal: np.ndarray    # 卖出信号


class VolatilityStop:
    """
    Volatility Stop

    基于 ATR 的动态止损/趋势跟踪指标。

    Parameters:
        period: ATR 周期 - 默认 20
        mult: ATR 乘数 - 默认 2.0
    """

    def __init__(
        self,
        period: int = 20,
        mult: float = 2.0,
    ):
        self.period = period
        self.mult = mult

    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """计算 ATR (使用 RMA)"""
        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        alpha = 1.0 / self.period
        atr = np.full(n, np.nan)
        if n >= self.period:
            atr[self.period - 1] = np.mean(tr[:self.period])
            for i in range(self.period, n):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        return atr

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        src: Optional[np.ndarray] = None,
    ) -> VolatilityStopResult:
        """
        计算 Volatility Stop

        Parameters:
            high, low, close: OHLC 数据
            src: 源数据 (默认使用 close)
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        src = close if src is None else np.asarray(src, dtype=float)
        n = len(close)

        # 计算 ATR
        atr = self._atr(high, low, close)

        # 初始化
        vstop = np.full(n, np.nan)
        trend = np.zeros(n, dtype=int)

        # 找到第一个有效位置
        start_idx = self.period - 1

        if start_idx < n:
            # 初始化
            vstop[start_idx] = src[start_idx] - self.mult * atr[start_idx]
            trend[start_idx] = 1  # 假设初始为上涨

        # 主计算循环
        for i in range(start_idx + 1, n):
            if np.isnan(atr[i]):
                continue

            atr_stop = self.mult * atr[i]
            prev_vstop = vstop[i - 1]
            prev_trend = trend[i - 1]

            if prev_trend == 1:  # 上涨趋势
                # 止损只能上移
                new_stop = max(prev_vstop, src[i] - atr_stop)

                if src[i] < prev_vstop:
                    # 趋势反转为下跌
                    trend[i] = -1
                    vstop[i] = src[i] + atr_stop
                else:
                    trend[i] = 1
                    vstop[i] = new_stop

            else:  # 下跌趋势
                # 止损只能下移
                new_stop = min(prev_vstop, src[i] + atr_stop)

                if src[i] > prev_vstop:
                    # 趋势反转为上涨
                    trend[i] = 1
                    vstop[i] = src[i] - atr_stop
                else:
                    trend[i] = -1
                    vstop[i] = new_stop

        # 信号检测
        buy_signal = np.zeros(n, dtype=bool)
        sell_signal = np.zeros(n, dtype=bool)

        for i in range(start_idx + 1, n):
            if trend[i] == 1 and trend[i - 1] == -1:
                buy_signal[i] = True
            elif trend[i] == -1 and trend[i - 1] == 1:
                sell_signal[i] = True

        return VolatilityStopResult(
            vstop=vstop,
            trend=trend,
            atr=atr,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
        )


class VolatilityStopPyneCore:
    """PyneCore 兼容的 Volatility Stop 实现"""

    def __init__(self, period: int = 20, mult: float = 2.0):
        self.indicator = VolatilityStop(period=period, mult=mult)

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
            "vstop": result.vstop,
            "trend": result.trend,
            "buy": result.buy_signal,
            "sell": result.sell_signal,
        }, index=close.index)


PINE_SCRIPT_VSTOP = '''
//@version=5
indicator("Volatility Stop", overlay=true)

length = input.int(20, "ATR Period")
mult = input.float(2.0, "ATR Multiplier")
src = input.source(close, "Source")

atr_val = ta.atr(length)

var float vstop = na
var int trend = 1

atr_stop = mult * atr_val

if trend == 1
    vstop := math.max(nz(vstop[1]), src - atr_stop)
    if src < nz(vstop[1])
        trend := -1
        vstop := src + atr_stop
else
    vstop := math.min(nz(vstop[1]), src + atr_stop)
    if src > nz(vstop[1])
        trend := 1
        vstop := src - atr_stop

plot(vstop, "V-Stop", color=trend == 1 ? color.green : color.red, linewidth=2)
'''


def main():
    print("=" * 60)
    print("Volatility Stop - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 100

    # 带趋势的数据
    base_price = 100.0
    trend = np.linspace(0, 15, n)
    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5

    indicator = VolatilityStop()
    result = indicator.calculate(high, low, close)

    print(f"\n参数: period={indicator.period}, mult={indicator.mult}")
    print(f"\n最后 15 个数据点:")
    print("-" * 60)
    print(f"{'Bar':<5} {'Close':<10} {'V-Stop':<10} {'Trend':<8} {'Signal':<10}")
    print("-" * 60)

    for i in range(n - 15, n):
        vs_str = f"{result.vstop[i]:.2f}" if not np.isnan(result.vstop[i]) else "NaN"
        trend_str = "UP" if result.trend[i] == 1 else "DOWN"
        signal = ""
        if result.buy_signal[i]:
            signal = "BUY"
        elif result.sell_signal[i]:
            signal = "SELL"
        print(f"{i:<5} {close[i]:<10.2f} {vs_str:<10} {trend_str:<8} {signal:<10}")

    print(f"\n信号统计:")
    print(f"  买入信号: {np.sum(result.buy_signal)}")
    print(f"  卖出信号: {np.sum(result.sell_signal)}")


if __name__ == "__main__":
    main()
