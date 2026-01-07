"""
Awesome Oscillator (AO) - Python Conversion

TradingView 内置指标

核心算法:
    AO = SMA(median_price, 5) - SMA(median_price, 34)
    median_price = (high + low) / 2

信号:
    - 零线穿越
    - 茶碟形态 (Saucer)
    - 双峰形态 (Twin Peaks)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AOResult:
    """Awesome Oscillator 计算结果"""
    ao: np.ndarray             # AO 值
    color: np.ndarray          # 颜色 (1=绿色上涨, -1=红色下跌)
    zero_cross_up: np.ndarray  # 上穿零线
    zero_cross_down: np.ndarray  # 下穿零线


class AwesomeOscillator:
    """
    Awesome Oscillator (AO)

    由 Bill Williams 发明，用于衡量市场动量。

    Parameters:
        fast_period: 快速 SMA 周期 - 默认 5
        slow_period: 慢速 SMA 周期 - 默认 34
    """

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 34,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> AOResult:
        """计算 Awesome Oscillator"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        n = len(high)

        # 中间价
        median_price = (high + low) / 2

        # 计算 AO
        fast_sma = self._sma(median_price, self.fast_period)
        slow_sma = self._sma(median_price, self.slow_period)
        ao = fast_sma - slow_sma

        # 颜色 (基于 AO 变化)
        color = np.zeros(n, dtype=int)
        for i in range(1, n):
            if not np.isnan(ao[i]) and not np.isnan(ao[i-1]):
                color[i] = 1 if ao[i] > ao[i-1] else -1

        # 零线穿越
        zero_cross_up = np.zeros(n, dtype=bool)
        zero_cross_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(ao[i]) and not np.isnan(ao[i-1]):
                zero_cross_up[i] = ao[i] > 0 and ao[i-1] <= 0
                zero_cross_down[i] = ao[i] < 0 and ao[i-1] >= 0

        return AOResult(
            ao=ao,
            color=color,
            zero_cross_up=zero_cross_up,
            zero_cross_down=zero_cross_down,
        )

    def detect_saucer(self, result: AOResult) -> np.ndarray:
        """
        检测茶碟形态

        看涨茶碟: AO > 0, 三根柱子, 中间柱子最低且为红色, 第三根为绿色
        看跌茶碟: AO < 0, 三根柱子, 中间柱子最高且为绿色, 第三根为红色
        """
        n = len(result.ao)
        bullish_saucer = np.zeros(n, dtype=bool)
        bearish_saucer = np.zeros(n, dtype=bool)

        for i in range(2, n):
            if (not np.isnan(result.ao[i]) and not np.isnan(result.ao[i-1])
                and not np.isnan(result.ao[i-2])):

                # 看涨茶碟
                if (result.ao[i] > 0 and result.ao[i-1] > 0 and result.ao[i-2] > 0 and
                    result.ao[i-1] < result.ao[i-2] and result.ao[i] > result.ao[i-1] and
                    result.color[i-1] == -1 and result.color[i] == 1):
                    bullish_saucer[i] = True

                # 看跌茶碟
                if (result.ao[i] < 0 and result.ao[i-1] < 0 and result.ao[i-2] < 0 and
                    result.ao[i-1] > result.ao[i-2] and result.ao[i] < result.ao[i-1] and
                    result.color[i-1] == 1 and result.color[i] == -1):
                    bearish_saucer[i] = True

        return bullish_saucer, bearish_saucer


class AOPyneCore:
    """PyneCore 兼容的 AO 实现"""

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 34,
    ):
        self.indicator = AwesomeOscillator(
            fast_period=fast_period, slow_period=slow_period
        )

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(high.values, low.values)
        return pd.DataFrame({
            "ao": result.ao,
            "color": result.color,
        }, index=high.index)


PINE_SCRIPT_AO = '''
//@version=5
indicator("Awesome Oscillator", overlay=false)

ao = ta.sma(hl2, 5) - ta.sma(hl2, 34)
color_ao = ao > ao[1] ? color.green : color.red

plot(ao, "AO", style=plot.style_histogram, color=color_ao)
hline(0, "Zero", color=color.gray)
'''


def main():
    print("=" * 60)
    print("Awesome Oscillator - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 60

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

    indicator = AwesomeOscillator()
    result = indicator.calculate(high, low)

    print(f"\n参数: fast={indicator.fast_period}, slow={indicator.slow_period}")
    print(f"\n最后 15 个数据点:")
    print("-" * 50)
    print(f"{'Bar':<5} {'AO':<12} {'Color':<8} {'Signal':<15}")
    print("-" * 50)

    for i in range(n - 15, n):
        ao_str = f"{result.ao[i]:.4f}" if not np.isnan(result.ao[i]) else "NaN"
        color_str = "Green" if result.color[i] == 1 else "Red" if result.color[i] == -1 else "-"

        signal = ""
        if result.zero_cross_up[i]:
            signal = "Zero Cross UP"
        elif result.zero_cross_down[i]:
            signal = "Zero Cross DOWN"

        print(f"{i:<5} {ao_str:<12} {color_str:<8} {signal:<15}")


if __name__ == "__main__":
    main()
