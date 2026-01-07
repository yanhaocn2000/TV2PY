"""
MACD (Moving Average Convergence Divergence) - Python Conversion

TradingView 内置指标: ta.macd()

核心算法:
    fast_ema = ema(close, fast_period)
    slow_ema = ema(close, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

信号:
    - 买入: MACD 上穿信号线
    - 卖出: MACD 下穿信号线
    - 零线穿越也是重要信号
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MACDResult:
    """MACD 计算结果"""
    macd: np.ndarray           # MACD 线
    signal: np.ndarray         # 信号线
    histogram: np.ndarray      # 柱状图
    fast_ema: np.ndarray       # 快速 EMA
    slow_ema: np.ndarray       # 慢速 EMA
    cross_up: np.ndarray       # MACD 上穿信号线
    cross_down: np.ndarray     # MACD 下穿信号线
    above_zero: np.ndarray     # MACD 在零线上方


class MACDIndicator:
    """
    MACD (Moving Average Convergence Divergence)

    这是最经典的动量指标之一，由 Gerald Appel 发明。

    Parameters:
        fast_period: 快速 EMA 周期 - 默认 12
        slow_period: 慢速 EMA 周期 - 默认 26
        signal_period: 信号线周期 - 默认 9
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算 EMA (使用 SMA 作为种子值，与 TradingView 一致)
        """
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        # 使用 SMA 作为种子
        result[period - 1] = np.mean(data[:period])

        # 递归计算
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def calculate(self, close: np.ndarray) -> MACDResult:
        """
        计算 MACD

        算法步骤:
        1. 计算快速 EMA
        2. 计算慢速 EMA
        3. MACD 线 = 快速 EMA - 慢速 EMA
        4. 信号线 = MACD 线的 EMA
        5. 柱状图 = MACD 线 - 信号线

        Returns:
            MACDResult 包含所有计算结果
        """
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1 & 2: 计算 EMA
        fast_ema = self._ema(close, self.fast_period)
        slow_ema = self._ema(close, self.slow_period)

        # Step 3: 计算 MACD 线
        macd = fast_ema - slow_ema

        # Step 4: 计算信号线
        signal = self._ema(macd, self.signal_period)

        # Step 5: 计算柱状图
        histogram = macd - signal

        # 计算交叉信号
        cross_up = np.zeros(n, dtype=bool)
        cross_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(macd[i]) and not np.isnan(signal[i]):
                if not np.isnan(macd[i-1]) and not np.isnan(signal[i-1]):
                    cross_up[i] = macd[i] > signal[i] and macd[i-1] <= signal[i-1]
                    cross_down[i] = macd[i] < signal[i] and macd[i-1] >= signal[i-1]

        # 零线判断
        above_zero = macd > 0

        return MACDResult(
            macd=macd,
            signal=signal,
            histogram=histogram,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            cross_up=cross_up,
            cross_down=cross_down,
            above_zero=above_zero,
        )

    def get_signals(self, result: MACDResult) -> dict:
        """获取交易信号"""
        return {
            "buy": result.cross_up,
            "sell": result.cross_down,
            "bullish": result.above_zero,
        }


class MACDStrategy:
    """
    MACD 交易策略

    买入条件: MACD 上穿信号线
    卖出条件: MACD 下穿信号线
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.indicator = MACDIndicator(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def generate_signals(self, close: np.ndarray) -> pd.DataFrame:
        """生成交易信号 DataFrame"""
        result = self.indicator.calculate(close)
        signals = self.indicator.get_signals(result)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        signal[signals["buy"]] = 1
        signal[signals["sell"]] = -1

        return pd.DataFrame({
            "macd": result.macd,
            "signal_line": result.signal,
            "histogram": result.histogram,
            "cross_up": result.cross_up,
            "cross_down": result.cross_down,
            "signal": signal,
        })


class MACDPyneCore:
    """PyneCore 兼容的 MACD 实现"""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.indicator = MACDIndicator(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def __call__(self, close: pd.Series) -> pd.DataFrame:
        """计算并返回 DataFrame 格式的结果"""
        result = self.indicator.calculate(close.values)

        return pd.DataFrame({
            "macd": result.macd,
            "signal": result.signal,
            "histogram": result.histogram,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_MACD = '''
//@version=5
indicator("MACD", overlay=false)

// 参数
fast_length = input.int(12, "Fast Length")
slow_length = input.int(26, "Slow Length")
signal_length = input.int(9, "Signal Length")
src = input.source(close, "Source")

// 计算
fast_ma = ta.ema(src, fast_length)
slow_ma = ta.ema(src, slow_length)
macd = fast_ma - slow_ma
signal = ta.ema(macd, signal_length)
hist = macd - signal

// 绘图
plot(hist, title="Histogram", style=plot.style_histogram, color=(hist >= 0 ? (hist[1] < hist ? color.green : color.lime) : (hist[1] < hist ? color.red : color.maroon)))
plot(macd, title="MACD", color=color.blue)
plot(signal, title="Signal", color=color.orange)
hline(0, "Zero Line", color=color.gray)
'''


def main():
    """测试 MACD 指标"""
    print("=" * 60)
    print("MACD - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟价格数据
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)

    # 计算 MACD
    indicator = MACDIndicator()
    result = indicator.calculate(close)

    # 显示结果
    print(f"\n参数: fast={indicator.fast_period}, slow={indicator.slow_period}, signal={indicator.signal_period}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 70)
    print(f"{'Bar':<5} {'Close':<10} {'MACD':<10} {'Signal':<10} {'Hist':<10}")
    print("-" * 70)

    for i in range(n - 15, n):
        macd_str = f"{result.macd[i]:.4f}" if not np.isnan(result.macd[i]) else "NaN"
        sig_str = f"{result.signal[i]:.4f}" if not np.isnan(result.signal[i]) else "NaN"
        hist_str = f"{result.histogram[i]:.4f}" if not np.isnan(result.histogram[i]) else "NaN"

        print(f"{i:<5} {close[i]:<10.2f} {macd_str:<10} {sig_str:<10} {hist_str:<10}")

    print()
    print("信号统计:")
    print(f"  买入信号 (MACD 上穿信号线): {np.sum(result.cross_up)}")
    print(f"  卖出信号 (MACD 下穿信号线): {np.sum(result.cross_down)}")


if __name__ == "__main__":
    main()
