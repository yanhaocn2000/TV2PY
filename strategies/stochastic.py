"""
Stochastic Oscillator - Python Conversion

TradingView 内置指标: ta.stoch()

核心算法:
    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = sma(%K, smooth_d)

    Full Stochastic 还包括 %K 的平滑

信号:
    - %K 上穿 %D 且在超卖区 (< 20): 买入
    - %K 下穿 %D 且在超买区 (> 80): 卖出
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StochasticResult:
    """Stochastic 计算结果"""
    k: np.ndarray              # %K 线
    d: np.ndarray              # %D 线 (信号线)
    overbought: np.ndarray     # 超买区域
    oversold: np.ndarray       # 超卖区域
    cross_up: np.ndarray       # %K 上穿 %D
    cross_down: np.ndarray     # %K 下穿 %D


class StochasticIndicator:
    """
    Stochastic Oscillator (随机振荡器)

    由 George Lane 发明，用于识别超买/超卖状态和趋势反转。

    Parameters:
        k_period: %K 周期 - 默认 14
        d_period: %D 周期 (SMA of %K) - 默认 3
        smooth_k: %K 平滑周期 - 默认 3 (Full Stochastic)
        overbought: 超买阈值 - 默认 80
        oversold: 超卖阈值 - 默认 20
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
        overbought: float = 80,
        oversold: float = 20,
    ):
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.overbought_level = overbought
        self.oversold_level = oversold

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

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid = data[i - period + 1:i + 1]
            valid = valid[~np.isnan(valid)]
            if len(valid) >= period:
                result[i] = np.mean(valid)
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> StochasticResult:
        """
        计算 Stochastic Oscillator

        算法步骤:
        1. 计算回望期内的最高价和最低价
        2. 计算原始 %K
        3. 平滑 %K (Full Stochastic)
        4. 计算 %D (信号线)

        Returns:
            StochasticResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算最高/最低
        highest_high = self._highest(high, self.k_period)
        lowest_low = self._lowest(low, self.k_period)

        # Step 2: 计算原始 %K
        range_hl = highest_high - lowest_low
        raw_k = np.where(range_hl != 0, (close - lowest_low) / range_hl * 100, 50)

        # Step 3: 平滑 %K (Full Stochastic)
        if self.smooth_k > 1:
            k = self._sma(raw_k, self.smooth_k)
        else:
            k = raw_k

        # Step 4: 计算 %D
        d = self._sma(k, self.d_period)

        # 计算交叉信号
        cross_up = np.zeros(n, dtype=bool)
        cross_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(k[i]) and not np.isnan(d[i]):
                if not np.isnan(k[i-1]) and not np.isnan(d[i-1]):
                    cross_up[i] = k[i] > d[i] and k[i-1] <= d[i-1]
                    cross_down[i] = k[i] < d[i] and k[i-1] >= d[i-1]

        # 超买/超卖
        overbought = k > self.overbought_level
        oversold = k < self.oversold_level

        return StochasticResult(
            k=k,
            d=d,
            overbought=overbought,
            oversold=oversold,
            cross_up=cross_up,
            cross_down=cross_down,
        )

    def get_signals(self, result: StochasticResult) -> dict:
        """
        获取交易信号

        买入: %K 上穿 %D 且在超卖区
        卖出: %K 下穿 %D 且在超买区
        """
        buy_signal = result.cross_up & result.oversold
        sell_signal = result.cross_down & result.overbought

        return {
            "buy": buy_signal,
            "sell": sell_signal,
            "cross_up": result.cross_up,
            "cross_down": result.cross_down,
        }


class StochasticRSI:
    """
    Stochastic RSI

    结合 RSI 和 Stochastic 的优点:
        rsi = RSI(close, rsi_period)
        stoch_rsi = (rsi - lowest(rsi)) / (highest(rsi) - lowest(rsi))
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ):
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period

    def _rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 RMA"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def _rsi(self, close: np.ndarray) -> np.ndarray:
        """计算 RSI"""
        n = len(close)
        change = np.zeros(n)
        change[1:] = close[1:] - close[:-1]

        gain = np.maximum(change, 0)
        loss = np.maximum(-change, 0)

        avg_gain = self._rma(gain, self.rsi_period)
        avg_loss = self._rma(loss, self.rsi_period)

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.where(avg_loss == 0, 100, rsi)

        return rsi

    def _highest(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid = data[i - period + 1:i + 1]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 0:
                result[i] = np.max(valid)
        return result

    def _lowest(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid = data[i - period + 1:i + 1]
            valid = valid[~np.isnan(valid)]
            if len(valid) > 0:
                result[i] = np.min(valid)
        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid = data[i - period + 1:i + 1]
            valid = valid[~np.isnan(valid)]
            if len(valid) >= period:
                result[i] = np.mean(valid)
        return result

    def calculate(self, close: np.ndarray) -> dict:
        """计算 Stochastic RSI"""
        rsi = self._rsi(close)

        highest_rsi = self._highest(rsi, self.stoch_period)
        lowest_rsi = self._lowest(rsi, self.stoch_period)

        range_rsi = highest_rsi - lowest_rsi
        stoch_rsi = np.where(range_rsi != 0, (rsi - lowest_rsi) / range_rsi * 100, 50)

        k = self._sma(stoch_rsi, self.k_period)
        d = self._sma(k, self.d_period)

        return {
            "rsi": rsi,
            "stoch_rsi": stoch_rsi,
            "k": k,
            "d": d,
        }


class StochasticPyneCore:
    """PyneCore 兼容的 Stochastic 实现"""

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ):
        self.indicator = StochasticIndicator(
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
        )

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """计算并返回 DataFrame 格式的结果"""
        result = self.indicator.calculate(
            high.values,
            low.values,
            close.values,
        )

        return pd.DataFrame({
            "k": result.k,
            "d": result.d,
            "overbought": result.overbought,
            "oversold": result.oversold,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_STOCHASTIC = '''
//@version=5
indicator("Stochastic Oscillator", overlay=false)

// 参数
periodK = input.int(14, "%K Period")
smoothK = input.int(3, "Smooth K")
periodD = input.int(3, "%D Period")

// 计算
k = ta.sma(ta.stoch(close, high, low, periodK), smoothK)
d = ta.sma(k, periodD)

// 绘图
plot(k, "%K", color=color.blue)
plot(d, "%D", color=color.orange)
hline(80, "Overbought", color=color.red)
hline(20, "Oversold", color=color.green)
hline(50, "Middle", color=color.gray)

// 背景色
bgcolor(k < 20 ? color.new(color.green, 90) : k > 80 ? color.new(color.red, 90) : na)
'''


def main():
    """测试 Stochastic 指标"""
    print("=" * 60)
    print("Stochastic Oscillator - Python Implementation")
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

    # 计算 Stochastic
    indicator = StochasticIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: K={indicator.k_period}, smooth_K={indicator.smooth_k}, D={indicator.d_period}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 70)
    print(f"{'Bar':<5} {'Close':<10} {'%K':<10} {'%D':<10} {'Zone':<12} {'Signal':<10}")
    print("-" * 70)

    signals = indicator.get_signals(result)

    for i in range(n - 15, n):
        k_str = f"{result.k[i]:.2f}" if not np.isnan(result.k[i]) else "NaN"
        d_str = f"{result.d[i]:.2f}" if not np.isnan(result.d[i]) else "NaN"

        if result.overbought[i]:
            zone = "Overbought"
        elif result.oversold[i]:
            zone = "Oversold"
        else:
            zone = "Neutral"

        signal = ""
        if signals["buy"][i]:
            signal = "BUY"
        elif signals["sell"][i]:
            signal = "SELL"

        print(f"{i:<5} {close[i]:<10.2f} {k_str:<10} {d_str:<10} {zone:<12} {signal:<10}")

    print()
    print("信号统计:")
    print(f"  买入信号: {np.sum(signals['buy'])}")
    print(f"  卖出信号: {np.sum(signals['sell'])}")


if __name__ == "__main__":
    main()
