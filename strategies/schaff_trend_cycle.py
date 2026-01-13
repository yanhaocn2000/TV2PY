"""
Schaff Trend Cycle (STC) - Python Conversion

由 Doug Schaff 发明

核心算法:
    1. 计算 MACD
    2. 对 MACD 应用 Stochastic
    3. 对结果再次应用 Stochastic (双重平滑)

    MACD = EMA(close, fast) - EMA(close, slow)
    Stoch1 = Stochastic(MACD, cycle)
    PF = EMA(Stoch1, factor)
    Stoch2 = Stochastic(PF, cycle)
    STC = EMA(Stoch2, factor)

信号:
    - STC 上穿 25: 买入信号
    - STC 下穿 75: 卖出信号
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class STCResult:
    """Schaff Trend Cycle 计算结果"""
    stc: np.ndarray            # STC 值
    macd: np.ndarray           # MACD 值
    pf: np.ndarray             # 第一次平滑后的值
    overbought: np.ndarray     # 超买
    oversold: np.ndarray       # 超卖


class SchaffTrendCycle:
    """
    Schaff Trend Cycle (STC)

    结合 MACD 和 Stochastic 的优点，产生更快的信号。

    Parameters:
        fast_period: 快速 EMA 周期 - 默认 23
        slow_period: 慢速 EMA 周期 - 默认 50
        cycle: Stochastic 周期 - 默认 10
        factor: 平滑因子 - 默认 0.5
        overbought: 超买阈值 - 默认 75
        oversold: 超卖阈值 - 默认 25
    """

    def __init__(
        self,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle: int = 10,
        factor: float = 0.5,
        overbought: float = 75,
        oversold: float = 25,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.cycle = cycle
        self.factor = factor
        self.overbought_level = overbought
        self.oversold_level = oversold

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    def _stochastic(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 Stochastic 值 (0-100)"""
        n = len(data)
        result = np.full(n, np.nan)

        for i in range(period - 1, n):
            window = data[i - period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= period:
                highest = np.max(valid)
                lowest = np.min(valid)
                if highest != lowest:
                    result[i] = (data[i] - lowest) / (highest - lowest) * 100
                else:
                    result[i] = 50

        return result

    def _smooth(self, data: np.ndarray, factor: float) -> np.ndarray:
        """使用指数平滑"""
        n = len(data)
        result = np.full(n, np.nan)

        # 找到第一个有效值
        first_valid = -1
        for i in range(n):
            if not np.isnan(data[i]):
                first_valid = i
                result[i] = data[i]
                break

        if first_valid < 0:
            return result

        # 指数平滑
        for i in range(first_valid + 1, n):
            if not np.isnan(data[i]):
                result[i] = result[i - 1] + factor * (data[i] - result[i - 1])

        return result

    def calculate(self, close: np.ndarray) -> STCResult:
        """计算 Schaff Trend Cycle"""
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算 MACD
        fast_ema = self._ema(close, self.fast_period)
        slow_ema = self._ema(close, self.slow_period)
        macd = fast_ema - slow_ema

        # Step 2: 对 MACD 应用 Stochastic
        stoch1 = self._stochastic(macd, self.cycle)

        # Step 3: 平滑第一次 Stochastic
        pf = self._smooth(stoch1, self.factor)

        # Step 4: 对 PF 应用第二次 Stochastic
        stoch2 = self._stochastic(pf, self.cycle)

        # Step 5: 平滑得到最终 STC
        stc = self._smooth(stoch2, self.factor)

        # 超买/超卖
        overbought = stc > self.overbought_level
        oversold = stc < self.oversold_level

        return STCResult(
            stc=stc,
            macd=macd,
            pf=pf,
            overbought=overbought,
            oversold=oversold,
        )

    def get_signals(self, result: STCResult) -> dict:
        """获取交易信号"""
        n = len(result.stc)

        buy_signal = np.zeros(n, dtype=bool)
        sell_signal = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(result.stc[i]) and not np.isnan(result.stc[i-1]):
                # 上穿 25
                if result.stc[i] > self.oversold_level and result.stc[i-1] <= self.oversold_level:
                    buy_signal[i] = True
                # 下穿 75
                if result.stc[i] < self.overbought_level and result.stc[i-1] >= self.overbought_level:
                    sell_signal[i] = True

        return {
            "buy": buy_signal,
            "sell": sell_signal,
        }


class STCPyneCore:
    """PyneCore 兼容的 STC 实现"""

    def __init__(
        self,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle: int = 10,
    ):
        self.indicator = SchaffTrendCycle(
            fast_period=fast_period,
            slow_period=slow_period,
            cycle=cycle,
        )

    def __call__(self, close: pd.Series) -> pd.DataFrame:
        result = self.indicator.calculate(close.values)
        return pd.DataFrame({
            "stc": result.stc,
            "overbought": result.overbought,
            "oversold": result.oversold,
        }, index=close.index)


PINE_SCRIPT_STC = '''
//@version=5
indicator("Schaff Trend Cycle", overlay=false)

// 参数
fastLength = input.int(23, "Fast Period")
slowLength = input.int(50, "Slow Period")
cycleLength = input.int(10, "Cycle Period")
factor = input.float(0.5, "Factor")

// MACD
macd = ta.ema(close, fastLength) - ta.ema(close, slowLength)

// 第一次 Stochastic
lowest_macd = ta.lowest(macd, cycleLength)
highest_macd = ta.highest(macd, cycleLength)
stoch1 = highest_macd != lowest_macd ? (macd - lowest_macd) / (highest_macd - lowest_macd) * 100 : 50

// 平滑
var float pf = na
pf := nz(pf[1]) + factor * (stoch1 - nz(pf[1]))

// 第二次 Stochastic
lowest_pf = ta.lowest(pf, cycleLength)
highest_pf = ta.highest(pf, cycleLength)
stoch2 = highest_pf != lowest_pf ? (pf - lowest_pf) / (highest_pf - lowest_pf) * 100 : 50

// STC
var float stc = na
stc := nz(stc[1]) + factor * (stoch2 - nz(stc[1]))

// 绘图
plot(stc, "STC", color=color.blue, linewidth=2)
hline(75, "Overbought", color=color.red)
hline(25, "Oversold", color=color.green)
'''


def main():
    print("=" * 60)
    print("Schaff Trend Cycle - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 150

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)

    indicator = SchaffTrendCycle()
    result = indicator.calculate(close)

    print(f"\n参数: fast={indicator.fast_period}, slow={indicator.slow_period}, cycle={indicator.cycle}")
    print(f"\n最后 15 个数据点:")
    print("-" * 50)
    print(f"{'Bar':<5} {'Close':<10} {'STC':<10} {'Zone':<12}")
    print("-" * 50)

    for i in range(n - 15, n):
        stc_str = f"{result.stc[i]:.2f}" if not np.isnan(result.stc[i]) else "NaN"
        if result.overbought[i]:
            zone = "Overbought"
        elif result.oversold[i]:
            zone = "Oversold"
        else:
            zone = "Neutral"
        print(f"{i:<5} {close[i]:<10.2f} {stc_str:<10} {zone:<12}")

    signals = indicator.get_signals(result)
    print(f"\n信号统计:")
    print(f"  买入信号: {np.sum(signals['buy'])}")
    print(f"  卖出信号: {np.sum(signals['sell'])}")


if __name__ == "__main__":
    main()
