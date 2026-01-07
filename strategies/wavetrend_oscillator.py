"""
WaveTrend Oscillator [LazyBear] - Python Conversion

TradingView 原版: https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/

核心算法:
    ap = hlc3
    esa = ema(ap, n1)
    d = ema(abs(ap - esa), n1)
    ci = (ap - esa) / (0.015 * d)
    tci = ema(ci, n2)
    wt1 = tci
    wt2 = sma(wt1, 4)

信号:
    - 买入: wt1 上穿 wt2 且在超卖区 (< -53)
    - 卖出: wt1 下穿 wt2 且在超买区 (> 53)

参考:
    - https://medium.com/the-modern-scientist/understanding-the-wavetrend-lazybear-indicator-71254f4234ec
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class WaveTrendResult:
    """WaveTrend 计算结果"""
    wt1: np.ndarray          # WaveTrend Line 1 (主线)
    wt2: np.ndarray          # WaveTrend Line 2 (信号线)
    hlc3: np.ndarray         # HLC3 (典型价格)
    esa: np.ndarray          # EMA of HLC3
    d: np.ndarray            # EMA of |hlc3 - esa|
    ci: np.ndarray           # Channel Index
    cross_up: np.ndarray     # wt1 上穿 wt2
    cross_down: np.ndarray   # wt1 下穿 wt2
    overbought: np.ndarray   # 超买区域
    oversold: np.ndarray     # 超卖区域


class WaveTrendIndicator:
    """
    WaveTrend Oscillator [LazyBear]

    这是一个动量震荡指标，结合了价格通道和移动平均的概念。

    Parameters:
        n1: Channel Length (EMA period for price channel) - 默认 10
        n2: Average Length (EMA period for smoothing) - 默认 21
        ob_level1: Overbought Level 1 - 默认 60
        ob_level2: Overbought Level 2 - 默认 53
        os_level1: Oversold Level 1 - 默认 -60
        os_level2: Oversold Level 2 - 默认 -53
    """

    def __init__(
        self,
        n1: int = 10,
        n2: int = 21,
        ob_level1: float = 60,
        ob_level2: float = 53,
        os_level1: float = -60,
        os_level2: float = -53,
    ):
        self.n1 = n1
        self.n2 = n2
        self.ob_level1 = ob_level1
        self.ob_level2 = ob_level2
        self.os_level1 = os_level1
        self.os_level2 = os_level2

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算 EMA (使用 SMA 作为种子值，与 TradingView 一致)

        TradingView EMA 公式:
            alpha = 2 / (period + 1)
            ema[0] = sma[period-1]  (前 period 个值的 SMA 作为种子)
            ema[i] = alpha * src[i] + (1 - alpha) * ema[i-1]
        """
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan, dtype=float)

        # 找到第一个有效的种子位置
        if len(data) < period:
            return result

        # 使用 SMA 作为种子值
        result[period - 1] = np.mean(data[:period])

        # 递归计算 EMA
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> WaveTrendResult:
        """
        计算 WaveTrend Oscillator

        算法步骤:
        1. ap = hlc3 = (high + low + close) / 3
        2. esa = ema(ap, n1)
        3. d = ema(abs(ap - esa), n1)
        4. ci = (ap - esa) / (0.015 * d)
        5. tci = ema(ci, n2)
        6. wt1 = tci
        7. wt2 = sma(wt1, 4)

        Returns:
            WaveTrendResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # Step 1: 计算 HLC3 (典型价格)
        hlc3 = (high + low + close) / 3.0

        # Step 2: 计算 ESA (EMA of HLC3)
        esa = self._ema(hlc3, self.n1)

        # Step 3: 计算 D (EMA of absolute deviation)
        abs_dev = np.abs(hlc3 - esa)
        d = self._ema(abs_dev, self.n1)

        # Step 4: 计算 CI (Channel Index)
        # 避免除零
        ci = np.where(d != 0, (hlc3 - esa) / (0.015 * d), 0.0)

        # Step 5: 计算 TCI (smoothed CI)
        tci = self._ema(ci, self.n2)

        # Step 6 & 7: wt1 和 wt2
        wt1 = tci
        wt2 = self._sma(wt1, 4)

        # 计算交叉信号
        cross_up = np.zeros(len(close), dtype=bool)
        cross_down = np.zeros(len(close), dtype=bool)

        for i in range(1, len(close)):
            if not np.isnan(wt1[i]) and not np.isnan(wt2[i]):
                if not np.isnan(wt1[i-1]) and not np.isnan(wt2[i-1]):
                    # 上穿: wt1 从下方穿过 wt2
                    cross_up[i] = wt1[i] > wt2[i] and wt1[i-1] <= wt2[i-1]
                    # 下穿: wt1 从上方穿过 wt2
                    cross_down[i] = wt1[i] < wt2[i] and wt1[i-1] >= wt2[i-1]

        # 超买/超卖区域判断
        overbought = wt1 > self.ob_level2
        oversold = wt1 < self.os_level2

        return WaveTrendResult(
            wt1=wt1,
            wt2=wt2,
            hlc3=hlc3,
            esa=esa,
            d=d,
            ci=ci,
            cross_up=cross_up,
            cross_down=cross_down,
            overbought=overbought,
            oversold=oversold,
        )

    def get_signals(self, result: WaveTrendResult) -> dict:
        """
        获取交易信号

        买入信号: wt1 上穿 wt2 且在超卖区
        卖出信号: wt1 下穿 wt2 且在超买区
        """
        buy_signals = result.cross_up & result.oversold
        sell_signals = result.cross_down & result.overbought

        return {
            "buy": buy_signals,
            "sell": sell_signals,
            "cross_up": result.cross_up,
            "cross_down": result.cross_down,
        }


class WaveTrendPyneCore:
    """
    PyneCore 兼容的 WaveTrend 实现

    用于与 PyneCore 框架集成
    """

    def __init__(
        self,
        n1: int = 10,
        n2: int = 21,
        ob_level: float = 53,
        os_level: float = -53,
    ):
        self.n1 = n1
        self.n2 = n2
        self.ob_level = ob_level
        self.os_level = os_level
        self.indicator = WaveTrendIndicator(n1=n1, n2=n2)

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
            "wt1": result.wt1,
            "wt2": result.wt2,
            "hlc3": result.hlc3,
            "cross_up": result.cross_up,
            "cross_down": result.cross_down,
            "overbought": result.overbought,
            "oversold": result.oversold,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_WAVETREND = '''
//@version=5
indicator("WaveTrend [LazyBear]", shorttitle="WT_LB")

// 参数
n1 = input.int(10, "Channel Length")
n2 = input.int(21, "Average Length")
obLevel1 = input.int(60, "Over Bought Level 1")
obLevel2 = input.int(53, "Over Bought Level 2")
osLevel1 = input.int(-60, "Over Sold Level 1")
osLevel2 = input.int(-53, "Over Sold Level 2")

// 核心计算
ap = hlc3
esa = ta.ema(ap, n1)
d = ta.ema(math.abs(ap - esa), n1)
ci = (ap - esa) / (0.015 * d)
tci = ta.ema(ci, n2)

wt1 = tci
wt2 = ta.sma(wt1, 4)

// 绘图
plot(0, color=color.gray)
plot(obLevel1, color=color.red)
plot(osLevel1, color=color.green)
plot(obLevel2, color=color.red, style=plot.style_line)
plot(osLevel2, color=color.green, style=plot.style_line)

plot(wt1, color=color.green)
plot(wt2, color=color.red, style=plot.style_cross)
plot(wt1 - wt2, color=color.blue, style=plot.style_area)
'''


def main():
    """测试 WaveTrend 指标"""
    print("=" * 60)
    print("WaveTrend Oscillator [LazyBear] - Python Implementation")
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

    # 计算 WaveTrend
    indicator = WaveTrendIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: n1={indicator.n1}, n2={indicator.n2}")
    print(f"数据点数: {n}")
    print()

    print("最后 10 个数据点:")
    print("-" * 60)
    print(f"{'Bar':<5} {'Close':<10} {'WT1':<12} {'WT2':<12} {'Signal':<10}")
    print("-" * 60)

    signals = indicator.get_signals(result)

    for i in range(n - 10, n):
        signal = ""
        if signals["buy"][i]:
            signal = "BUY"
        elif signals["sell"][i]:
            signal = "SELL"
        elif result.cross_up[i]:
            signal = "cross_up"
        elif result.cross_down[i]:
            signal = "cross_dn"

        print(f"{i:<5} {close[i]:<10.2f} {result.wt1[i]:<12.4f} {result.wt2[i]:<12.4f} {signal:<10}")

    print()
    print("超买/超卖区域统计:")
    print(f"  超买 (WT1 > {indicator.ob_level2}): {np.sum(result.overbought)} bars")
    print(f"  超卖 (WT1 < {indicator.os_level2}): {np.sum(result.oversold)} bars")

    print()
    print("交易信号统计:")
    print(f"  买入信号: {np.sum(signals['buy'])}")
    print(f"  卖出信号: {np.sum(signals['sell'])}")


if __name__ == "__main__":
    main()
