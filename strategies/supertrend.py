"""
SuperTrend Indicator [KivancOzbilgic] - Python Conversion

TradingView 原版: https://www.tradingview.com/script/r6dAP7yi/

核心算法:
    atr = ta.atr(period)  # 或 sma(tr, period)
    up = hl2 - (mult * atr)
    dn = hl2 + (mult * atr)

    趋势逻辑:
    - trend == 1 (上涨) 且 close < up_band → 趋势翻转为 -1
    - trend == -1 (下跌) 且 close > dn_band → 趋势翻转为 1

参考:
    - https://www.tradingview.com/script/r6dAP7yi/
    - https://www.tradingview.com/script/P5Gu6F8k/
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class ATRMethod(Enum):
    """ATR 计算方法"""
    RMA = "rma"    # TradingView 默认 (Wilder's smoothing)
    SMA = "sma"    # 简单移动平均


@dataclass
class SuperTrendResult:
    """SuperTrend 计算结果"""
    supertrend: np.ndarray    # SuperTrend 线
    trend: np.ndarray         # 趋势方向 (1=上涨, -1=下跌)
    upper_band: np.ndarray    # 上轨
    lower_band: np.ndarray    # 下轨
    atr: np.ndarray           # ATR 值
    buy_signal: np.ndarray    # 买入信号 (趋势从-1变为1)
    sell_signal: np.ndarray   # 卖出信号 (趋势从1变为-1)


class SuperTrendIndicator:
    """
    SuperTrend Indicator [KivancOzbilgic]

    SuperTrend 是一个趋势跟踪指标，基于 ATR 计算动态支撑/阻力带。
    当价格突破上轨，趋势转为上涨；当价格跌破下轨，趋势转为下跌。

    Parameters:
        period: ATR 周期 - 默认 10
        multiplier: ATR 乘数 - 默认 3.0
        atr_method: ATR 计算方法 - 默认 RMA (Wilder's)
    """

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        atr_method: ATRMethod = ATRMethod.RMA,
    ):
        self.period = period
        self.multiplier = multiplier
        self.atr_method = atr_method

    def _true_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """
        计算 True Range

        TR = max(high - low, abs(high - close[1]), abs(low - close[1]))
        """
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        return tr

    def _rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算 RMA (Wilder's smoothing method)

        TradingView RMA 公式:
            alpha = 1 / period
            rma[0:period] = sma(data[0:period])
            rma[i] = alpha * data[i] + (1 - alpha) * rma[i-1]
        """
        alpha = 1.0 / period
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        # 使用 SMA 作为种子
        result[period - 1] = np.mean(data[:period])

        # 递归计算
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def _atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """
        计算 ATR

        根据 atr_method 使用 RMA 或 SMA
        """
        tr = self._true_range(high, low, close)

        if self.atr_method == ATRMethod.RMA:
            return self._rma(tr, self.period)
        else:
            return self._sma(tr, self.period)

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> SuperTrendResult:
        """
        计算 SuperTrend

        算法步骤:
        1. 计算 ATR
        2. 计算基础上下轨:
           basic_upper = hl2 + (mult * atr)
           basic_lower = hl2 - (mult * atr)
        3. 计算最终上下轨 (考虑前值):
           upper = min(basic_upper, upper[1]) if close[1] > upper[1] else basic_upper
           lower = max(basic_lower, lower[1]) if close[1] < lower[1] else basic_lower
        4. 确定趋势方向

        Returns:
            SuperTrendResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算 ATR
        atr = self._atr(high, low, close)

        # Step 2: 计算 HL2 (中间价)
        hl2 = (high + low) / 2.0

        # Step 3: 计算基础上下轨
        basic_upper = hl2 + (self.multiplier * atr)
        basic_lower = hl2 - (self.multiplier * atr)

        # Step 4: 初始化最终上下轨和趋势
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        trend = np.zeros(n, dtype=int)
        supertrend = np.full(n, np.nan)

        # 找到第一个有效位置
        start_idx = self.period - 1

        # 初始化第一个有效值
        if start_idx < n:
            upper_band[start_idx] = basic_upper[start_idx]
            lower_band[start_idx] = basic_lower[start_idx]
            trend[start_idx] = 1  # 初始假设为上涨趋势
            supertrend[start_idx] = lower_band[start_idx]

        # Step 5: 递归计算
        for i in range(start_idx + 1, n):
            # 更新上轨
            if basic_upper[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                upper_band[i] = basic_upper[i]
            else:
                upper_band[i] = upper_band[i - 1]

            # 更新下轨
            if basic_lower[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                lower_band[i] = basic_lower[i]
            else:
                lower_band[i] = lower_band[i - 1]

            # 更新趋势
            prev_trend = trend[i - 1]

            if prev_trend == -1 and close[i] > upper_band[i - 1]:
                trend[i] = 1
            elif prev_trend == 1 and close[i] < lower_band[i - 1]:
                trend[i] = -1
            else:
                trend[i] = prev_trend

            # 更新 SuperTrend 值
            if trend[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]

        # Step 6: 计算买卖信号
        buy_signal = np.zeros(n, dtype=bool)
        sell_signal = np.zeros(n, dtype=bool)

        for i in range(start_idx + 1, n):
            buy_signal[i] = trend[i] == 1 and trend[i - 1] == -1
            sell_signal[i] = trend[i] == -1 and trend[i - 1] == 1

        return SuperTrendResult(
            supertrend=supertrend,
            trend=trend,
            upper_band=upper_band,
            lower_band=lower_band,
            atr=atr,
            buy_signal=buy_signal,
            sell_signal=sell_signal,
        )

    def get_signals(self, result: SuperTrendResult) -> dict:
        """获取交易信号"""
        return {
            "buy": result.buy_signal,
            "sell": result.sell_signal,
            "trend": result.trend,
        }


class SuperTrendStrategy:
    """
    SuperTrend 交易策略

    买入条件: 趋势从下跌转为上涨 (buy_signal)
    卖出条件: 趋势从上涨转为下跌 (sell_signal)
    """

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        atr_method: ATRMethod = ATRMethod.RMA,
    ):
        self.indicator = SuperTrendIndicator(
            period=period,
            multiplier=multiplier,
            atr_method=atr_method,
        )

    def generate_signals(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> pd.DataFrame:
        """
        生成交易信号 DataFrame

        Returns:
            DataFrame 包含:
            - supertrend: SuperTrend 值
            - trend: 趋势方向
            - signal: 交易信号 (1=买入, -1=卖出, 0=无信号)
            - position: 持仓方向
        """
        result = self.indicator.calculate(high, low, close)
        signals = self.indicator.get_signals(result)

        n = len(close)
        signal = np.zeros(n, dtype=int)
        signal[signals["buy"]] = 1
        signal[signals["sell"]] = -1

        # 计算持仓
        position = np.zeros(n, dtype=int)
        for i in range(len(position)):
            if signal[i] == 1:
                position[i] = 1
            elif signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i - 1] if i > 0 else 0

        return pd.DataFrame({
            "supertrend": result.supertrend,
            "trend": result.trend,
            "upper_band": result.upper_band,
            "lower_band": result.lower_band,
            "atr": result.atr,
            "buy_signal": result.buy_signal,
            "sell_signal": result.sell_signal,
            "signal": signal,
            "position": position,
        })


class SuperTrendPyneCore:
    """
    PyneCore 兼容的 SuperTrend 实现
    """

    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        atr_method: str = "rma",
    ):
        method = ATRMethod.RMA if atr_method.lower() == "rma" else ATRMethod.SMA
        self.indicator = SuperTrendIndicator(
            period=period,
            multiplier=multiplier,
            atr_method=method,
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
            "supertrend": result.supertrend,
            "trend": result.trend,
            "upper": result.upper_band,
            "lower": result.lower_band,
            "buy": result.buy_signal,
            "sell": result.sell_signal,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_SUPERTREND = '''
//@version=5
indicator("SuperTrend", overlay=true)

// 参数
Periods = input.int(10, title="ATR Period")
Multiplier = input.float(3.0, title="ATR Multiplier", step=0.1)
changeATR = input.bool(true, title="Change ATR Calculation Method?")
src = input.source(hl2, title="Source")

// ATR 计算
atr2 = ta.sma(ta.tr, Periods)
atr = changeATR ? ta.atr(Periods) : atr2

// 上下轨计算
up = src - (Multiplier * atr)
up1 = nz(up[1], up)
up := close[1] > up1 ? math.max(up, up1) : up

dn = src + (Multiplier * atr)
dn1 = nz(dn[1], dn)
dn := close[1] < dn1 ? math.min(dn, dn1) : dn

// 趋势判断
var trend = 1
trend := nz(trend[1], trend)
trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend

// SuperTrend 线
superTrend = trend == 1 ? up : dn

// 信号
buySignal = trend == 1 and trend[1] == -1
sellSignal = trend == -1 and trend[1] == 1

// 绘图
upPlot = plot(trend == 1 ? superTrend : na, title="Up Trend", style=plot.style_linebr, linewidth=2, color=color.green)
dnPlot = plot(trend == 1 ? na : superTrend, title="Down Trend", style=plot.style_linebr, linewidth=2, color=color.red)

plotshape(buySignal, title="Buy", text="Buy", location=location.belowbar, style=shape.labelup, size=size.tiny, color=color.green, textcolor=color.white)
plotshape(sellSignal, title="Sell", text="Sell", location=location.abovebar, style=shape.labeldown, size=size.tiny, color=color.red, textcolor=color.white)
'''


def main():
    """测试 SuperTrend 指标"""
    print("=" * 60)
    print("SuperTrend Indicator [KivancOzbilgic] - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟价格数据 (带有趋势)
    base_price = 100.0
    trend_component = np.linspace(0, 20, n)  # 上涨趋势
    noise = np.cumsum(np.random.randn(n) * 0.5)
    close = base_price + trend_component + noise
    high = close + np.abs(np.random.randn(n)) * 1.0
    low = close - np.abs(np.random.randn(n)) * 1.0

    # 计算 SuperTrend
    indicator = SuperTrendIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: period={indicator.period}, multiplier={indicator.multiplier}")
    print(f"ATR 方法: {indicator.atr_method.value}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'SuperTrend':<12} {'Trend':<8} {'Signal':<10}")
    print("-" * 80)

    for i in range(n - 15, n):
        signal = ""
        if result.buy_signal[i]:
            signal = "BUY"
        elif result.sell_signal[i]:
            signal = "SELL"

        trend_str = "UP" if result.trend[i] == 1 else "DOWN"
        st = result.supertrend[i]
        st_str = f"{st:.2f}" if not np.isnan(st) else "NaN"

        print(f"{i:<5} {close[i]:<10.2f} {st_str:<12} {trend_str:<8} {signal:<10}")

    print()
    print("交易信号统计:")
    print(f"  买入信号: {np.sum(result.buy_signal)}")
    print(f"  卖出信号: {np.sum(result.sell_signal)}")

    # 趋势统计
    valid_trend = result.trend[~np.isnan(result.supertrend)]
    print(f"\n趋势统计 (有效数据):")
    print(f"  上涨趋势: {np.sum(valid_trend == 1)} bars")
    print(f"  下跌趋势: {np.sum(valid_trend == -1)} bars")


if __name__ == "__main__":
    main()
