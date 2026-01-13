"""
Divergence Detection (RSI/MACD) - Python Conversion

核心概念:
    常规背离 (Regular Divergence):
        - 看涨背离: 价格创新低，但指标未创新低 (底背离)
        - 看跌背离: 价格创新高，但指标未创新高 (顶背离)

    隐藏背离 (Hidden Divergence):
        - 看涨隐藏背离: 价格创更高低点，但指标创更低低点
        - 看跌隐藏背离: 价格创更低高点，但指标创更高高点

应用:
    - 识别潜在趋势反转
    - 确认趋势延续
    - 结合其他指标使用
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd


class DivergenceType(Enum):
    REGULAR_BULLISH = "regular_bullish"    # 常规看涨背离 (底背离)
    REGULAR_BEARISH = "regular_bearish"    # 常规看跌背离 (顶背离)
    HIDDEN_BULLISH = "hidden_bullish"      # 隐藏看涨背离
    HIDDEN_BEARISH = "hidden_bearish"      # 隐藏看跌背离


@dataclass
class Divergence:
    """背离信息"""
    type: DivergenceType
    start_idx: int
    end_idx: int
    price_start: float
    price_end: float
    indicator_start: float
    indicator_end: float


@dataclass
class DivergenceResult:
    """背离检测结果"""
    divergences: List[Divergence]
    regular_bullish: np.ndarray    # 常规看涨背离信号
    regular_bearish: np.ndarray    # 常规看跌背离信号
    hidden_bullish: np.ndarray     # 隐藏看涨背离信号
    hidden_bearish: np.ndarray     # 隐藏看跌背离信号


class DivergenceDetector:
    """
    Divergence Detection

    检测价格和指标之间的背离。

    Parameters:
        pivot_lookback: Pivot 检测回望周期 - 默认 5
        max_bars: 两个 pivot 之间的最大K线数 - 默认 60
        min_bars: 两个 pivot 之间的最小K线数 - 默认 5
    """

    def __init__(
        self,
        pivot_lookback: int = 5,
        max_bars: int = 60,
        min_bars: int = 5,
    ):
        self.pivot_lookback = pivot_lookback
        self.max_bars = max_bars
        self.min_bars = min_bars

    def _find_pivots(
        self,
        data: np.ndarray,
        lookback: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        找到 Pivot High 和 Pivot Low

        Returns:
            (pivot_highs, pivot_lows) - 包含价格的数组，非 pivot 位置为 NaN
        """
        n = len(data)
        pivot_highs = np.full(n, np.nan)
        pivot_lows = np.full(n, np.nan)

        for i in range(lookback, n - lookback):
            # Pivot High
            is_pivot_high = True
            for j in range(1, lookback + 1):
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs[i] = data[i]

            # Pivot Low
            is_pivot_low = True
            for j in range(1, lookback + 1):
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows[i] = data[i]

        return pivot_highs, pivot_lows

    def _get_pivot_indices(
        self,
        pivots: np.ndarray,
    ) -> List[int]:
        """获取所有 pivot 的索引"""
        return [i for i in range(len(pivots)) if not np.isnan(pivots[i])]

    def calculate(
        self,
        price_high: np.ndarray,
        price_low: np.ndarray,
        indicator: np.ndarray,
    ) -> DivergenceResult:
        """
        检测背离

        Parameters:
            price_high: 价格高点数据
            price_low: 价格低点数据
            indicator: 指标数据 (RSI, MACD 等)

        Returns:
            DivergenceResult 包含所有检测到的背离
        """
        price_high = np.asarray(price_high, dtype=float)
        price_low = np.asarray(price_low, dtype=float)
        indicator = np.asarray(indicator, dtype=float)
        n = len(indicator)

        # 找到价格和指标的 pivots
        price_pivot_highs, price_pivot_lows = self._find_pivots(price_high, self.pivot_lookback)
        _, ind_pivot_lows_from_low = self._find_pivots(price_low, self.pivot_lookback)
        ind_pivot_highs, ind_pivot_lows = self._find_pivots(indicator, self.pivot_lookback)

        # 结果数组
        regular_bullish = np.zeros(n, dtype=bool)
        regular_bearish = np.zeros(n, dtype=bool)
        hidden_bullish = np.zeros(n, dtype=bool)
        hidden_bearish = np.zeros(n, dtype=bool)
        divergences = []

        # 获取价格低点和指标低点的索引
        price_low_indices = self._get_pivot_indices(price_pivot_lows)
        price_high_indices = self._get_pivot_indices(price_pivot_highs)
        ind_low_indices = self._get_pivot_indices(ind_pivot_lows)
        ind_high_indices = self._get_pivot_indices(ind_pivot_highs)

        # 检测看涨背离 (在价格低点)
        for i, curr_idx in enumerate(price_low_indices):
            if i == 0:
                continue

            # 找到前一个价格低点
            for j in range(i - 1, -1, -1):
                prev_idx = price_low_indices[j]
                bar_diff = curr_idx - prev_idx

                if bar_diff < self.min_bars:
                    continue
                if bar_diff > self.max_bars:
                    break

                # 获取对应的指标值
                prev_price = price_low[prev_idx]
                curr_price = price_low[curr_idx]
                prev_ind = indicator[prev_idx]
                curr_ind = indicator[curr_idx]

                if np.isnan(prev_ind) or np.isnan(curr_ind):
                    continue

                # 常规看涨背离: 价格创新低，指标未创新低
                if curr_price < prev_price and curr_ind > prev_ind:
                    regular_bullish[curr_idx] = True
                    divergences.append(Divergence(
                        type=DivergenceType.REGULAR_BULLISH,
                        start_idx=prev_idx,
                        end_idx=curr_idx,
                        price_start=prev_price,
                        price_end=curr_price,
                        indicator_start=prev_ind,
                        indicator_end=curr_ind,
                    ))
                    break

                # 隐藏看涨背离: 价格创更高低点，指标创更低低点
                if curr_price > prev_price and curr_ind < prev_ind:
                    hidden_bullish[curr_idx] = True
                    divergences.append(Divergence(
                        type=DivergenceType.HIDDEN_BULLISH,
                        start_idx=prev_idx,
                        end_idx=curr_idx,
                        price_start=prev_price,
                        price_end=curr_price,
                        indicator_start=prev_ind,
                        indicator_end=curr_ind,
                    ))
                    break

        # 检测看跌背离 (在价格高点)
        for i, curr_idx in enumerate(price_high_indices):
            if i == 0:
                continue

            for j in range(i - 1, -1, -1):
                prev_idx = price_high_indices[j]
                bar_diff = curr_idx - prev_idx

                if bar_diff < self.min_bars:
                    continue
                if bar_diff > self.max_bars:
                    break

                prev_price = price_high[prev_idx]
                curr_price = price_high[curr_idx]
                prev_ind = indicator[prev_idx]
                curr_ind = indicator[curr_idx]

                if np.isnan(prev_ind) or np.isnan(curr_ind):
                    continue

                # 常规看跌背离: 价格创新高，指标未创新高
                if curr_price > prev_price and curr_ind < prev_ind:
                    regular_bearish[curr_idx] = True
                    divergences.append(Divergence(
                        type=DivergenceType.REGULAR_BEARISH,
                        start_idx=prev_idx,
                        end_idx=curr_idx,
                        price_start=prev_price,
                        price_end=curr_price,
                        indicator_start=prev_ind,
                        indicator_end=curr_ind,
                    ))
                    break

                # 隐藏看跌背离: 价格创更低高点，指标创更高高点
                if curr_price < prev_price and curr_ind > prev_ind:
                    hidden_bearish[curr_idx] = True
                    divergences.append(Divergence(
                        type=DivergenceType.HIDDEN_BEARISH,
                        start_idx=prev_idx,
                        end_idx=curr_idx,
                        price_start=prev_price,
                        price_end=curr_price,
                        indicator_start=prev_ind,
                        indicator_end=curr_ind,
                    ))
                    break

        return DivergenceResult(
            divergences=divergences,
            regular_bullish=regular_bullish,
            regular_bearish=regular_bearish,
            hidden_bullish=hidden_bullish,
            hidden_bearish=hidden_bearish,
        )


class RSIDivergence:
    """RSI 背离检测"""

    def __init__(
        self,
        rsi_period: int = 14,
        pivot_lookback: int = 5,
    ):
        self.rsi_period = rsi_period
        self.detector = DivergenceDetector(pivot_lookback=pivot_lookback)

    def _rsi(self, close: np.ndarray) -> np.ndarray:
        """计算 RSI"""
        n = len(close)
        change = np.zeros(n)
        change[1:] = close[1:] - close[:-1]

        gain = np.maximum(change, 0)
        loss = np.maximum(-change, 0)

        alpha = 1.0 / self.rsi_period
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)

        if n >= self.rsi_period:
            avg_gain[self.rsi_period - 1] = np.mean(gain[:self.rsi_period])
            avg_loss[self.rsi_period - 1] = np.mean(loss[:self.rsi_period])

            for i in range(self.rsi_period, n):
                avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
                avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> Tuple[DivergenceResult, np.ndarray]:
        """
        计算 RSI 并检测背离

        Returns:
            (DivergenceResult, RSI values)
        """
        rsi = self._rsi(close)
        result = self.detector.calculate(high, low, rsi)
        return result, rsi


class DivergencePyneCore:
    """PyneCore 兼容的背离检测实现"""

    def __init__(
        self,
        rsi_period: int = 14,
        pivot_lookback: int = 5,
    ):
        self.rsi_div = RSIDivergence(
            rsi_period=rsi_period,
            pivot_lookback=pivot_lookback,
        )

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        result, rsi = self.rsi_div.calculate(
            high.values, low.values, close.values
        )
        return pd.DataFrame({
            "rsi": rsi,
            "regular_bullish": result.regular_bullish,
            "regular_bearish": result.regular_bearish,
            "hidden_bullish": result.hidden_bullish,
            "hidden_bearish": result.hidden_bearish,
        }, index=close.index)


PINE_SCRIPT_DIVERGENCE = '''
//@version=5
indicator("RSI Divergence", overlay=false)

rsiLength = input.int(14, "RSI Length")
pivotLookback = input.int(5, "Pivot Lookback")

rsi_val = ta.rsi(close, rsiLength)

// Pivot 检测
pivotLow = ta.pivotlow(rsi_val, pivotLookback, pivotLookback)
pivotHigh = ta.pivothigh(rsi_val, pivotLookback, pivotLookback)

plot(rsi_val, "RSI", color=color.blue)
hline(70, "Overbought", color=color.red)
hline(30, "Oversold", color=color.green)

// 注: 完整的背离检测需要更复杂的逻辑
'''


def main():
    print("=" * 60)
    print("Divergence Detection - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 150

    # 生成带背离的数据
    base_price = 100.0

    # 创建一个价格创新低但动量上升的情况 (模拟底背离)
    price_trend = np.concatenate([
        np.linspace(0, -15, 50),    # 下跌
        np.linspace(-15, -20, 50),  # 继续下跌创新低
        np.linspace(-20, 0, 50),    # 反弹
    ])

    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + price_trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.8
    low = close - np.abs(np.random.randn(n)) * 0.8

    # 计算 RSI 背离
    rsi_div = RSIDivergence(rsi_period=14, pivot_lookback=3)
    result, rsi = rsi_div.calculate(high, low, close)

    print(f"\n参数: RSI period=14, pivot_lookback=3")
    print(f"\n检测到 {len(result.divergences)} 个背离:")

    for div in result.divergences:
        print(f"\n  类型: {div.type.value}")
        print(f"  位置: Bar {div.start_idx} -> Bar {div.end_idx}")
        print(f"  价格: {div.price_start:.2f} -> {div.price_end:.2f}")
        print(f"  指标: {div.indicator_start:.2f} -> {div.indicator_end:.2f}")

    print(f"\n信号统计:")
    print(f"  常规看涨背离: {np.sum(result.regular_bullish)}")
    print(f"  常规看跌背离: {np.sum(result.regular_bearish)}")
    print(f"  隐藏看涨背离: {np.sum(result.hidden_bullish)}")
    print(f"  隐藏看跌背离: {np.sum(result.hidden_bearish)}")


if __name__ == "__main__":
    main()
