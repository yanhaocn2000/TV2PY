"""
Swing Detection & Market Structure - Python Conversion

核心算法:
    Swing High: 当前高点高于左右 N 根K线的高点
    Swing Low: 当前低点低于左右 N 根K线的低点

Market Structure:
    Higher High (HH): 新高点高于前一个高点
    Higher Low (HL): 新低点高于前一个低点
    Lower High (LH): 新高点低于前一个高点
    Lower Low (LL): 新低点低于前一个低点

    上涨趋势: HH + HL
    下跌趋势: LH + LL
    趋势反转: Break of Structure (BOS)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SwingPoint:
    """Swing 点"""
    index: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class SwingResult:
    """Swing Detection 计算结果"""
    swing_highs: np.ndarray        # Swing High 位置 (True/False)
    swing_lows: np.ndarray         # Swing Low 位置 (True/False)
    swing_high_prices: np.ndarray  # Swing High 价格 (NaN if not swing)
    swing_low_prices: np.ndarray   # Swing Low 价格 (NaN if not swing)


@dataclass
class MarketStructureResult:
    """Market Structure 计算结果"""
    swing_result: SwingResult
    structure: np.ndarray          # 市场结构 ('HH', 'HL', 'LH', 'LL', '')
    trend: np.ndarray              # 趋势 (1=上涨, -1=下跌, 0=横盘)
    bos: np.ndarray                # Break of Structure


class SwingDetector:
    """
    Swing Point Detection

    识别价格的摆动高点和低点。

    Parameters:
        left_bars: 左侧确认K线数 - 默认 5
        right_bars: 右侧确认K线数 - 默认 5
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
    ):
        self.left_bars = left_bars
        self.right_bars = right_bars

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> SwingResult:
        """检测 Swing Points"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        n = len(high)

        swing_highs = np.zeros(n, dtype=bool)
        swing_lows = np.zeros(n, dtype=bool)
        swing_high_prices = np.full(n, np.nan)
        swing_low_prices = np.full(n, np.nan)

        # 需要足够的数据
        min_idx = self.left_bars
        max_idx = n - self.right_bars

        for i in range(min_idx, max_idx):
            # 检查 Swing High
            is_swing_high = True
            for j in range(1, self.left_bars + 1):
                if high[i] <= high[i - j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                for j in range(1, self.right_bars + 1):
                    if high[i] <= high[i + j]:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs[i] = True
                swing_high_prices[i] = high[i]

            # 检查 Swing Low
            is_swing_low = True
            for j in range(1, self.left_bars + 1):
                if low[i] >= low[i - j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                for j in range(1, self.right_bars + 1):
                    if low[i] >= low[i + j]:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows[i] = True
                swing_low_prices[i] = low[i]

        return SwingResult(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            swing_high_prices=swing_high_prices,
            swing_low_prices=swing_low_prices,
        )

    def get_swing_points(self, result: SwingResult) -> List[SwingPoint]:
        """获取所有 Swing Points 列表"""
        points = []
        n = len(result.swing_highs)

        for i in range(n):
            if result.swing_highs[i]:
                points.append(SwingPoint(i, result.swing_high_prices[i], 'high'))
            if result.swing_lows[i]:
                points.append(SwingPoint(i, result.swing_low_prices[i], 'low'))

        return sorted(points, key=lambda x: x.index)


class MarketStructure:
    """
    Market Structure Analysis

    分析市场结构，识别趋势和结构突破。

    Parameters:
        left_bars: Swing 检测左侧K线数 - 默认 5
        right_bars: Swing 检测右侧K线数 - 默认 5
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 5,
    ):
        self.swing_detector = SwingDetector(left_bars, right_bars)

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> MarketStructureResult:
        """分析市场结构"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 检测 Swing Points
        swing_result = self.swing_detector.calculate(high, low)
        swing_points = self.swing_detector.get_swing_points(swing_result)

        # 初始化结果数组
        structure = np.full(n, '', dtype=object)
        trend = np.zeros(n, dtype=int)
        bos = np.zeros(n, dtype=bool)

        # 分析市场结构
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None
        current_trend = 0

        for point in swing_points:
            if point.type == 'high':
                if last_high is not None:
                    if point.price > last_high.price:
                        structure[point.index] = 'HH'  # Higher High
                    else:
                        structure[point.index] = 'LH'  # Lower High
                last_high = point

            else:  # low
                if last_low is not None:
                    if point.price > last_low.price:
                        structure[point.index] = 'HL'  # Higher Low
                    else:
                        structure[point.index] = 'LL'  # Lower Low
                last_low = point

        # 确定趋势和 BOS
        last_swing_high = np.nan
        last_swing_low = np.nan

        for i in range(n):
            # 更新最近的 Swing 点
            if swing_result.swing_highs[i]:
                last_swing_high = swing_result.swing_high_prices[i]
            if swing_result.swing_lows[i]:
                last_swing_low = swing_result.swing_low_prices[i]

            # 检测 Break of Structure
            if current_trend == 1:  # 上涨趋势
                if not np.isnan(last_swing_low) and close[i] < last_swing_low:
                    bos[i] = True
                    current_trend = -1
            elif current_trend == -1:  # 下跌趋势
                if not np.isnan(last_swing_high) and close[i] > last_swing_high:
                    bos[i] = True
                    current_trend = 1
            else:  # 初始化趋势
                if structure[i] == 'HH' or structure[i] == 'HL':
                    current_trend = 1
                elif structure[i] == 'LH' or structure[i] == 'LL':
                    current_trend = -1

            trend[i] = current_trend

        return MarketStructureResult(
            swing_result=swing_result,
            structure=structure,
            trend=trend,
            bos=bos,
        )


class SwingPyneCore:
    """PyneCore 兼容的 Swing Detection 实现"""

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        self.detector = SwingDetector(left_bars, right_bars)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> pd.DataFrame:
        result = self.detector.calculate(high.values, low.values)
        return pd.DataFrame({
            "swing_high": result.swing_highs,
            "swing_low": result.swing_lows,
            "swing_high_price": result.swing_high_prices,
            "swing_low_price": result.swing_low_prices,
        }, index=high.index)


class MarketStructurePyneCore:
    """PyneCore 兼容的 Market Structure 实现"""

    def __init__(self, left_bars: int = 5, right_bars: int = 5):
        self.analyzer = MarketStructure(left_bars, right_bars)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        result = self.analyzer.calculate(
            high.values, low.values, close.values
        )
        return pd.DataFrame({
            "swing_high": result.swing_result.swing_highs,
            "swing_low": result.swing_result.swing_lows,
            "structure": result.structure,
            "trend": result.trend,
            "bos": result.bos,
        }, index=close.index)


PINE_SCRIPT_SWING = '''
//@version=5
indicator("Swing Detection & Market Structure", overlay=true)

leftBars = input.int(5, "Left Bars")
rightBars = input.int(5, "Right Bars")

// Swing High/Low 检测
swingHigh = ta.pivothigh(high, leftBars, rightBars)
swingLow = ta.pivotlow(low, leftBars, rightBars)

// 绘制 Swing Points
plotshape(swingHigh, title="Swing High", location=location.abovebar,
          style=shape.triangledown, color=color.red, size=size.small)
plotshape(swingLow, title="Swing Low", location=location.belowbar,
          style=shape.triangleup, color=color.green, size=size.small)
'''


def main():
    print("=" * 60)
    print("Swing Detection & Market Structure - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 100

    # 生成带趋势的数据
    base_price = 100.0
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.randn(n) * 0.5)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.8
    low = close - np.abs(np.random.randn(n)) * 0.8

    # Swing Detection
    detector = SwingDetector(left_bars=3, right_bars=3)
    swing_result = detector.calculate(high, low)

    print(f"\n参数: left_bars={detector.left_bars}, right_bars={detector.right_bars}")

    swing_points = detector.get_swing_points(swing_result)
    print(f"\n检测到 {len(swing_points)} 个 Swing Points:")
    for point in swing_points[-10:]:
        print(f"  Bar {point.index}: {point.type.upper()} @ {point.price:.2f}")

    # Market Structure
    print("\n" + "=" * 60)
    print("Market Structure Analysis")
    print("=" * 60)

    ms = MarketStructure(left_bars=3, right_bars=3)
    ms_result = ms.calculate(high, low, close)

    # 显示结构点
    print("\n市场结构标记:")
    for i in range(n):
        if ms_result.structure[i]:
            print(f"  Bar {i}: {ms_result.structure[i]} @ {close[i]:.2f}")
        if ms_result.bos[i]:
            print(f"  Bar {i}: *** BOS (Break of Structure) ***")

    print(f"\n趋势统计:")
    print(f"  上涨趋势: {np.sum(ms_result.trend == 1)} bars")
    print(f"  下跌趋势: {np.sum(ms_result.trend == -1)} bars")
    print(f"  结构突破: {np.sum(ms_result.bos)} 次")


if __name__ == "__main__":
    main()
