"""
Fibonacci Retracement & Extension - Python Conversion

核心算法:
    基于 Swing High 和 Swing Low 计算 Fibonacci 水平

    回撤水平 (从高到低):
        0.0%   - 高点
        23.6%  - High - 0.236 * (High - Low)
        38.2%  - High - 0.382 * (High - Low)
        50.0%  - High - 0.500 * (High - Low)
        61.8%  - High - 0.618 * (High - Low)
        78.6%  - High - 0.786 * (High - Low)
        100.0% - 低点

    扩展水平:
        127.2% - High + 0.272 * (High - Low)
        161.8% - High + 0.618 * (High - Low)
        261.8% - High + 1.618 * (High - Low)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


# 标准 Fibonacci 比率
FIB_RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSION_LEVELS = [1.0, 1.272, 1.618, 2.0, 2.618, 3.618]


@dataclass
class FibonacciLevels:
    """Fibonacci 水平"""
    high: float
    low: float
    direction: str  # 'up' (低到高) or 'down' (高到低)
    retracement: Dict[float, float]  # 回撤水平
    extension: Dict[float, float]    # 扩展水平


class FibonacciRetracement:
    """
    Fibonacci Retracement & Extension

    计算两点之间的 Fibonacci 水平。

    Parameters:
        levels: 自定义回撤水平
        extension_levels: 自定义扩展水平
    """

    def __init__(
        self,
        levels: Optional[List[float]] = None,
        extension_levels: Optional[List[float]] = None,
    ):
        self.levels = levels or FIB_RETRACEMENT_LEVELS
        self.extension_levels = extension_levels or FIB_EXTENSION_LEVELS

    def calculate_levels(
        self,
        high: float,
        low: float,
        direction: str = 'down',
    ) -> FibonacciLevels:
        """
        计算 Fibonacci 水平

        Parameters:
            high: 高点价格
            low: 低点价格
            direction: 'down' (从高到低回撤) 或 'up' (从低到高回撤)

        Returns:
            FibonacciLevels 包含所有水平
        """
        range_price = high - low

        retracement = {}
        extension = {}

        if direction == 'down':
            # 下跌回撤 (从高点开始)
            for level in self.levels:
                retracement[level] = high - level * range_price

            for level in self.extension_levels:
                extension[level] = high - level * range_price
        else:
            # 上涨回撤 (从低点开始)
            for level in self.levels:
                retracement[level] = low + level * range_price

            for level in self.extension_levels:
                extension[level] = low + level * range_price

        return FibonacciLevels(
            high=high,
            low=low,
            direction=direction,
            retracement=retracement,
            extension=extension,
        )

    def auto_detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        lookback: int = 50,
    ) -> Optional[FibonacciLevels]:
        """
        自动检测最近的 Swing High/Low 并计算 Fibonacci

        Parameters:
            high, low, close: OHLC 数据
            lookback: 回望周期

        Returns:
            FibonacciLevels 或 None
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        if len(close) < lookback:
            return None

        # 在回望期内找到最高点和最低点
        recent_high = high[-lookback:]
        recent_low = low[-lookback:]

        highest_idx = np.argmax(recent_high)
        lowest_idx = np.argmin(recent_low)

        highest_price = recent_high[highest_idx]
        lowest_price = recent_low[lowest_idx]

        # 确定方向 (高点在低点之前 = 下跌回撤)
        if highest_idx < lowest_idx:
            direction = 'down'
        else:
            direction = 'up'

        return self.calculate_levels(highest_price, lowest_price, direction)

    def find_nearest_level(
        self,
        price: float,
        levels: FibonacciLevels,
    ) -> tuple:
        """
        找到最近的 Fibonacci 水平

        Returns:
            (level_name, level_price, distance)
        """
        all_levels = {**levels.retracement}
        min_distance = float('inf')
        nearest_level = None
        nearest_price = None

        for level, level_price in all_levels.items():
            distance = abs(price - level_price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = level
                nearest_price = level_price

        return nearest_level, nearest_price, min_distance


class FibonacciPyneCore:
    """PyneCore 兼容的 Fibonacci 实现"""

    def __init__(self, lookback: int = 50):
        self.fib = FibonacciRetracement()
        self.lookback = lookback

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """计算并返回 Fibonacci 水平"""
        levels = self.fib.auto_detect(
            high.values, low.values, close.values, self.lookback
        )

        if levels is None:
            return pd.DataFrame(index=close.index)

        # 创建 DataFrame，包含各个水平
        data = {
            "fib_high": levels.high,
            "fib_low": levels.low,
            "fib_236": levels.retracement.get(0.236, np.nan),
            "fib_382": levels.retracement.get(0.382, np.nan),
            "fib_500": levels.retracement.get(0.5, np.nan),
            "fib_618": levels.retracement.get(0.618, np.nan),
            "fib_786": levels.retracement.get(0.786, np.nan),
        }

        return pd.DataFrame(data, index=close.index)


PINE_SCRIPT_FIB = '''
//@version=5
indicator("Fibonacci Retracement", overlay=true)

lookback = input.int(50, "Lookback Period")

// 找到回望期内的最高点和最低点
highestBar = ta.highest(high, lookback)
lowestBar = ta.lowest(low, lookback)

// Fibonacci 水平
fib_range = highestBar - lowestBar
fib_236 = highestBar - 0.236 * fib_range
fib_382 = highestBar - 0.382 * fib_range
fib_500 = highestBar - 0.500 * fib_range
fib_618 = highestBar - 0.618 * fib_range
fib_786 = highestBar - 0.786 * fib_range

// 绘制水平线
plot(highestBar, "100%", color=color.red)
plot(fib_786, "78.6%", color=color.orange)
plot(fib_618, "61.8%", color=color.yellow)
plot(fib_500, "50%", color=color.gray)
plot(fib_382, "38.2%", color=color.lime)
plot(fib_236, "23.6%", color=color.green)
plot(lowestBar, "0%", color=color.blue)
'''


def main():
    print("=" * 60)
    print("Fibonacci Retracement - Python Implementation")
    print("=" * 60)

    fib = FibonacciRetracement()

    # 示例 1: 手动计算
    print("\n示例 1: 手动指定高低点")
    print("-" * 40)

    levels = fib.calculate_levels(high=150, low=100, direction='down')

    print(f"High: {levels.high}, Low: {levels.low}")
    print(f"Direction: {levels.direction}")
    print("\n回撤水平:")
    for level, price in sorted(levels.retracement.items()):
        print(f"  {level*100:5.1f}%: {price:.2f}")

    print("\n扩展水平:")
    for level, price in sorted(levels.extension.items()):
        print(f"  {level*100:5.1f}%: {price:.2f}")

    # 示例 2: 自动检测
    print("\n" + "=" * 60)
    print("示例 2: 自动检测")
    print("-" * 40)

    np.random.seed(42)
    n = 100

    # 生成带趋势的数据
    base_price = 100.0
    trend = np.concatenate([
        np.linspace(0, 30, 50),   # 上涨
        np.linspace(30, 15, 50),  # 回调
    ])
    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5

    auto_levels = fib.auto_detect(high, low, close, lookback=100)

    if auto_levels:
        print(f"检测到的高点: {auto_levels.high:.2f}")
        print(f"检测到的低点: {auto_levels.low:.2f}")
        print(f"方向: {auto_levels.direction}")

        print("\n关键回撤水平:")
        for level in [0.382, 0.5, 0.618]:
            price = auto_levels.retracement[level]
            print(f"  {level*100:.1f}%: {price:.2f}")

        # 找到当前价格最近的水平
        current_price = close[-1]
        level, level_price, distance = fib.find_nearest_level(current_price, auto_levels)
        print(f"\n当前价格: {current_price:.2f}")
        print(f"最近的 Fib 水平: {level*100:.1f}% @ {level_price:.2f} (距离: {distance:.2f})")


if __name__ == "__main__":
    main()
