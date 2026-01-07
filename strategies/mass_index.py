"""
Mass Index - Python Conversion

由 Donald Dorsey 发明

核心算法:
    Single EMA = EMA(High - Low, 9)
    Double EMA = EMA(Single EMA, 9)
    Ratio = Single EMA / Double EMA
    Mass Index = Sum(Ratio, 25)

信号:
    - Mass Index > 27 然后下穿 26.5: "反转凸起" (Reversal Bulge)
    - 预示趋势可能反转
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MassIndexResult:
    """Mass Index 计算结果"""
    mass_index: np.ndarray     # Mass Index 值
    single_ema: np.ndarray     # Single EMA
    double_ema: np.ndarray     # Double EMA
    reversal_bulge: np.ndarray # 反转凸起信号


class MassIndexIndicator:
    """
    Mass Index

    用于识别趋势反转，基于高低价区间的变化。

    Parameters:
        ema_period: EMA 周期 - 默认 9
        sum_period: 求和周期 - 默认 25
        bulge_threshold: 凸起阈值 - 默认 27
        trigger_level: 触发水平 - 默认 26.5
    """

    def __init__(
        self,
        ema_period: int = 9,
        sum_period: int = 25,
        bulge_threshold: float = 27.0,
        trigger_level: float = 26.5,
    ):
        self.ema_period = ema_period
        self.sum_period = sum_period
        self.bulge_threshold = bulge_threshold
        self.trigger_level = trigger_level

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

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> MassIndexResult:
        """计算 Mass Index"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        n = len(high)

        # 高低价区间
        hl_range = high - low

        # Single EMA
        single_ema = self._ema(hl_range, self.ema_period)

        # Double EMA
        double_ema = self._ema(single_ema, self.ema_period)

        # Ratio
        ratio = np.where(double_ema != 0, single_ema / double_ema, 1.0)

        # Mass Index = Sum of Ratio
        mass_index = np.full(n, np.nan)
        for i in range(self.sum_period - 1, n):
            window = ratio[i - self.sum_period + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) == self.sum_period:
                mass_index[i] = np.sum(valid)

        # 检测反转凸起
        reversal_bulge = np.zeros(n, dtype=bool)
        above_bulge = False

        for i in range(1, n):
            if not np.isnan(mass_index[i]):
                if mass_index[i] > self.bulge_threshold:
                    above_bulge = True
                elif above_bulge and mass_index[i] < self.trigger_level:
                    reversal_bulge[i] = True
                    above_bulge = False

        return MassIndexResult(
            mass_index=mass_index,
            single_ema=single_ema,
            double_ema=double_ema,
            reversal_bulge=reversal_bulge,
        )


class MassIndexPyneCore:
    """PyneCore 兼容的 Mass Index 实现"""

    def __init__(self, ema_period: int = 9, sum_period: int = 25):
        self.indicator = MassIndexIndicator(
            ema_period=ema_period, sum_period=sum_period
        )

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(high.values, low.values)
        return pd.DataFrame({
            "mass_index": result.mass_index,
            "reversal_bulge": result.reversal_bulge,
        }, index=high.index)


PINE_SCRIPT_MASS_INDEX = '''
//@version=5
indicator("Mass Index", overlay=false)

ema_period = input.int(9, "EMA Period")
sum_period = input.int(25, "Sum Period")

hl = high - low
single_ema = ta.ema(hl, ema_period)
double_ema = ta.ema(single_ema, ema_period)
ratio = single_ema / double_ema
mass_index = math.sum(ratio, sum_period)

plot(mass_index, "Mass Index", color=color.blue)
hline(27, "Bulge", color=color.red)
hline(26.5, "Trigger", color=color.orange)
'''


def main():
    print("=" * 60)
    print("Mass Index - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 60

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.015)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.015)

    indicator = MassIndexIndicator()
    result = indicator.calculate(high, low)

    print(f"\n参数: ema_period={indicator.ema_period}, sum_period={indicator.sum_period}")
    print(f"\n最后 10 个数据点:")
    print("-" * 50)
    print(f"{'Bar':<5} {'Mass Index':<15} {'Reversal':<10}")
    print("-" * 50)

    for i in range(n - 10, n):
        mi_str = f"{result.mass_index[i]:.4f}" if not np.isnan(result.mass_index[i]) else "NaN"
        rev_str = "YES" if result.reversal_bulge[i] else ""
        print(f"{i:<5} {mi_str:<15} {rev_str:<10}")


if __name__ == "__main__":
    main()
