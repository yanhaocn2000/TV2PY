"""
Pivot Points - Python Conversion

TradingView 内置指标: 多种 Pivot Points 类型

核心算法 (Standard/Classic):
    PP = (High + Low + Close) / 3
    R1 = 2 * PP - Low
    S1 = 2 * PP - High
    R2 = PP + (High - Low)
    S2 = PP - (High - Low)
    R3 = High + 2 * (PP - Low)
    S3 = Low - 2 * (High - PP)

类型:
    - Standard (Classic)
    - Fibonacci
    - Woodie
    - Camarilla
    - DeMark
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class PivotType(Enum):
    STANDARD = "standard"
    FIBONACCI = "fibonacci"
    WOODIE = "woodie"
    CAMARILLA = "camarilla"
    DEMARK = "demark"


@dataclass
class PivotPointsResult:
    """Pivot Points 计算结果"""
    pp: np.ndarray     # Pivot Point
    r1: np.ndarray     # Resistance 1
    r2: np.ndarray     # Resistance 2
    r3: np.ndarray     # Resistance 3
    s1: np.ndarray     # Support 1
    s2: np.ndarray     # Support 2
    s3: np.ndarray     # Support 3


class PivotPointsIndicator:
    """
    Pivot Points 支撑阻力计算

    Parameters:
        pivot_type: Pivot 类型 - 默认 STANDARD
    """

    def __init__(self, pivot_type: PivotType = PivotType.STANDARD):
        self.pivot_type = pivot_type

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_price: Optional[np.ndarray] = None,
    ) -> PivotPointsResult:
        """
        计算 Pivot Points

        注意: 通常使用前一日的 OHLC 计算当日的 Pivot Points
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        pp = np.full(n, np.nan)
        r1 = np.full(n, np.nan)
        r2 = np.full(n, np.nan)
        r3 = np.full(n, np.nan)
        s1 = np.full(n, np.nan)
        s2 = np.full(n, np.nan)
        s3 = np.full(n, np.nan)

        for i in range(1, n):
            h = high[i - 1]
            l = low[i - 1]
            c = close[i - 1]
            o = open_price[i - 1] if open_price is not None else c

            if self.pivot_type == PivotType.STANDARD:
                pp[i] = (h + l + c) / 3
                r1[i] = 2 * pp[i] - l
                s1[i] = 2 * pp[i] - h
                r2[i] = pp[i] + (h - l)
                s2[i] = pp[i] - (h - l)
                r3[i] = h + 2 * (pp[i] - l)
                s3[i] = l - 2 * (h - pp[i])

            elif self.pivot_type == PivotType.FIBONACCI:
                pp[i] = (h + l + c) / 3
                range_hl = h - l
                r1[i] = pp[i] + 0.382 * range_hl
                r2[i] = pp[i] + 0.618 * range_hl
                r3[i] = pp[i] + range_hl
                s1[i] = pp[i] - 0.382 * range_hl
                s2[i] = pp[i] - 0.618 * range_hl
                s3[i] = pp[i] - range_hl

            elif self.pivot_type == PivotType.WOODIE:
                pp[i] = (h + l + 2 * c) / 4
                r1[i] = 2 * pp[i] - l
                s1[i] = 2 * pp[i] - h
                r2[i] = pp[i] + (h - l)
                s2[i] = pp[i] - (h - l)
                r3[i] = h + 2 * (pp[i] - l)
                s3[i] = l - 2 * (h - pp[i])

            elif self.pivot_type == PivotType.CAMARILLA:
                pp[i] = (h + l + c) / 3
                range_hl = h - l
                r1[i] = c + range_hl * 1.1 / 12
                r2[i] = c + range_hl * 1.1 / 6
                r3[i] = c + range_hl * 1.1 / 4
                s1[i] = c - range_hl * 1.1 / 12
                s2[i] = c - range_hl * 1.1 / 6
                s3[i] = c - range_hl * 1.1 / 4

            elif self.pivot_type == PivotType.DEMARK:
                if c < o:
                    x = h + 2 * l + c
                elif c > o:
                    x = 2 * h + l + c
                else:
                    x = h + l + 2 * c

                pp[i] = x / 4
                r1[i] = x / 2 - l
                s1[i] = x / 2 - h
                r2[i] = r1[i]  # DeMark 通常只计算 R1/S1
                s2[i] = s1[i]
                r3[i] = r1[i]
                s3[i] = s1[i]

        return PivotPointsResult(
            pp=pp, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3
        )


class PivotPointsPyneCore:
    """PyneCore 兼容的 Pivot Points 实现"""

    def __init__(self, pivot_type: str = "standard"):
        type_map = {
            "standard": PivotType.STANDARD,
            "fibonacci": PivotType.FIBONACCI,
            "woodie": PivotType.WOODIE,
            "camarilla": PivotType.CAMARILLA,
            "demark": PivotType.DEMARK,
        }
        self.indicator = PivotPointsIndicator(
            pivot_type=type_map.get(pivot_type.lower(), PivotType.STANDARD)
        )

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(
            high.values, low.values, close.values
        )
        return pd.DataFrame({
            "pp": result.pp,
            "r1": result.r1, "r2": result.r2, "r3": result.r3,
            "s1": result.s1, "s2": result.s2, "s3": result.s3,
        }, index=close.index)


PINE_SCRIPT_PIVOT = '''
//@version=5
indicator("Pivot Points", overlay=true)

// Standard Pivot Points
pp = (high[1] + low[1] + close[1]) / 3
r1 = 2 * pp - low[1]
s1 = 2 * pp - high[1]
r2 = pp + (high[1] - low[1])
s2 = pp - (high[1] - low[1])

plot(pp, "PP", color=color.yellow)
plot(r1, "R1", color=color.red)
plot(s1, "S1", color=color.green)
'''


def main():
    print("=" * 60)
    print("Pivot Points - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 30

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)

    print("\n各种 Pivot 类型比较 (最后一根K线):")
    print("-" * 70)

    for ptype in PivotType:
        indicator = PivotPointsIndicator(pivot_type=ptype)
        result = indicator.calculate(high, low, close)

        print(f"\n{ptype.value.upper()}:")
        print(f"  PP: {result.pp[-1]:.2f}")
        print(f"  R1: {result.r1[-1]:.2f}, R2: {result.r2[-1]:.2f}, R3: {result.r3[-1]:.2f}")
        print(f"  S1: {result.s1[-1]:.2f}, S2: {result.s2[-1]:.2f}, S3: {result.s3[-1]:.2f}")


if __name__ == "__main__":
    main()
