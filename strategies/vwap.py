"""
VWAP (Volume Weighted Average Price) - Python Conversion

TradingView 内置指标: ta.vwap()

核心算法:
    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3

特点:
    - 机构交易者常用的基准价格
    - 通常在每日开盘时重置
    - 价格高于 VWAP = 看涨，低于 VWAP = 看跌
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class VWAPResult:
    """VWAP 计算结果"""
    vwap: np.ndarray               # VWAP 值
    typical_price: np.ndarray      # 典型价格
    cum_volume: np.ndarray         # 累计成交量
    cum_pv: np.ndarray             # 累计 (价格 * 成交量)
    upper_band: np.ndarray         # 上轨 (可选)
    lower_band: np.ndarray         # 下轨 (可选)


class VWAPIndicator:
    """
    VWAP (Volume Weighted Average Price)

    成交量加权平均价格，机构投资者用来评估交易执行质量的基准。

    Parameters:
        anchor: 锚定周期 ('session', 'week', 'month', 'none')
        stdev_mult: 标准差带乘数 (用于计算上下轨)
    """

    def __init__(
        self,
        anchor: str = "session",
        stdev_mult: float = 2.0,
    ):
        self.anchor = anchor
        self.stdev_mult = stdev_mult

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> VWAPResult:
        """
        计算 VWAP

        算法步骤:
        1. 计算典型价格 = (high + low + close) / 3
        2. 累计 (典型价格 * 成交量)
        3. 累计成交量
        4. VWAP = 累计PV / 累计Volume

        如果提供 timestamps，则在每个新周期重置累计值

        Returns:
            VWAPResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = len(close)

        # Step 1: 计算典型价格
        typical_price = (high + low + close) / 3.0

        # Step 2 & 3: 累计计算
        cum_pv = np.zeros(n)
        cum_volume = np.zeros(n)
        cum_pv2 = np.zeros(n)  # 用于计算标准差

        # 检测周期重置点
        reset_points = self._get_reset_points(timestamps, n)

        for i in range(n):
            if reset_points[i]:
                # 新周期开始，重置累计值
                cum_pv[i] = typical_price[i] * volume[i]
                cum_volume[i] = volume[i]
                cum_pv2[i] = (typical_price[i] ** 2) * volume[i]
            else:
                cum_pv[i] = cum_pv[i - 1] + typical_price[i] * volume[i]
                cum_volume[i] = cum_volume[i - 1] + volume[i]
                cum_pv2[i] = cum_pv2[i - 1] + (typical_price[i] ** 2) * volume[i]

        # Step 4: 计算 VWAP
        vwap = np.where(cum_volume > 0, cum_pv / cum_volume, typical_price)

        # 计算标准差带
        variance = np.where(
            cum_volume > 0,
            (cum_pv2 / cum_volume) - (vwap ** 2),
            0
        )
        variance = np.maximum(variance, 0)  # 确保非负
        stdev = np.sqrt(variance)

        upper_band = vwap + self.stdev_mult * stdev
        lower_band = vwap - self.stdev_mult * stdev

        return VWAPResult(
            vwap=vwap,
            typical_price=typical_price,
            cum_volume=cum_volume,
            cum_pv=cum_pv,
            upper_band=upper_band,
            lower_band=lower_band,
        )

    def _get_reset_points(
        self,
        timestamps: Optional[np.ndarray],
        n: int,
    ) -> np.ndarray:
        """获取重置点"""
        reset_points = np.zeros(n, dtype=bool)
        reset_points[0] = True  # 第一个点总是重置

        if timestamps is None or self.anchor == "none":
            return reset_points

        timestamps = pd.to_datetime(timestamps)

        for i in range(1, n):
            if self.anchor == "session":
                # 每日重置
                if timestamps[i].date() != timestamps[i - 1].date():
                    reset_points[i] = True
            elif self.anchor == "week":
                # 每周重置
                if timestamps[i].isocalendar()[1] != timestamps[i - 1].isocalendar()[1]:
                    reset_points[i] = True
            elif self.anchor == "month":
                # 每月重置
                if timestamps[i].month != timestamps[i - 1].month:
                    reset_points[i] = True

        return reset_points

    def get_signals(
        self,
        close: np.ndarray,
        result: VWAPResult,
    ) -> dict:
        """
        获取交易信号

        - 价格高于 VWAP: 看涨
        - 价格低于 VWAP: 看跌
        - 价格触及上轨: 可能超买
        - 价格触及下轨: 可能超卖
        """
        above_vwap = close > result.vwap
        below_vwap = close < result.vwap

        # 穿越信号
        n = len(close)
        cross_above = np.zeros(n, dtype=bool)
        cross_below = np.zeros(n, dtype=bool)

        for i in range(1, n):
            cross_above[i] = above_vwap[i] and not above_vwap[i - 1]
            cross_below[i] = below_vwap[i] and not below_vwap[i - 1]

        return {
            "above_vwap": above_vwap,
            "below_vwap": below_vwap,
            "cross_above": cross_above,
            "cross_below": cross_below,
            "at_upper": close >= result.upper_band,
            "at_lower": close <= result.lower_band,
        }


class VWAPPyneCore:
    """PyneCore 兼容的 VWAP 实现"""

    def __init__(
        self,
        anchor: str = "session",
        stdev_mult: float = 2.0,
    ):
        self.indicator = VWAPIndicator(anchor=anchor, stdev_mult=stdev_mult)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """计算并返回 DataFrame 格式的结果"""
        timestamps = close.index.values if isinstance(close.index, pd.DatetimeIndex) else None

        result = self.indicator.calculate(
            high.values,
            low.values,
            close.values,
            volume.values,
            timestamps,
        )

        return pd.DataFrame({
            "vwap": result.vwap,
            "upper": result.upper_band,
            "lower": result.lower_band,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_VWAP = '''
//@version=5
indicator("VWAP", overlay=true)

// VWAP 计算
vwap_value = ta.vwap(hlc3)

// 标准差带
stdev_mult = input.float(2.0, "StDev Multiplier")

// 使用手动计算的标准差带
var float cumPV = 0.0
var float cumV = 0.0
var float cumPV2 = 0.0

if ta.change(time("D"))
    cumPV := 0.0
    cumV := 0.0
    cumPV2 := 0.0

cumPV := cumPV + hlc3 * volume
cumV := cumV + volume
cumPV2 := cumPV2 + hlc3 * hlc3 * volume

vwap_manual = cumV > 0 ? cumPV / cumV : hlc3
variance = cumV > 0 ? (cumPV2 / cumV) - (vwap_manual * vwap_manual) : 0
stdev = math.sqrt(math.max(variance, 0))

upperBand = vwap_manual + stdev_mult * stdev
lowerBand = vwap_manual - stdev_mult * stdev

// 绘图
plot(vwap_value, "VWAP", color=color.blue, linewidth=2)
plot(upperBand, "Upper Band", color=color.gray)
plot(lowerBand, "Lower Band", color=color.gray)
'''


def main():
    """测试 VWAP 指标"""
    print("=" * 60)
    print("VWAP - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟价格数据
    base_price = 100.0
    returns = np.random.randn(n) * 0.01
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    volume = np.random.uniform(1000000, 5000000, n)

    # 计算 VWAP (无锚定重置)
    indicator = VWAPIndicator(anchor="none")
    result = indicator.calculate(high, low, close, volume)

    # 显示结果
    print(f"\n数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'VWAP':<10} {'Upper':<10} {'Lower':<10} {'Position':<10}")
    print("-" * 80)

    signals = indicator.get_signals(close, result)

    for i in range(n - 15, n):
        vwap_str = f"{result.vwap[i]:.2f}"
        upper_str = f"{result.upper_band[i]:.2f}"
        lower_str = f"{result.lower_band[i]:.2f}"

        if signals["above_vwap"][i]:
            pos = "Above"
        elif signals["below_vwap"][i]:
            pos = "Below"
        else:
            pos = "At"

        print(f"{i:<5} {close[i]:<10.2f} {vwap_str:<10} {upper_str:<10} {lower_str:<10} {pos:<10}")

    print()
    print("信号统计:")
    print(f"  高于 VWAP: {np.sum(signals['above_vwap'])} bars")
    print(f"  低于 VWAP: {np.sum(signals['below_vwap'])} bars")
    print(f"  上穿 VWAP: {np.sum(signals['cross_above'])}")
    print(f"  下穿 VWAP: {np.sum(signals['cross_below'])}")


if __name__ == "__main__":
    main()
