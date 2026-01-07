"""
CM Ultimate RSI Multi-TimeFrame [ChrisMoody] - Python Conversion

核心算法:
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss over N periods

    TradingView RSI 使用 RMA (Wilder's smoothing) 计算平均涨跌幅

特点:
    - 支持多时间框架 RSI
    - 超买/超卖区域可视化
    - 动量变化颜色指示
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RSIResult:
    """RSI 计算结果"""
    rsi: np.ndarray           # RSI 值
    avg_gain: np.ndarray      # 平均涨幅
    avg_loss: np.ndarray      # 平均跌幅
    overbought: np.ndarray    # 超买区域
    oversold: np.ndarray      # 超卖区域


class RSIIndicator:
    """
    RSI (Relative Strength Index) 指标

    使用 Wilder's smoothing (RMA) 方法，与 TradingView ta.rsi() 一致

    Parameters:
        period: RSI 周期 - 默认 14
        overbought: 超买阈值 - 默认 70
        oversold: 超卖阈值 - 默认 30
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
    ):
        self.period = period
        self.overbought_level = overbought
        self.oversold_level = oversold

    def _rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算 RMA (Wilder's Smoothed Moving Average)

        TradingView 公式:
            alpha = 1 / period
            rma[period-1] = sma(data[0:period])
            rma[i] = alpha * data[i] + (1 - alpha) * rma[i-1]
        """
        alpha = 1.0 / period
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        # 第一个值使用 SMA
        result[period - 1] = np.mean(data[:period])

        # 递归计算
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def calculate(self, close: np.ndarray) -> RSIResult:
        """
        计算 RSI

        算法步骤:
        1. 计算价格变化 change = close - close[1]
        2. 分离涨幅和跌幅:
           gain = max(change, 0)
           loss = max(-change, 0)
        3. 使用 RMA 平滑:
           avg_gain = rma(gain, period)
           avg_loss = rma(loss, period)
        4. 计算 RS 和 RSI:
           RS = avg_gain / avg_loss
           RSI = 100 - (100 / (1 + RS))

        Returns:
            RSIResult 包含所有计算结果
        """
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算价格变化
        change = np.zeros(n)
        change[1:] = close[1:] - close[:-1]

        # Step 2: 分离涨跌幅
        gain = np.maximum(change, 0)
        loss = np.maximum(-change, 0)

        # Step 3: RMA 平滑
        avg_gain = self._rma(gain, self.period)
        avg_loss = self._rma(loss, self.period)

        # Step 4: 计算 RSI
        # 避免除零
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi = 100 - (100 / (1 + rs))

        # 当 avg_loss 为 0 时，RSI = 100
        rsi = np.where(avg_loss == 0, 100, rsi)

        # 超买/超卖判断
        overbought = rsi > self.overbought_level
        oversold = rsi < self.oversold_level

        return RSIResult(
            rsi=rsi,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            overbought=overbought,
            oversold=oversold,
        )


class UltimateRSI:
    """
    CM Ultimate RSI MTF [ChrisMoody]

    多时间框架 RSI 指标，支持在当前时间框架显示更高时间框架的 RSI

    Features:
        - 标准 RSI 计算
        - 多时间框架支持
        - 超买/超卖区域可视化
        - 动量颜色指示
    """

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        extreme_ob: float = 80,
        extreme_os: float = 20,
    ):
        self.rsi_indicator = RSIIndicator(
            period=rsi_period,
            overbought=overbought,
            oversold=oversold,
        )
        self.extreme_ob = extreme_ob
        self.extreme_os = extreme_os

    def calculate(self, close: np.ndarray) -> dict:
        """
        计算 Ultimate RSI

        Returns:
            dict 包含:
            - rsi: RSI 值
            - zone: 区域 ('extreme_ob', 'ob', 'neutral', 'os', 'extreme_os')
            - momentum: 动量方向 (1=上升, -1=下降, 0=横盘)
            - color_intensity: 颜色强度 (0-1)
        """
        result = self.rsi_indicator.calculate(close)
        n = len(close)

        # 区域判断
        zone = np.full(n, "neutral", dtype=object)
        zone[result.rsi > self.extreme_ob] = "extreme_ob"
        zone[(result.rsi > self.rsi_indicator.overbought_level) &
             (result.rsi <= self.extreme_ob)] = "ob"
        zone[result.rsi < self.extreme_os] = "extreme_os"
        zone[(result.rsi < self.rsi_indicator.oversold_level) &
             (result.rsi >= self.extreme_os)] = "os"

        # 动量方向
        momentum = np.zeros(n, dtype=int)
        for i in range(1, n):
            if not np.isnan(result.rsi[i]) and not np.isnan(result.rsi[i-1]):
                if result.rsi[i] > result.rsi[i-1]:
                    momentum[i] = 1
                elif result.rsi[i] < result.rsi[i-1]:
                    momentum[i] = -1

        # 颜色强度 (基于距离中线 50 的距离)
        color_intensity = np.abs(result.rsi - 50) / 50
        color_intensity = np.clip(color_intensity, 0, 1)

        return {
            "rsi": result.rsi,
            "avg_gain": result.avg_gain,
            "avg_loss": result.avg_loss,
            "overbought": result.overbought,
            "oversold": result.oversold,
            "zone": zone,
            "momentum": momentum,
            "color_intensity": color_intensity,
        }

    def resample_to_mtf(
        self,
        close: pd.Series,
        higher_tf: str,
    ) -> pd.DataFrame:
        """
        重采样到更高时间框架并计算 RSI

        Parameters:
            close: 收盘价 Series (需要 datetime index)
            higher_tf: 目标时间框架 (如 '1H', '4H', 'D')

        Returns:
            DataFrame 包含重采样后的 RSI 值
        """
        # 重采样
        ohlc = close.resample(higher_tf).ohlc()
        htf_close = ohlc["close"]

        # 计算高时间框架 RSI
        htf_result = self.rsi_indicator.calculate(htf_close.values)

        # 创建高时间框架 DataFrame
        htf_df = pd.DataFrame({
            "rsi": htf_result.rsi,
        }, index=htf_close.index)

        # 向前填充到原始时间框架
        return htf_df.reindex(close.index, method="ffill")


class UltimateRSIPyneCore:
    """
    PyneCore 兼容的 Ultimate RSI 实现
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
    ):
        self.indicator = UltimateRSI(
            rsi_period=period,
            overbought=overbought,
            oversold=oversold,
        )

    def __call__(self, close: pd.Series) -> pd.DataFrame:
        """计算并返回 DataFrame 格式的结果"""
        result = self.indicator.calculate(close.values)

        return pd.DataFrame({
            "rsi": result["rsi"],
            "overbought": result["overbought"],
            "oversold": result["oversold"],
            "momentum": result["momentum"],
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_ULTIMATE_RSI = '''
//@version=5
indicator("CM Ultimate RSI MTF", overlay=false)

// 参数
src = input.source(close, "Source")
len = input.int(14, "RSI Period")
ob = input.float(70, "Overbought")
os = input.float(30, "Oversold")
extremeOB = input.float(80, "Extreme Overbought")
extremeOS = input.float(20, "Extreme Oversold")

// RSI 计算
rsi_value = ta.rsi(src, len)

// 动量判断
momentum = rsi_value > rsi_value[1] ? 1 : rsi_value < rsi_value[1] ? -1 : 0

// 区域颜色
rsiColor = rsi_value >= extremeOB ? color.new(color.red, 0) :
           rsi_value >= ob ? color.new(color.red, 50) :
           rsi_value <= extremeOS ? color.new(color.green, 0) :
           rsi_value <= os ? color.new(color.green, 50) :
           color.gray

// 绘图
hline(ob, "Overbought", color=color.red)
hline(os, "Oversold", color=color.green)
hline(50, "Middle", color=color.gray)
hline(extremeOB, "Extreme OB", color=color.red, linestyle=hline.style_dotted)
hline(extremeOS, "Extreme OS", color=color.green, linestyle=hline.style_dotted)

plot(rsi_value, "RSI", color=rsiColor, linewidth=2)

// 背景色
bgcolor(rsi_value >= ob ? color.new(color.red, 90) :
        rsi_value <= os ? color.new(color.green, 90) : na)
'''


def main():
    """测试 Ultimate RSI 指标"""
    print("=" * 60)
    print("CM Ultimate RSI MTF - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟价格数据
    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)

    # 计算 RSI
    indicator = UltimateRSI()
    result = indicator.calculate(close)

    # 显示结果
    print(f"\n参数: period={indicator.rsi_indicator.period}")
    print(f"超买: {indicator.rsi_indicator.overbought_level}, 超卖: {indicator.rsi_indicator.oversold_level}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 70)
    print(f"{'Bar':<5} {'Close':<10} {'RSI':<10} {'Zone':<12} {'Momentum':<10}")
    print("-" * 70)

    for i in range(n - 15, n):
        rsi_val = result["rsi"][i]
        rsi_str = f"{rsi_val:.2f}" if not np.isnan(rsi_val) else "NaN"
        zone = result["zone"][i]
        mom = result["momentum"][i]
        mom_str = "↑" if mom == 1 else "↓" if mom == -1 else "→"

        print(f"{i:<5} {close[i]:<10.2f} {rsi_str:<10} {zone:<12} {mom_str:<10}")

    print()
    print("区域统计:")
    zones, counts = np.unique(result["zone"], return_counts=True)
    for zone, count in zip(zones, counts):
        print(f"  {zone}: {count} bars")


if __name__ == "__main__":
    main()
