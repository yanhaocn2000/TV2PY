"""
Volume Indicators Collection - Python Conversion

包含:
- OBV (On Balance Volume) - 能量潮
- MFI (Money Flow Index) - 资金流量指数
- CMF (Chaikin Money Flow) - 蔡金资金流量
- A/D Line (Accumulation/Distribution Line) - 累积/派发线
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# OBV (On Balance Volume)
# ============================================================

@dataclass
class OBVResult:
    """OBV 计算结果"""
    obv: np.ndarray            # OBV 值
    obv_ema: np.ndarray        # OBV 的 EMA (信号线)
    direction: np.ndarray      # 方向 (1=上, -1=下, 0=不变)


class OBVIndicator:
    """
    OBV (On Balance Volume) - 能量潮

    由 Joseph Granville 发明，通过成交量累积来预测价格变化。

    核心算法:
        if close > close[1]: OBV = OBV[1] + volume
        if close < close[1]: OBV = OBV[1] - volume
        if close == close[1]: OBV = OBV[1]

    Parameters:
        signal_period: 信号线 EMA 周期 - 默认 21
    """

    def __init__(self, signal_period: int = 21):
        self.signal_period = signal_period

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 EMA"""
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def calculate(
        self,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> OBVResult:
        """计算 OBV"""
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = len(close)

        obv = np.zeros(n)
        direction = np.zeros(n, dtype=int)
        obv[0] = volume[0]

        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
                direction[i] = 1
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
                direction[i] = -1
            else:
                obv[i] = obv[i - 1]
                direction[i] = 0

        obv_ema = self._ema(obv, self.signal_period)

        return OBVResult(
            obv=obv,
            obv_ema=obv_ema,
            direction=direction,
        )


# ============================================================
# MFI (Money Flow Index)
# ============================================================

@dataclass
class MFIResult:
    """MFI 计算结果"""
    mfi: np.ndarray            # MFI 值
    money_flow: np.ndarray     # 资金流量
    overbought: np.ndarray     # 超买
    oversold: np.ndarray       # 超卖


class MFIIndicator:
    """
    MFI (Money Flow Index) - 资金流量指数

    结合价格和成交量的动量指标，也被称为"成交量加权 RSI"。

    核心算法:
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = sum of money_flow when tp > tp[1]
        negative_flow = sum of money_flow when tp < tp[1]
        money_ratio = positive_flow / negative_flow
        MFI = 100 - (100 / (1 + money_ratio))

    Parameters:
        period: MFI 周期 - 默认 14
        overbought: 超买阈值 - 默认 80
        oversold: 超卖阈值 - 默认 20
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 80,
        oversold: float = 20,
    ):
        self.period = period
        self.overbought_level = overbought
        self.oversold_level = oversold

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> MFIResult:
        """计算 MFI"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = len(close)

        # 典型价格
        tp = (high + low + close) / 3.0

        # 资金流量
        money_flow = tp * volume

        # 分类正负流量
        positive_flow = np.zeros(n)
        negative_flow = np.zeros(n)

        for i in range(1, n):
            if tp[i] > tp[i - 1]:
                positive_flow[i] = money_flow[i]
            elif tp[i] < tp[i - 1]:
                negative_flow[i] = money_flow[i]

        # 计算 MFI
        mfi = np.full(n, np.nan)

        for i in range(self.period, n):
            pos_sum = np.sum(positive_flow[i - self.period + 1:i + 1])
            neg_sum = np.sum(negative_flow[i - self.period + 1:i + 1])

            if neg_sum == 0:
                mfi[i] = 100
            else:
                money_ratio = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + money_ratio))

        overbought = mfi > self.overbought_level
        oversold = mfi < self.oversold_level

        return MFIResult(
            mfi=mfi,
            money_flow=money_flow,
            overbought=overbought,
            oversold=oversold,
        )


# ============================================================
# CMF (Chaikin Money Flow)
# ============================================================

@dataclass
class CMFResult:
    """CMF 计算结果"""
    cmf: np.ndarray            # CMF 值
    ad: np.ndarray             # 累积/派发值
    bullish: np.ndarray        # CMF > 0
    bearish: np.ndarray        # CMF < 0


class CMFIndicator:
    """
    CMF (Chaikin Money Flow) - 蔡金资金流量

    衡量一段时间内资金流入/流出的程度。

    核心算法:
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = multiplier * volume
        CMF = sum(money_flow_volume, period) / sum(volume, period)

    Parameters:
        period: CMF 周期 - 默认 20
    """

    def __init__(self, period: int = 20):
        self.period = period

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> CMFResult:
        """计算 CMF"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = len(close)

        # 资金流量乘数 (CLV - Close Location Value)
        hl_range = high - low
        clv = np.where(
            hl_range != 0,
            ((close - low) - (high - close)) / hl_range,
            0
        )

        # 资金流量成交量
        mfv = clv * volume

        # 计算 CMF
        cmf = np.full(n, np.nan)

        for i in range(self.period - 1, n):
            mfv_sum = np.sum(mfv[i - self.period + 1:i + 1])
            vol_sum = np.sum(volume[i - self.period + 1:i + 1])

            if vol_sum != 0:
                cmf[i] = mfv_sum / vol_sum

        bullish = cmf > 0
        bearish = cmf < 0

        return CMFResult(
            cmf=cmf,
            ad=mfv,  # 实际上是 MFV，但代表 AD 思想
            bullish=bullish,
            bearish=bearish,
        )


# ============================================================
# A/D Line (Accumulation/Distribution)
# ============================================================

@dataclass
class ADLineResult:
    """A/D Line 计算结果"""
    ad: np.ndarray             # A/D Line 值
    ad_ema: np.ndarray         # A/D Line 的 EMA


class ADLineIndicator:
    """
    A/D Line (Accumulation/Distribution Line) - 累积/派发线

    由 Marc Chaikin 发明，用于衡量累积买入或派发卖出的程度。

    核心算法:
        CLV = ((close - low) - (high - close)) / (high - low)
        A/D = cumsum(CLV * volume)

    Parameters:
        signal_period: 信号线 EMA 周期 - 默认 21
    """

    def __init__(self, signal_period: int = 21):
        self.signal_period = signal_period

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 EMA"""
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> ADLineResult:
        """计算 A/D Line"""
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)
        n = len(close)

        # CLV (Close Location Value)
        hl_range = high - low
        clv = np.where(
            hl_range != 0,
            ((close - low) - (high - close)) / hl_range,
            0
        )

        # A/D = cumsum(CLV * volume)
        ad = np.cumsum(clv * volume)

        # 信号线
        ad_ema = self._ema(ad, self.signal_period)

        return ADLineResult(
            ad=ad,
            ad_ema=ad_ema,
        )


# ============================================================
# PyneCore 兼容实现
# ============================================================

class OBVPyneCore:
    """PyneCore 兼容的 OBV 实现"""

    def __init__(self, signal_period: int = 21):
        self.indicator = OBVIndicator(signal_period=signal_period)

    def __call__(
        self,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(close.values, volume.values)
        return pd.DataFrame({
            "obv": result.obv,
            "obv_ema": result.obv_ema,
        }, index=close.index)


class MFIPyneCore:
    """PyneCore 兼容的 MFI 实现"""

    def __init__(self, period: int = 14):
        self.indicator = MFIIndicator(period=period)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(
            high.values, low.values, close.values, volume.values
        )
        return pd.DataFrame({
            "mfi": result.mfi,
            "overbought": result.overbought,
            "oversold": result.oversold,
        }, index=close.index)


class CMFPyneCore:
    """PyneCore 兼容的 CMF 实现"""

    def __init__(self, period: int = 20):
        self.indicator = CMFIndicator(period=period)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        result = self.indicator.calculate(
            high.values, low.values, close.values, volume.values
        )
        return pd.DataFrame({
            "cmf": result.cmf,
            "bullish": result.bullish,
            "bearish": result.bearish,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================

PINE_SCRIPT_VOLUME = '''
//@version=5
indicator("Volume Indicators", overlay=false)

// OBV
obv = ta.obv

// MFI
mfi = ta.mfi(close, 14)

// CMF (手动计算)
length = 20
ad = close == high and close == low or high == low ? 0 :
     ((2 * close - low - high) / (high - low)) * volume
cmf = math.sum(ad, length) / math.sum(volume, length)

// A/D Line
adLine = ta.cum(close == high and close == low or high == low ? 0 :
                ((2 * close - low - high) / (high - low)) * volume)

// 绘图
plot(obv, "OBV", color=color.blue)
plot(mfi, "MFI", color=color.orange)
plot(cmf * 100, "CMF", color=color.green)  // 乘以 100 方便显示
'''


def main():
    """测试成交量指标"""
    print("=" * 60)
    print("Volume Indicators - Python Implementation")
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
    volume = np.random.uniform(1000000, 5000000, n)

    # 计算各指标
    obv_indicator = OBVIndicator()
    obv_result = obv_indicator.calculate(close, volume)

    mfi_indicator = MFIIndicator()
    mfi_result = mfi_indicator.calculate(high, low, close, volume)

    cmf_indicator = CMFIndicator()
    cmf_result = cmf_indicator.calculate(high, low, close, volume)

    ad_indicator = ADLineIndicator()
    ad_result = ad_indicator.calculate(high, low, close, volume)

    # 显示结果
    print(f"\n数据点数: {n}")
    print()

    print("最后 10 个数据点:")
    print("-" * 90)
    print(f"{'Bar':<5} {'Close':<10} {'OBV':<15} {'MFI':<10} {'CMF':<10} {'A/D':<15}")
    print("-" * 90)

    for i in range(n - 10, n):
        obv_str = f"{obv_result.obv[i]:.0f}"
        mfi_str = f"{mfi_result.mfi[i]:.2f}" if not np.isnan(mfi_result.mfi[i]) else "NaN"
        cmf_str = f"{cmf_result.cmf[i]:.4f}" if not np.isnan(cmf_result.cmf[i]) else "NaN"
        ad_str = f"{ad_result.ad[i]:.0f}"

        print(f"{i:<5} {close[i]:<10.2f} {obv_str:<15} {mfi_str:<10} {cmf_str:<10} {ad_str:<15}")

    print()
    print("指标解读:")
    print(f"  OBV 方向: {'上升' if obv_result.obv[-1] > obv_result.obv[-10] else '下降'}")
    print(f"  MFI 状态: {'超买' if mfi_result.overbought[-1] else '超卖' if mfi_result.oversold[-1] else '中性'}")
    print(f"  CMF 方向: {'资金流入' if cmf_result.bullish[-1] else '资金流出'}")


if __name__ == "__main__":
    main()
