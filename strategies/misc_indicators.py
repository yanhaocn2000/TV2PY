"""
Miscellaneous Indicators Collection - Python Conversion

包含:
- Heikin Ashi
- TRIX
- Ultimate Oscillator
- Aroon
- Connors RSI
- Know Sure Thing (KST)
- Coppock Curve
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Heikin Ashi
# ============================================================

@dataclass
class HeikinAshiResult:
    """Heikin Ashi 计算结果"""
    ha_open: np.ndarray
    ha_high: np.ndarray
    ha_low: np.ndarray
    ha_close: np.ndarray
    trend: np.ndarray  # 1=上涨, -1=下跌


class HeikinAshi:
    """
    Heikin Ashi - 平均K线

    核心算法:
        HA_Close = (Open + High + Low + Close) / 4
        HA_Open = (HA_Open[1] + HA_Close[1]) / 2
        HA_High = max(High, HA_Open, HA_Close)
        HA_Low = min(Low, HA_Open, HA_Close)
    """

    def calculate(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> HeikinAshiResult:
        open_price = np.asarray(open_price, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        ha_close = (open_price + high + low + close) / 4
        ha_open = np.zeros(n)
        ha_high = np.zeros(n)
        ha_low = np.zeros(n)

        ha_open[0] = (open_price[0] + close[0]) / 2

        for i in range(1, n):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(high[i], ha_open[i], ha_close[i])
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])

        ha_high[0] = max(high[0], ha_open[0], ha_close[0])
        ha_low[0] = min(low[0], ha_open[0], ha_close[0])

        trend = np.where(ha_close > ha_open, 1, -1)

        return HeikinAshiResult(
            ha_open=ha_open,
            ha_high=ha_high,
            ha_low=ha_low,
            ha_close=ha_close,
            trend=trend,
        )


# ============================================================
# TRIX
# ============================================================

@dataclass
class TRIXResult:
    """TRIX 计算结果"""
    trix: np.ndarray
    signal: np.ndarray


class TRIXIndicator:
    """
    TRIX - Triple Exponential Average

    核心算法:
        EMA1 = EMA(close, period)
        EMA2 = EMA(EMA1, period)
        EMA3 = EMA(EMA2, period)
        TRIX = (EMA3 - EMA3[1]) / EMA3[1] * 100

    Parameters:
        period: TRIX 周期 - 默认 14
        signal_period: 信号线周期 - 默认 9
    """

    def __init__(self, period: int = 14, signal_period: int = 9):
        self.period = period
        self.signal_period = signal_period

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

    def calculate(self, close: np.ndarray) -> TRIXResult:
        close = np.asarray(close, dtype=float)
        n = len(close)

        ema1 = self._ema(close, self.period)
        ema2 = self._ema(ema1, self.period)
        ema3 = self._ema(ema2, self.period)

        trix = np.full(n, np.nan)
        for i in range(1, n):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100

        signal = self._ema(trix, self.signal_period)

        return TRIXResult(trix=trix, signal=signal)


# ============================================================
# Ultimate Oscillator
# ============================================================

@dataclass
class UltimateOscillatorResult:
    """Ultimate Oscillator 计算结果"""
    uo: np.ndarray
    overbought: np.ndarray
    oversold: np.ndarray


class UltimateOscillator:
    """
    Ultimate Oscillator

    由 Larry Williams 发明，结合三个不同周期的买入压力。

    核心算法:
        BP = Close - min(Low, Close[1])
        TR = max(High, Close[1]) - min(Low, Close[1])
        Average = BP_sum / TR_sum
        UO = 100 * (4*Avg7 + 2*Avg14 + Avg28) / (4 + 2 + 1)

    Parameters:
        period1: 短周期 - 默认 7
        period2: 中周期 - 默认 14
        period3: 长周期 - 默认 28
    """

    def __init__(
        self,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        overbought: float = 70,
        oversold: float = 30,
    ):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.overbought_level = overbought
        self.oversold_level = oversold

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> UltimateOscillatorResult:
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 买入压力
        bp = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            bp[i] = close[i] - min(low[i], close[i-1])
            tr[i] = max(high[i], close[i-1]) - min(low[i], close[i-1])

        def rolling_sum(data, period):
            result = np.full(n, np.nan)
            for i in range(period, n):
                result[i] = np.sum(data[i - period + 1:i + 1])
            return result

        bp1 = rolling_sum(bp, self.period1)
        tr1 = rolling_sum(tr, self.period1)
        bp2 = rolling_sum(bp, self.period2)
        tr2 = rolling_sum(tr, self.period2)
        bp3 = rolling_sum(bp, self.period3)
        tr3 = rolling_sum(tr, self.period3)

        avg1 = np.where(tr1 != 0, bp1 / tr1, 0)
        avg2 = np.where(tr2 != 0, bp2 / tr2, 0)
        avg3 = np.where(tr3 != 0, bp3 / tr3, 0)

        uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return UltimateOscillatorResult(
            uo=uo,
            overbought=uo > self.overbought_level,
            oversold=uo < self.oversold_level,
        )


# ============================================================
# Aroon
# ============================================================

@dataclass
class AroonResult:
    """Aroon 计算结果"""
    aroon_up: np.ndarray
    aroon_down: np.ndarray
    aroon_osc: np.ndarray


class AroonIndicator:
    """
    Aroon Indicator

    核心算法:
        Aroon Up = ((period - bars since highest high) / period) * 100
        Aroon Down = ((period - bars since lowest low) / period) * 100
        Aroon Oscillator = Aroon Up - Aroon Down

    Parameters:
        period: Aroon 周期 - 默认 25
    """

    def __init__(self, period: int = 25):
        self.period = period

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> AroonResult:
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        n = len(high)

        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)

        for i in range(self.period, n):
            window_high = high[i - self.period:i + 1]
            window_low = low[i - self.period:i + 1]

            # 找到最高点和最低点的位置
            bars_since_high = self.period - np.argmax(window_high)
            bars_since_low = self.period - np.argmin(window_low)

            aroon_up[i] = (self.period - bars_since_high) / self.period * 100
            aroon_down[i] = (self.period - bars_since_low) / self.period * 100

        aroon_osc = aroon_up - aroon_down

        return AroonResult(
            aroon_up=aroon_up,
            aroon_down=aroon_down,
            aroon_osc=aroon_osc,
        )


# ============================================================
# Connors RSI
# ============================================================

@dataclass
class ConnorsRSIResult:
    """Connors RSI 计算结果"""
    crsi: np.ndarray
    rsi: np.ndarray
    streak_rsi: np.ndarray
    percent_rank: np.ndarray


class ConnorsRSI:
    """
    Connors RSI

    由 Larry Connors 发明，结合三个组件:
    1. 传统 RSI
    2. 连续涨跌天数的 RSI
    3. 价格变化的百分位排名

    CRSI = (RSI + StreakRSI + PercentRank) / 3
    """

    def __init__(
        self,
        rsi_period: int = 3,
        streak_period: int = 2,
        rank_period: int = 100,
    ):
        self.rsi_period = rsi_period
        self.streak_period = streak_period
        self.rank_period = rank_period

    def _rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        change = np.zeros(n)
        change[1:] = data[1:] - data[:-1]

        gain = np.maximum(change, 0)
        loss = np.maximum(-change, 0)

        alpha = 1.0 / period
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)

        if n >= period:
            avg_gain[period - 1] = np.mean(gain[:period])
            avg_loss[period - 1] = np.mean(loss[:period])

            for i in range(period, n):
                avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
                avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate(self, close: np.ndarray) -> ConnorsRSIResult:
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 1. 传统 RSI
        rsi = self._rsi(close, self.rsi_period)

        # 2. 连续涨跌天数
        streak = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
            elif close[i] < close[i - 1]:
                streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
            else:
                streak[i] = 0

        streak_rsi = self._rsi(streak, self.streak_period)

        # 3. 百分位排名
        percent_rank = np.full(n, np.nan)
        change = np.zeros(n)
        change[1:] = close[1:] - close[:-1]

        for i in range(self.rank_period, n):
            window = change[i - self.rank_period + 1:i + 1]
            count_less = np.sum(window < change[i])
            percent_rank[i] = count_less / self.rank_period * 100

        # Connors RSI
        crsi = (rsi + streak_rsi + percent_rank) / 3

        return ConnorsRSIResult(
            crsi=crsi,
            rsi=rsi,
            streak_rsi=streak_rsi,
            percent_rank=percent_rank,
        )


# ============================================================
# Know Sure Thing (KST)
# ============================================================

@dataclass
class KSTResult:
    """KST 计算结果"""
    kst: np.ndarray
    signal: np.ndarray


class KSTIndicator:
    """
    Know Sure Thing (KST)

    由 Martin Pring 发明，综合多个 ROC 指标。

    核心算法:
        ROC1 = ROC(close, roc1) smoothed by SMA(sma1)
        ROC2 = ROC(close, roc2) smoothed by SMA(sma2)
        ROC3 = ROC(close, roc3) smoothed by SMA(sma3)
        ROC4 = ROC(close, roc4) smoothed by SMA(sma4)
        KST = ROC1 + 2*ROC2 + 3*ROC3 + 4*ROC4
    """

    def __init__(
        self,
        roc1: int = 10, sma1: int = 10,
        roc2: int = 15, sma2: int = 10,
        roc3: int = 20, sma3: int = 10,
        roc4: int = 30, sma4: int = 15,
        signal_period: int = 9,
    ):
        self.roc1 = roc1
        self.sma1 = sma1
        self.roc2 = roc2
        self.sma2 = sma2
        self.roc3 = roc3
        self.sma3 = sma3
        self.roc4 = roc4
        self.sma4 = sma4
        self.signal_period = signal_period

    def _roc(self, data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        result = np.full(n, np.nan)
        for i in range(period, n):
            if data[i - period] != 0:
                result[i] = (data[i] - data[i - period]) / data[i - period] * 100
        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid = data[i - period + 1:i + 1]
            valid = valid[~np.isnan(valid)]
            if len(valid) >= period:
                result[i] = np.mean(valid)
        return result

    def calculate(self, close: np.ndarray) -> KSTResult:
        close = np.asarray(close, dtype=float)

        roc1 = self._sma(self._roc(close, self.roc1), self.sma1)
        roc2 = self._sma(self._roc(close, self.roc2), self.sma2)
        roc3 = self._sma(self._roc(close, self.roc3), self.sma3)
        roc4 = self._sma(self._roc(close, self.roc4), self.sma4)

        kst = roc1 + 2 * roc2 + 3 * roc3 + 4 * roc4
        signal = self._sma(kst, self.signal_period)

        return KSTResult(kst=kst, signal=signal)


# ============================================================
# Coppock Curve
# ============================================================

@dataclass
class CoppockResult:
    """Coppock Curve 计算结果"""
    coppock: np.ndarray
    signal: np.ndarray  # 零线穿越


class CoppockCurve:
    """
    Coppock Curve

    由 Edwin Coppock 发明，长期动量指标。

    核心算法:
        Coppock = WMA(ROC(close, 14) + ROC(close, 11), 10)
    """

    def __init__(
        self,
        roc_long: int = 14,
        roc_short: int = 11,
        wma_period: int = 10,
    ):
        self.roc_long = roc_long
        self.roc_short = roc_short
        self.wma_period = wma_period

    def _roc(self, data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        result = np.full(n, np.nan)
        for i in range(period, n):
            if data[i - period] != 0:
                result[i] = (data[i] - data[i - period]) / data[i - period] * 100
        return result

    def _wma(self, data: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(data, np.nan, dtype=float)
        weights = np.arange(1, period + 1)
        weight_sum = np.sum(weights)

        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(weights * window) / weight_sum
        return result

    def calculate(self, close: np.ndarray) -> CoppockResult:
        close = np.asarray(close, dtype=float)
        n = len(close)

        roc_long = self._roc(close, self.roc_long)
        roc_short = self._roc(close, self.roc_short)

        roc_sum = roc_long + roc_short
        coppock = self._wma(roc_sum, self.wma_period)

        # 零线穿越信号
        signal = np.zeros(n, dtype=int)
        for i in range(1, n):
            if not np.isnan(coppock[i]) and not np.isnan(coppock[i-1]):
                if coppock[i] > 0 and coppock[i-1] <= 0:
                    signal[i] = 1  # 买入
                elif coppock[i] < 0 and coppock[i-1] >= 0:
                    signal[i] = -1  # 卖出

        return CoppockResult(coppock=coppock, signal=signal)


def main():
    print("=" * 60)
    print("Miscellaneous Indicators - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 100

    base_price = 100.0
    returns = np.random.randn(n) * 0.02
    close = base_price * np.cumprod(1 + returns)
    open_price = close * (1 + np.random.randn(n) * 0.005)
    high = np.maximum(open_price, close) * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = np.minimum(open_price, close) * (1 - np.abs(np.random.randn(n)) * 0.01)

    print("\n1. Heikin Ashi:")
    ha = HeikinAshi()
    ha_result = ha.calculate(open_price, high, low, close)
    print(f"   最后一根 HA: O={ha_result.ha_open[-1]:.2f}, H={ha_result.ha_high[-1]:.2f}, "
          f"L={ha_result.ha_low[-1]:.2f}, C={ha_result.ha_close[-1]:.2f}")

    print("\n2. TRIX:")
    trix = TRIXIndicator()
    trix_result = trix.calculate(close)
    print(f"   最后值: TRIX={trix_result.trix[-1]:.4f}, Signal={trix_result.signal[-1]:.4f}")

    print("\n3. Ultimate Oscillator:")
    uo = UltimateOscillator()
    uo_result = uo.calculate(high, low, close)
    print(f"   最后值: UO={uo_result.uo[-1]:.2f}")

    print("\n4. Aroon:")
    aroon = AroonIndicator()
    aroon_result = aroon.calculate(high, low)
    print(f"   最后值: Up={aroon_result.aroon_up[-1]:.2f}, Down={aroon_result.aroon_down[-1]:.2f}")

    print("\n5. Connors RSI:")
    crsi = ConnorsRSI()
    crsi_result = crsi.calculate(close)
    print(f"   最后值: CRSI={crsi_result.crsi[-1]:.2f}")

    print("\n6. KST:")
    kst = KSTIndicator()
    kst_result = kst.calculate(close)
    print(f"   最后值: KST={kst_result.kst[-1]:.4f}")

    print("\n7. Coppock Curve:")
    coppock = CoppockCurve()
    coppock_result = coppock.calculate(close)
    print(f"   最后值: Coppock={coppock_result.coppock[-1]:.4f}")


if __name__ == "__main__":
    main()
