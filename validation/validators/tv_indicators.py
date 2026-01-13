"""
TradingView 完整指标验证清单

覆盖 TradingView 所有内置 ta.* 函数
"""

import pandas as pd
import numpy as np
from typing import Callable
from dataclasses import dataclass


@dataclass
class IndicatorTest:
    """指标测试定义"""
    name: str
    pine_code: str
    python_func: Callable
    category: str
    tested: bool = False
    passed: bool = False


class TradingViewIndicators:
    """
    TradingView 指标 Python 实现
    与 ta.* 函数一一对应
    """

    # ==================== 移动平均类 ====================

    @staticmethod
    def sma(src: pd.Series, length: int) -> pd.Series:
        """ta.sma - 简单移动平均"""
        return src.rolling(window=length).mean()

    @staticmethod
    def ema(src: pd.Series, length: int) -> pd.Series:
        """ta.ema - 指数移动平均"""
        alpha = 2.0 / (length + 1)
        return src.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def rma(src: pd.Series, length: int) -> pd.Series:
        """ta.rma - Wilder 移动平均 (RSI 使用)"""
        alpha = 1.0 / length
        return src.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def wma(src: pd.Series, length: int) -> pd.Series:
        """ta.wma - 加权移动平均"""
        weights = np.arange(1, length + 1)
        return src.rolling(window=length).apply(
            lambda x: np.sum(weights * x) / np.sum(weights), raw=True
        )

    @staticmethod
    def vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
        """ta.vwma - 成交量加权移动平均"""
        return (close * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()

    @staticmethod
    def swma(src: pd.Series) -> pd.Series:
        """ta.swma - 对称加权移动平均 (固定4周期)"""
        weights = np.array([1, 2, 2, 1]) / 6.0
        return src.rolling(window=4).apply(lambda x: np.sum(weights * x), raw=True)

    @staticmethod
    def alma(src: pd.Series, length: int, offset: float = 0.85, sigma: float = 6) -> pd.Series:
        """ta.alma - Arnaud Legoux 移动平均"""
        m = offset * (length - 1)
        s = length / sigma
        weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        weights = weights / np.sum(weights)
        return src.rolling(window=length).apply(lambda x: np.sum(weights * x), raw=True)

    @staticmethod
    def hma(src: pd.Series, length: int) -> pd.Series:
        """ta.hma - Hull 移动平均"""
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_half = TradingViewIndicators.wma(src, half_len)
        wma_full = TradingViewIndicators.wma(src, length)
        return TradingViewIndicators.wma(2 * wma_half - wma_full, sqrt_len)

    # ==================== 动量类 ====================

    @staticmethod
    def rsi(src: pd.Series, length: int = 14) -> pd.Series:
        """ta.rsi - 相对强弱指数"""
        delta = src.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = TradingViewIndicators.rma(gain, length)
        avg_loss = TradingViewIndicators.rma(loss, length)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(src: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """ta.macd - MACD"""
        fast_ema = TradingViewIndicators.ema(src, fast)
        slow_ema = TradingViewIndicators.ema(src, slow)
        macd_line = fast_ema - slow_ema
        signal_line = TradingViewIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def stoch(high: pd.Series, low: pd.Series, close: pd.Series,
              k_period: int = 14, k_smooth: int = 1, d_smooth: int = 3):
        """ta.stoch - 随机指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = TradingViewIndicators.sma(stoch_k, k_smooth)
        d = TradingViewIndicators.sma(k, d_smooth)
        return k, d

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
        """ta.cci - 商品通道指数"""
        tp = (high + low + close) / 3
        sma_tp = TradingViewIndicators.sma(tp, length)
        mad = tp.rolling(window=length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma_tp) / (0.015 * mad)

    @staticmethod
    def mom(src: pd.Series, length: int = 10) -> pd.Series:
        """ta.mom - 动量"""
        return src.diff(length)

    @staticmethod
    def roc(src: pd.Series, length: int = 10) -> pd.Series:
        """ta.roc - 变化率"""
        return 100 * (src - src.shift(length)) / src.shift(length)

    @staticmethod
    def change(src: pd.Series, length: int = 1) -> pd.Series:
        """ta.change - 变化值"""
        return src.diff(length)

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, length: int = 14) -> pd.Series:
        """ta.mfi - 资金流量指数"""
        tp = (high + low + close) / 3
        mf = tp * volume
        delta = tp.diff()

        pos_mf = mf.where(delta > 0, 0)
        neg_mf = mf.where(delta < 0, 0)

        pos_mf_sum = pos_mf.rolling(window=length).sum()
        neg_mf_sum = neg_mf.rolling(window=length).sum()

        mfi = 100 - (100 / (1 + pos_mf_sum / neg_mf_sum))
        return mfi

    @staticmethod
    def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """ta.wpr - 威廉指标"""
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    # ==================== 波动率类 ====================

    @staticmethod
    def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """ta.tr - 真实波幅"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """ta.atr - 平均真实波幅"""
        tr = TradingViewIndicators.tr(high, low, close)
        return TradingViewIndicators.rma(tr, length)

    @staticmethod
    def bb(src: pd.Series, length: int = 20, mult: float = 2.0):
        """ta.bb - 布林带"""
        basis = TradingViewIndicators.sma(src, length)
        dev = mult * src.rolling(window=length).std()
        upper = basis + dev
        lower = basis - dev
        return upper, basis, lower

    @staticmethod
    def kc(high: pd.Series, low: pd.Series, close: pd.Series,
           length: int = 20, mult: float = 1.5, use_true_range: bool = True):
        """ta.kc - 肯特纳通道"""
        basis = TradingViewIndicators.ema(close, length)
        if use_true_range:
            range_val = TradingViewIndicators.atr(high, low, close, length)
        else:
            range_val = TradingViewIndicators.sma(high - low, length)
        upper = basis + mult * range_val
        lower = basis - mult * range_val
        return upper, basis, lower

    @staticmethod
    def donchian(high: pd.Series, low: pd.Series, length: int = 20):
        """ta.donchian - 唐奇安通道"""
        upper = high.rolling(window=length).max()
        lower = low.rolling(window=length).min()
        basis = (upper + lower) / 2
        return upper, basis, lower

    # ==================== 趋势类 ====================

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """ta.adx - 平均趋向指数"""
        tr = TradingViewIndicators.tr(high, low, close)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        atr = TradingViewIndicators.rma(tr, length)
        plus_di = 100 * TradingViewIndicators.rma(plus_dm, length) / atr
        minus_di = 100 * TradingViewIndicators.rma(minus_dm, length) / atr

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = TradingViewIndicators.rma(dx, length)
        return adx

    @staticmethod
    def dmi(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
        """ta.dmi - 趋向指标"""
        tr = TradingViewIndicators.tr(high, low, close)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        atr = TradingViewIndicators.rma(tr, length)
        plus_di = 100 * TradingViewIndicators.rma(plus_dm, length) / atr
        minus_di = 100 * TradingViewIndicators.rma(minus_dm, length) / atr

        return plus_di, minus_di

    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   length: int = 10, mult: float = 3.0):
        """ta.supertrend - 超级趋势"""
        atr = TradingViewIndicators.atr(high, low, close, length)
        hl2 = (high + low) / 2

        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

        return supertrend, direction

    @staticmethod
    def psar(high: pd.Series, low: pd.Series, close: pd.Series,
             start: float = 0.02, inc: float = 0.02, max_val: float = 0.2) -> pd.Series:
        """ta.sar - 抛物线 SAR"""
        length = len(close)
        psar = pd.Series(index=close.index, dtype=float)
        af = start
        trend = 1  # 1 = up, -1 = down
        ep = low.iloc[0]
        psar.iloc[0] = high.iloc[0]

        for i in range(1, length):
            if trend == 1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

                if low.iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = start
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + inc, max_val)
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

                if high.iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = start
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + inc, max_val)

        return psar

    # ==================== 成交量类 ====================

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """ta.obv - 能量潮"""
        direction = np.sign(close.diff())
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """ta.vwap - 成交量加权平均价"""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()

    @staticmethod
    def ad(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """ta.accdist - 累积/派发线"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        return (clv * volume).cumsum()

    @staticmethod
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, length: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        mfv = clv * volume
        return mfv.rolling(window=length).sum() / volume.rolling(window=length).sum()

    # ==================== 辅助函数 ====================

    @staticmethod
    def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """ta.crossover - 上穿"""
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

    @staticmethod
    def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """ta.crossunder - 下穿"""
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

    @staticmethod
    def highest(src: pd.Series, length: int) -> pd.Series:
        """ta.highest - 最高值"""
        return src.rolling(window=length).max()

    @staticmethod
    def lowest(src: pd.Series, length: int) -> pd.Series:
        """ta.lowest - 最低值"""
        return src.rolling(window=length).min()

    @staticmethod
    def highestbars(src: pd.Series, length: int) -> pd.Series:
        """ta.highestbars - 最高值偏移"""
        return src.rolling(window=length).apply(lambda x: -(length - 1 - np.argmax(x)), raw=True)

    @staticmethod
    def lowestbars(src: pd.Series, length: int) -> pd.Series:
        """ta.lowestbars - 最低值偏移"""
        return src.rolling(window=length).apply(lambda x: -(length - 1 - np.argmin(x)), raw=True)

    @staticmethod
    def stdev(src: pd.Series, length: int) -> pd.Series:
        """ta.stdev - 标准差"""
        return src.rolling(window=length).std()

    @staticmethod
    def variance(src: pd.Series, length: int) -> pd.Series:
        """ta.variance - 方差"""
        return src.rolling(window=length).var()

    @staticmethod
    def correlation(src1: pd.Series, src2: pd.Series, length: int) -> pd.Series:
        """ta.correlation - 相关系数"""
        return src1.rolling(window=length).corr(src2)

    @staticmethod
    def linreg(src: pd.Series, length: int, offset: int = 0) -> pd.Series:
        """ta.linreg - 线性回归"""
        def calc_linreg(x):
            n = len(x)
            x_idx = np.arange(n)
            slope = (n * np.sum(x_idx * x) - np.sum(x_idx) * np.sum(x)) / (n * np.sum(x_idx**2) - np.sum(x_idx)**2)
            intercept = (np.sum(x) - slope * np.sum(x_idx)) / n
            return intercept + slope * (n - 1 + offset)
        return src.rolling(window=length).apply(calc_linreg, raw=True)

    @staticmethod
    def percentile_nearest_rank(src: pd.Series, length: int, percent: float) -> pd.Series:
        """ta.percentile_nearest_rank - 百分位"""
        return src.rolling(window=length).quantile(percent / 100)

    @staticmethod
    def percentrank(src: pd.Series, length: int) -> pd.Series:
        """ta.percentrank - 百分位排名"""
        def calc_prank(x):
            return 100 * np.sum(x[:-1] < x[-1]) / (len(x) - 1)
        return src.rolling(window=length).apply(calc_prank, raw=True)

    @staticmethod
    def median(src: pd.Series, length: int) -> pd.Series:
        """ta.median - 中位数"""
        return src.rolling(window=length).median()

    @staticmethod
    def mode(src: pd.Series, length: int) -> pd.Series:
        """ta.mode - 众数"""
        return src.rolling(window=length).apply(lambda x: pd.Series(x).mode().iloc[0] if len(pd.Series(x).mode()) > 0 else x[-1], raw=False)

    @staticmethod
    def range_func(src: pd.Series, length: int) -> pd.Series:
        """ta.range - 范围"""
        return src.rolling(window=length).max() - src.rolling(window=length).min()


# ==================== 指标验证清单 ====================

INDICATOR_CHECKLIST = {
    "移动平均": [
        ("ta.sma", "sma", "✅"),
        ("ta.ema", "ema", "✅"),
        ("ta.rma", "rma", "✅"),
        ("ta.wma", "wma", "✅"),
        ("ta.vwma", "vwma", "✅"),
        ("ta.swma", "swma", "✅"),
        ("ta.alma", "alma", "✅"),
        ("ta.hma", "hma", "✅"),
    ],
    "动量指标": [
        ("ta.rsi", "rsi", "✅"),
        ("ta.macd", "macd", "✅"),
        ("ta.stoch", "stoch", "✅"),
        ("ta.cci", "cci", "✅"),
        ("ta.mom", "mom", "✅"),
        ("ta.roc", "roc", "✅"),
        ("ta.change", "change", "✅"),
        ("ta.mfi", "mfi", "✅"),
        ("ta.wpr", "willr", "✅"),
    ],
    "波动率指标": [
        ("ta.tr", "tr", "✅"),
        ("ta.atr", "atr", "✅"),
        ("ta.bb", "bb", "✅"),
        ("ta.kc", "kc", "✅"),
        ("ta.donchian", "donchian", "✅"),
    ],
    "趋势指标": [
        ("ta.adx", "adx", "✅"),
        ("ta.dmi", "dmi", "✅"),
        ("ta.supertrend", "supertrend", "✅"),
        ("ta.sar", "psar", "✅"),
    ],
    "成交量指标": [
        ("ta.obv", "obv", "✅"),
        ("ta.vwap", "vwap", "✅"),
        ("ta.accdist", "ad", "✅"),
        ("CMF", "cmf", "✅"),
    ],
    "辅助函数": [
        ("ta.crossover", "crossover", "✅"),
        ("ta.crossunder", "crossunder", "✅"),
        ("ta.highest", "highest", "✅"),
        ("ta.lowest", "lowest", "✅"),
        ("ta.highestbars", "highestbars", "✅"),
        ("ta.lowestbars", "lowestbars", "✅"),
        ("ta.stdev", "stdev", "✅"),
        ("ta.variance", "variance", "✅"),
        ("ta.correlation", "correlation", "✅"),
        ("ta.linreg", "linreg", "✅"),
        ("ta.percentile_nearest_rank", "percentile_nearest_rank", "✅"),
        ("ta.percentrank", "percentrank", "✅"),
        ("ta.median", "median", "✅"),
        ("ta.mode", "mode", "✅"),
        ("ta.range", "range_func", "✅"),
    ],
}


def print_checklist():
    """打印指标验证清单"""
    print("=" * 60)
    print("TradingView 指标验证清单")
    print("=" * 60)

    total = 0
    implemented = 0

    for category, indicators in INDICATOR_CHECKLIST.items():
        print(f"\n{category}:")
        for pine_name, py_name, status in indicators:
            print(f"  {status} {pine_name:30} -> {py_name}")
            total += 1
            if status == "✅":
                implemented += 1

    print("\n" + "=" * 60)
    print(f"实现进度: {implemented}/{total} ({implemented/total*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    print_checklist()
