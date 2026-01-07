"""
ADX (Average Directional Index) - Python Conversion

TradingView 内置指标: ta.adx(), ta.dmi()

核心算法:
    +DM = max(high - high[1], 0) if (high - high[1]) > (low[1] - low) else 0
    -DM = max(low[1] - low, 0) if (low[1] - low) > (high - high[1]) else 0
    TR = max(high - low, abs(high - close[1]), abs(low - close[1]))

    +DI = 100 * RMA(+DM) / RMA(TR)
    -DI = 100 * RMA(-DM) / RMA(TR)
    DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    ADX = RMA(DX)

信号:
    - ADX > 25: 趋势明确
    - +DI > -DI: 上涨趋势
    - -DI > +DI: 下跌趋势
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ADXResult:
    """ADX 计算结果"""
    adx: np.ndarray            # ADX 线
    plus_di: np.ndarray        # +DI 线
    minus_di: np.ndarray       # -DI 线
    dx: np.ndarray             # DX (未平滑的方向指数)
    plus_dm: np.ndarray        # +DM
    minus_dm: np.ndarray       # -DM
    tr: np.ndarray             # True Range
    strong_trend: np.ndarray   # ADX > 阈值
    uptrend: np.ndarray        # +DI > -DI
    downtrend: np.ndarray      # -DI > +DI


class ADXIndicator:
    """
    ADX (Average Directional Index)

    由 J. Welles Wilder 发明，用于衡量趋势强度。

    Parameters:
        period: ADX 周期 - 默认 14
        adx_smoothing: ADX 平滑周期 - 默认 14
        threshold: 趋势强度阈值 - 默认 25
    """

    def __init__(
        self,
        period: int = 14,
        adx_smoothing: int = 14,
        threshold: float = 25,
    ):
        self.period = period
        self.adx_smoothing = adx_smoothing
        self.threshold = threshold

    def _rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算 RMA (Wilder's Smoothed Moving Average)
        """
        alpha = 1.0 / period
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        # 使用 SMA 作为种子
        result[period - 1] = np.mean(data[:period])

        # 递归计算
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> ADXResult:
        """
        计算 ADX 和 DMI

        算法步骤:
        1. 计算 +DM 和 -DM
        2. 计算 True Range
        3. 计算 +DI 和 -DI
        4. 计算 DX
        5. 计算 ADX (平滑后的 DX)

        Returns:
            ADXResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算 +DM 和 -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0

            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0

        # Step 2: 计算 True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # Step 3: 计算平滑后的 +DM, -DM, TR
        smooth_plus_dm = self._rma(plus_dm, self.period)
        smooth_minus_dm = self._rma(minus_dm, self.period)
        smooth_tr = self._rma(tr, self.period)

        # Step 4: 计算 +DI 和 -DI
        plus_di = np.where(smooth_tr > 0, 100 * smooth_plus_dm / smooth_tr, 0)
        minus_di = np.where(smooth_tr > 0, 100 * smooth_minus_dm / smooth_tr, 0)

        # Step 5: 计算 DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.where(di_sum > 0, 100 * di_diff / di_sum, 0)

        # Step 6: 计算 ADX
        adx = self._rma(dx, self.adx_smoothing)

        # 趋势判断
        strong_trend = adx > self.threshold
        uptrend = plus_di > minus_di
        downtrend = minus_di > plus_di

        return ADXResult(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            dx=dx,
            plus_dm=plus_dm,
            minus_dm=minus_dm,
            tr=tr,
            strong_trend=strong_trend,
            uptrend=uptrend,
            downtrend=downtrend,
        )

    def get_signals(self, result: ADXResult) -> dict:
        """
        获取交易信号

        买入: +DI 上穿 -DI 且 ADX > 阈值
        卖出: -DI 上穿 +DI 且 ADX > 阈值
        """
        n = len(result.adx)

        # 计算穿越信号
        di_cross_up = np.zeros(n, dtype=bool)
        di_cross_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(result.plus_di[i]) and not np.isnan(result.minus_di[i]):
                # +DI 上穿 -DI
                if (result.plus_di[i] > result.minus_di[i] and
                    result.plus_di[i-1] <= result.minus_di[i-1]):
                    di_cross_up[i] = True
                # -DI 上穿 +DI
                if (result.minus_di[i] > result.plus_di[i] and
                    result.minus_di[i-1] <= result.plus_di[i-1]):
                    di_cross_down[i] = True

        # 结合 ADX 强度
        buy_signal = di_cross_up & result.strong_trend
        sell_signal = di_cross_down & result.strong_trend

        return {
            "buy": buy_signal,
            "sell": sell_signal,
            "di_cross_up": di_cross_up,
            "di_cross_down": di_cross_down,
            "strong_uptrend": result.strong_trend & result.uptrend,
            "strong_downtrend": result.strong_trend & result.downtrend,
        }


class ADXPyneCore:
    """PyneCore 兼容的 ADX 实现"""

    def __init__(
        self,
        period: int = 14,
        adx_smoothing: int = 14,
    ):
        self.indicator = ADXIndicator(period=period, adx_smoothing=adx_smoothing)

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.DataFrame:
        """计算并返回 DataFrame 格式的结果"""
        result = self.indicator.calculate(
            high.values,
            low.values,
            close.values,
        )

        return pd.DataFrame({
            "adx": result.adx,
            "plus_di": result.plus_di,
            "minus_di": result.minus_di,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_ADX = '''
//@version=5
indicator("ADX", overlay=false)

// 参数
adxlen = input.int(14, "ADX Smoothing")
dilen = input.int(14, "DI Period")

// 计算
[diplus, diminus, adx] = ta.dmi(dilen, adxlen)

// 绘图
plot(adx, "ADX", color=color.yellow, linewidth=2)
plot(diplus, "+DI", color=color.green)
plot(diminus, "-DI", color=color.red)
hline(25, "Threshold", color=color.gray)
'''


def main():
    """测试 ADX 指标"""
    print("=" * 60)
    print("ADX - Python Implementation")
    print("=" * 60)

    # 生成测试数据 (带有趋势)
    np.random.seed(42)
    n = 100

    # 模拟有趋势的价格数据
    base_price = 100.0
    trend = np.linspace(0, 10, n)  # 上涨趋势
    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5

    # 计算 ADX
    indicator = ADXIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: period={indicator.period}, adx_smoothing={indicator.adx_smoothing}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'ADX':<10} {'+DI':<10} {'-DI':<10} {'Trend':<15}")
    print("-" * 80)

    for i in range(n - 15, n):
        adx_str = f"{result.adx[i]:.2f}" if not np.isnan(result.adx[i]) else "NaN"
        pdi_str = f"{result.plus_di[i]:.2f}" if not np.isnan(result.plus_di[i]) else "NaN"
        mdi_str = f"{result.minus_di[i]:.2f}" if not np.isnan(result.minus_di[i]) else "NaN"

        if result.strong_trend[i]:
            if result.uptrend[i]:
                trend_str = "Strong UP"
            else:
                trend_str = "Strong DOWN"
        else:
            if result.uptrend[i]:
                trend_str = "Weak UP"
            else:
                trend_str = "Weak DOWN"

        print(f"{i:<5} {close[i]:<10.2f} {adx_str:<10} {pdi_str:<10} {mdi_str:<10} {trend_str:<15}")

    print()
    signals = indicator.get_signals(result)
    print("信号统计:")
    print(f"  买入信号: {np.sum(signals['buy'])}")
    print(f"  卖出信号: {np.sum(signals['sell'])}")
    print(f"  强势上涨: {np.sum(signals['strong_uptrend'])} bars")
    print(f"  强势下跌: {np.sum(signals['strong_downtrend'])} bars")


if __name__ == "__main__":
    main()
