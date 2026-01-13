"""
ATR (Average True Range) - Python Conversion

TradingView 内置指标: ta.atr(), ta.tr()

核心算法:
    TR = max(high - low, abs(high - close[1]), abs(low - close[1]))
    ATR = RMA(TR, period)  # 使用 Wilder's smoothing

用途:
    - 衡量波动率
    - 设置止损/止盈 (通常为 2-3 ATR)
    - 仓位管理
    - 布林带/肯特纳通道的组成部分
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ATRResult:
    """ATR 计算结果"""
    atr: np.ndarray            # ATR 值
    tr: np.ndarray             # True Range
    atr_percent: np.ndarray    # ATR 百分比 (ATR / close * 100)
    volatility_zone: np.ndarray  # 波动率区域


class ATRIndicator:
    """
    ATR (Average True Range)

    由 J. Welles Wilder 发明，用于衡量市场波动率。

    Parameters:
        period: ATR 周期 - 默认 14
        smoothing: 平滑方法 ('rma', 'sma', 'ema') - 默认 'rma'
    """

    def __init__(
        self,
        period: int = 14,
        smoothing: str = "rma",
    ):
        self.period = period
        self.smoothing = smoothing.lower()

    def _rma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 RMA (Wilder's Smoothed Moving Average)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan, dtype=float)

        if len(data) < period:
            return result

        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

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

    def calculate_tr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """
        计算 True Range

        TR = max(
            high - low,
            abs(high - close[1]),
            abs(low - close[1])
        )
        """
        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        return tr

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> ATRResult:
        """
        计算 ATR

        Returns:
            ATRResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 计算 True Range
        tr = self.calculate_tr(high, low, close)

        # 根据平滑方法计算 ATR
        if self.smoothing == "rma":
            atr = self._rma(tr, self.period)
        elif self.smoothing == "sma":
            atr = self._sma(tr, self.period)
        elif self.smoothing == "ema":
            atr = self._ema(tr, self.period)
        else:
            atr = self._rma(tr, self.period)

        # 计算 ATR 百分比
        atr_percent = np.where(close > 0, atr / close * 100, 0)

        # 计算波动率区域
        volatility_zone = np.full(n, "normal", dtype=object)
        atr_ma = self._sma(atr, self.period * 2)

        for i in range(len(atr)):
            if not np.isnan(atr[i]) and not np.isnan(atr_ma[i]):
                if atr[i] > atr_ma[i] * 1.5:
                    volatility_zone[i] = "high"
                elif atr[i] < atr_ma[i] * 0.5:
                    volatility_zone[i] = "low"

        return ATRResult(
            atr=atr,
            tr=tr,
            atr_percent=atr_percent,
            volatility_zone=volatility_zone,
        )

    def calculate_stop_loss(
        self,
        close: float,
        atr: float,
        multiplier: float = 2.0,
        direction: str = "long",
    ) -> float:
        """
        计算基于 ATR 的止损价位

        Parameters:
            close: 当前收盘价
            atr: 当前 ATR 值
            multiplier: ATR 乘数 - 默认 2.0
            direction: 交易方向 ('long' or 'short')

        Returns:
            止损价位
        """
        if direction == "long":
            return close - (atr * multiplier)
        else:
            return close + (atr * multiplier)

    def calculate_position_size(
        self,
        account_size: float,
        risk_percent: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """
        计算基于 ATR 的仓位大小

        Parameters:
            account_size: 账户总资金
            risk_percent: 风险百分比 (如 0.01 = 1%)
            atr: 当前 ATR 值
            atr_multiplier: ATR 乘数

        Returns:
            建议仓位大小 (股数/合约数)
        """
        risk_amount = account_size * risk_percent
        stop_distance = atr * atr_multiplier
        return risk_amount / stop_distance if stop_distance > 0 else 0


class ATRPyneCore:
    """PyneCore 兼容的 ATR 实现"""

    def __init__(
        self,
        period: int = 14,
        smoothing: str = "rma",
    ):
        self.indicator = ATRIndicator(period=period, smoothing=smoothing)

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
            "atr": result.atr,
            "tr": result.tr,
            "atr_percent": result.atr_percent,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_ATR = '''
//@version=5
indicator("ATR", overlay=false)

// 参数
length = input.int(14, "ATR Period")

// 计算
tr = ta.tr(true)
atr = ta.atr(length)
atr_percent = atr / close * 100

// 绘图
plot(atr, "ATR", color=color.blue, linewidth=2)

// ATR 百分比 (另一个面板)
// plot(atr_percent, "ATR %", color=color.orange)
'''


def main():
    """测试 ATR 指标"""
    print("=" * 60)
    print("ATR - Python Implementation")
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

    # 计算 ATR
    indicator = ATRIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: period={indicator.period}, smoothing={indicator.smoothing}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 70)
    print(f"{'Bar':<5} {'Close':<10} {'TR':<10} {'ATR':<10} {'ATR%':<10} {'Zone':<10}")
    print("-" * 70)

    for i in range(n - 15, n):
        atr_str = f"{result.atr[i]:.4f}" if not np.isnan(result.atr[i]) else "NaN"
        atr_pct = f"{result.atr_percent[i]:.2f}%" if not np.isnan(result.atr_percent[i]) else "NaN"

        print(f"{i:<5} {close[i]:<10.2f} {result.tr[i]:<10.4f} {atr_str:<10} {atr_pct:<10} {result.volatility_zone[i]:<10}")

    print()
    print("止损计算示例 (最后一根K线):")
    last_close = close[-1]
    last_atr = result.atr[-1]
    print(f"  当前价格: {last_close:.2f}")
    print(f"  当前 ATR: {last_atr:.4f}")
    print(f"  多头止损 (2x ATR): {indicator.calculate_stop_loss(last_close, last_atr, 2.0, 'long'):.2f}")
    print(f"  空头止损 (2x ATR): {indicator.calculate_stop_loss(last_close, last_atr, 2.0, 'short'):.2f}")

    print()
    print("仓位计算示例:")
    pos_size = indicator.calculate_position_size(
        account_size=100000,
        risk_percent=0.01,
        atr=last_atr,
        atr_multiplier=2.0
    )
    print(f"  账户: $100,000, 风险: 1%, ATR乘数: 2.0")
    print(f"  建议仓位: {pos_size:.0f} 股")


if __name__ == "__main__":
    main()
