"""
CCI (Commodity Channel Index) - Python Conversion

TradingView 内置指标: ta.cci()

核心算法:
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, period)
    mean_dev = mean(abs(typical_price - sma_tp), period)
    CCI = (typical_price - sma_tp) / (0.015 * mean_dev)

信号:
    - CCI > 100: 超买
    - CCI < -100: 超卖
    - CCI 从负值穿越 0: 可能的买入信号
    - CCI 从正值穿越 0: 可能的卖出信号
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CCIResult:
    """CCI 计算结果"""
    cci: np.ndarray            # CCI 值
    typical_price: np.ndarray  # 典型价格
    sma_tp: np.ndarray         # 典型价格的 SMA
    mean_dev: np.ndarray       # 平均偏差
    overbought: np.ndarray     # 超买
    oversold: np.ndarray       # 超卖
    zero_cross_up: np.ndarray  # 上穿零线
    zero_cross_down: np.ndarray  # 下穿零线


class CCIIndicator:
    """
    CCI (Commodity Channel Index)

    由 Donald Lambert 发明，最初用于商品市场，现广泛用于各种市场。

    Parameters:
        period: CCI 周期 - 默认 20
        constant: Lambert 常数 - 默认 0.015
        overbought: 超买阈值 - 默认 100
        oversold: 超卖阈值 - 默认 -100
    """

    def __init__(
        self,
        period: int = 20,
        constant: float = 0.015,
        overbought: float = 100,
        oversold: float = -100,
    ):
        self.period = period
        self.constant = constant
        self.overbought_level = overbought
        self.oversold_level = oversold

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def _mean_deviation(
        self,
        data: np.ndarray,
        mean: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """
        计算平均偏差

        mean_dev = mean(abs(data - mean), period)
        """
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            if not np.isnan(mean[i]):
                deviations = np.abs(data[i - period + 1:i + 1] - mean[i])
                result[i] = np.mean(deviations)
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> CCIResult:
        """
        计算 CCI

        算法步骤:
        1. 计算典型价格 = (high + low + close) / 3
        2. 计算典型价格的 SMA
        3. 计算平均偏差
        4. CCI = (tp - sma_tp) / (constant * mean_dev)

        Returns:
            CCIResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算典型价格
        typical_price = (high + low + close) / 3.0

        # Step 2: 计算典型价格的 SMA
        sma_tp = self._sma(typical_price, self.period)

        # Step 3: 计算平均偏差
        mean_dev = self._mean_deviation(typical_price, sma_tp, self.period)

        # Step 4: 计算 CCI
        cci = np.where(
            mean_dev != 0,
            (typical_price - sma_tp) / (self.constant * mean_dev),
            0
        )

        # 超买/超卖判断
        overbought = cci > self.overbought_level
        oversold = cci < self.oversold_level

        # 零线穿越
        zero_cross_up = np.zeros(n, dtype=bool)
        zero_cross_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(cci[i]) and not np.isnan(cci[i-1]):
                zero_cross_up[i] = cci[i] > 0 and cci[i-1] <= 0
                zero_cross_down[i] = cci[i] < 0 and cci[i-1] >= 0

        return CCIResult(
            cci=cci,
            typical_price=typical_price,
            sma_tp=sma_tp,
            mean_dev=mean_dev,
            overbought=overbought,
            oversold=oversold,
            zero_cross_up=zero_cross_up,
            zero_cross_down=zero_cross_down,
        )

    def get_signals(self, result: CCIResult) -> dict:
        """
        获取交易信号

        多种信号策略:
        1. 超买超卖反转
        2. 零线穿越
        3. 背离 (需要额外处理)
        """
        n = len(result.cci)

        # 从超卖区反弹买入
        buy_oversold = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if result.oversold[i-1] and not result.oversold[i] and result.cci[i] > result.cci[i-1]:
                buy_oversold[i] = True

        # 从超买区回落卖出
        sell_overbought = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if result.overbought[i-1] and not result.overbought[i] and result.cci[i] < result.cci[i-1]:
                sell_overbought[i] = True

        return {
            "buy_oversold": buy_oversold,
            "sell_overbought": sell_overbought,
            "zero_cross_up": result.zero_cross_up,
            "zero_cross_down": result.zero_cross_down,
        }


class CCIPyneCore:
    """PyneCore 兼容的 CCI 实现"""

    def __init__(
        self,
        period: int = 20,
        overbought: float = 100,
        oversold: float = -100,
    ):
        self.indicator = CCIIndicator(
            period=period,
            overbought=overbought,
            oversold=oversold,
        )

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
            "cci": result.cci,
            "overbought": result.overbought,
            "oversold": result.oversold,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_CCI = '''
//@version=5
indicator("CCI", overlay=false)

// 参数
length = input.int(20, "CCI Period")
src = input.source(hlc3, "Source")

// 计算
cci_value = ta.cci(src, length)

// 绘图
plot(cci_value, "CCI", color=color.blue, linewidth=2)
hline(100, "Overbought", color=color.red)
hline(-100, "Oversold", color=color.green)
hline(0, "Zero", color=color.gray)

// 背景色
bgcolor(cci_value > 100 ? color.new(color.red, 90) :
        cci_value < -100 ? color.new(color.green, 90) : na)
'''


def main():
    """测试 CCI 指标"""
    print("=" * 60)
    print("CCI - Python Implementation")
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

    # 计算 CCI
    indicator = CCIIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: period={indicator.period}")
    print(f"超买: {indicator.overbought_level}, 超卖: {indicator.oversold_level}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 70)
    print(f"{'Bar':<5} {'Close':<10} {'CCI':<12} {'Zone':<12} {'Signal':<15}")
    print("-" * 70)

    signals = indicator.get_signals(result)

    for i in range(n - 15, n):
        cci_str = f"{result.cci[i]:.2f}" if not np.isnan(result.cci[i]) else "NaN"

        if result.overbought[i]:
            zone = "Overbought"
        elif result.oversold[i]:
            zone = "Oversold"
        else:
            zone = "Neutral"

        signal = ""
        if signals["buy_oversold"][i]:
            signal = "BUY (oversold)"
        elif signals["sell_overbought"][i]:
            signal = "SELL (overbought)"
        elif signals["zero_cross_up"][i]:
            signal = "Cross UP"
        elif signals["zero_cross_down"][i]:
            signal = "Cross DOWN"

        print(f"{i:<5} {close[i]:<10.2f} {cci_str:<12} {zone:<12} {signal:<15}")

    print()
    print("信号统计:")
    print(f"  超卖反弹买入: {np.sum(signals['buy_oversold'])}")
    print(f"  超买回落卖出: {np.sum(signals['sell_overbought'])}")
    print(f"  零线上穿: {np.sum(signals['zero_cross_up'])}")
    print(f"  零线下穿: {np.sum(signals['zero_cross_down'])}")


if __name__ == "__main__":
    main()
