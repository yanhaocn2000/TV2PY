"""
Parabolic SAR (Stop and Reverse) - Python Conversion

TradingView 内置指标: ta.sar()

核心算法:
    1. 初始化:
       - SAR = 前一根K线的 low (上涨趋势) 或 high (下跌趋势)
       - AF (加速因子) = 起始值 (默认 0.02)
       - EP (极值点) = 前一根K线的 high (上涨) 或 low (下跌)

    2. 计算:
       - 上涨趋势:
         SAR = SAR + AF * (EP - SAR)
         if high > EP: EP = high, AF = min(AF + increment, max_af)
       - 下跌趋势:
         SAR = SAR + AF * (EP - SAR)
         if low < EP: EP = low, AF = min(AF + increment, max_af)

    3. 趋势反转:
       - 上涨趋势中 low < SAR → 反转为下跌
       - 下跌趋势中 high > SAR → 反转为上涨

用途:
    - 趋势跟踪
    - 止损设置
    - 入场/出场信号
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ParabolicSARResult:
    """Parabolic SAR 计算结果"""
    sar: np.ndarray            # SAR 值
    trend: np.ndarray          # 趋势方向 (1=上涨, -1=下跌)
    af: np.ndarray             # 加速因子
    ep: np.ndarray             # 极值点
    reversal: np.ndarray       # 反转信号


class ParabolicSARIndicator:
    """
    Parabolic SAR (Stop and Reverse)

    由 J. Welles Wilder 发明，用于确定趋势方向和潜在反转点。

    Parameters:
        start: 起始加速因子 - 默认 0.02
        increment: 加速因子增量 - 默认 0.02
        maximum: 最大加速因子 - 默认 0.2
    """

    def __init__(
        self,
        start: float = 0.02,
        increment: float = 0.02,
        maximum: float = 0.2,
    ):
        self.start = start
        self.increment = increment
        self.maximum = maximum

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> ParabolicSARResult:
        """
        计算 Parabolic SAR

        Returns:
            ParabolicSARResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        sar = np.zeros(n)
        trend = np.zeros(n, dtype=int)
        af = np.zeros(n)
        ep = np.zeros(n)
        reversal = np.zeros(n, dtype=bool)

        # 初始化 - 假设从上涨趋势开始
        trend[0] = 1
        af[0] = self.start
        ep[0] = high[0]
        sar[0] = low[0]

        if n < 2:
            return ParabolicSARResult(
                sar=sar, trend=trend, af=af, ep=ep, reversal=reversal
            )

        # 初始化第二根K线
        if close[1] > close[0]:
            trend[1] = 1
            ep[1] = high[1]
            sar[1] = low[0]
        else:
            trend[1] = -1
            ep[1] = low[1]
            sar[1] = high[0]
        af[1] = self.start

        # 主计算循环
        for i in range(2, n):
            prev_sar = sar[i - 1]
            prev_af = af[i - 1]
            prev_ep = ep[i - 1]
            prev_trend = trend[i - 1]

            # 计算新的 SAR
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

            if prev_trend == 1:  # 上涨趋势
                # SAR 不能高于前两根K线的最低价
                new_sar = min(new_sar, low[i - 1], low[i - 2])

                # 检查是否反转
                if low[i] < new_sar:
                    # 反转为下跌趋势
                    trend[i] = -1
                    sar[i] = prev_ep  # 使用前一个 EP 作为新 SAR
                    ep[i] = low[i]
                    af[i] = self.start
                    reversal[i] = True
                else:
                    # 继续上涨趋势
                    trend[i] = 1
                    sar[i] = new_sar

                    # 更新 EP 和 AF
                    if high[i] > prev_ep:
                        ep[i] = high[i]
                        af[i] = min(prev_af + self.increment, self.maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

            else:  # 下跌趋势
                # SAR 不能低于前两根K线的最高价
                new_sar = max(new_sar, high[i - 1], high[i - 2])

                # 检查是否反转
                if high[i] > new_sar:
                    # 反转为上涨趋势
                    trend[i] = 1
                    sar[i] = prev_ep  # 使用前一个 EP 作为新 SAR
                    ep[i] = high[i]
                    af[i] = self.start
                    reversal[i] = True
                else:
                    # 继续下跌趋势
                    trend[i] = -1
                    sar[i] = new_sar

                    # 更新 EP 和 AF
                    if low[i] < prev_ep:
                        ep[i] = low[i]
                        af[i] = min(prev_af + self.increment, self.maximum)
                    else:
                        ep[i] = prev_ep
                        af[i] = prev_af

        return ParabolicSARResult(
            sar=sar,
            trend=trend,
            af=af,
            ep=ep,
            reversal=reversal,
        )

    def get_signals(self, result: ParabolicSARResult) -> dict:
        """获取交易信号"""
        n = len(result.sar)

        buy_signal = np.zeros(n, dtype=bool)
        sell_signal = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if result.trend[i] == 1 and result.trend[i - 1] == -1:
                buy_signal[i] = True
            elif result.trend[i] == -1 and result.trend[i - 1] == 1:
                sell_signal[i] = True

        return {
            "buy": buy_signal,
            "sell": sell_signal,
            "reversal": result.reversal,
        }


class ParabolicSARPyneCore:
    """PyneCore 兼容的 Parabolic SAR 实现"""

    def __init__(
        self,
        start: float = 0.02,
        increment: float = 0.02,
        maximum: float = 0.2,
    ):
        self.indicator = ParabolicSARIndicator(
            start=start, increment=increment, maximum=maximum
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
            "sar": result.sar,
            "trend": result.trend,
            "reversal": result.reversal,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_PARABOLIC_SAR = '''
//@version=5
indicator("Parabolic SAR", overlay=true)

// 参数
start = input.float(0.02, "Start")
increment = input.float(0.02, "Increment")
maximum = input.float(0.2, "Maximum")

// 计算
sar = ta.sar(start, increment, maximum)

// 绘图
plot(sar, "SAR", style=plot.style_cross, color=close > sar ? color.green : color.red)
'''


def main():
    """测试 Parabolic SAR 指标"""
    print("=" * 60)
    print("Parabolic SAR - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 100

    # 模拟带有趋势的价格数据
    base_price = 100.0
    trend = np.linspace(0, 15, n)
    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5

    # 计算 Parabolic SAR
    indicator = ParabolicSARIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: start={indicator.start}, increment={indicator.increment}, max={indicator.maximum}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'SAR':<10} {'Trend':<8} {'AF':<8} {'Signal':<10}")
    print("-" * 80)

    signals = indicator.get_signals(result)

    for i in range(n - 15, n):
        trend_str = "UP" if result.trend[i] == 1 else "DOWN"
        signal = ""
        if signals["buy"][i]:
            signal = "BUY"
        elif signals["sell"][i]:
            signal = "SELL"

        print(f"{i:<5} {close[i]:<10.2f} {result.sar[i]:<10.2f} {trend_str:<8} {result.af[i]:<8.3f} {signal:<10}")

    print()
    print("信号统计:")
    print(f"  买入信号 (反转为上涨): {np.sum(signals['buy'])}")
    print(f"  卖出信号 (反转为下跌): {np.sum(signals['sell'])}")


if __name__ == "__main__":
    main()
