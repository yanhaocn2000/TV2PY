"""
Ichimoku Cloud (一目均衡表) - Python Conversion

TradingView 内置指标: Ichimoku Cloud

核心组件:
    Tenkan-sen (转换线) = (highest_high + lowest_low) / 2 over 9 periods
    Kijun-sen (基准线) = (highest_high + lowest_low) / 2 over 26 periods
    Senkou Span A (先行线A) = (Tenkan + Kijun) / 2, 向前位移 26 周期
    Senkou Span B (先行线B) = (highest_high + lowest_low) / 2 over 52, 向前位移 26 周期
    Chikou Span (迟行线) = close, 向后位移 26 周期

云带 (Kumo):
    - Span A 和 Span B 之间的区域
    - 绿色云: Span A > Span B (看涨)
    - 红色云: Span A < Span B (看跌)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class IchimokuResult:
    """Ichimoku 计算结果"""
    tenkan: np.ndarray         # 转换线
    kijun: np.ndarray          # 基准线
    senkou_a: np.ndarray       # 先行线 A
    senkou_b: np.ndarray       # 先行线 B
    chikou: np.ndarray         # 迟行线
    cloud_top: np.ndarray      # 云带顶部
    cloud_bottom: np.ndarray   # 云带底部
    bullish_cloud: np.ndarray  # 看涨云带
    above_cloud: np.ndarray    # 价格在云带上方
    below_cloud: np.ndarray    # 价格在云带下方


class IchimokuIndicator:
    """
    Ichimoku Cloud (一目均衡表)

    由日本记者细田悟一 (Goichi Hosoda) 发明，是一个综合趋势跟踪系统。

    Parameters:
        tenkan_period: 转换线周期 - 默认 9
        kijun_period: 基准线周期 - 默认 26
        senkou_b_period: 先行线 B 周期 - 默认 52
        displacement: 位移周期 - 默认 26
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    def _donchian_midline(
        self,
        high: np.ndarray,
        low: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """
        计算 Donchian 中线

        midline = (highest_high + lowest_low) / 2
        """
        n = len(high)
        result = np.full(n, np.nan)

        for i in range(period - 1, n):
            highest = np.max(high[i - period + 1:i + 1])
            lowest = np.min(low[i - period + 1:i + 1])
            result[i] = (highest + lowest) / 2

        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> IchimokuResult:
        """
        计算 Ichimoku Cloud

        Returns:
            IchimokuResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # 计算转换线 (Tenkan-sen)
        tenkan = self._donchian_midline(high, low, self.tenkan_period)

        # 计算基准线 (Kijun-sen)
        kijun = self._donchian_midline(high, low, self.kijun_period)

        # 计算先行线 A (Senkou Span A) - 需要向前位移
        senkou_a_raw = (tenkan + kijun) / 2
        senkou_a = np.full(n + self.displacement, np.nan)
        senkou_a[self.displacement:] = senkou_a_raw

        # 计算先行线 B (Senkou Span B) - 需要向前位移
        senkou_b_raw = self._donchian_midline(high, low, self.senkou_b_period)
        senkou_b = np.full(n + self.displacement, np.nan)
        senkou_b[self.displacement:] = senkou_b_raw

        # 截取到当前长度 (实际应用中云带会延伸到未来)
        senkou_a = senkou_a[:n]
        senkou_b = senkou_b[:n]

        # 计算迟行线 (Chikou Span) - 向后位移
        chikou = np.full(n, np.nan)
        if n > self.displacement:
            chikou[:n - self.displacement] = close[self.displacement:]

        # 计算云带顶部和底部
        cloud_top = np.maximum(senkou_a, senkou_b)
        cloud_bottom = np.minimum(senkou_a, senkou_b)

        # 云带方向
        bullish_cloud = senkou_a > senkou_b

        # 价格相对于云带的位置
        above_cloud = close > cloud_top
        below_cloud = close < cloud_bottom

        return IchimokuResult(
            tenkan=tenkan,
            kijun=kijun,
            senkou_a=senkou_a,
            senkou_b=senkou_b,
            chikou=chikou,
            cloud_top=cloud_top,
            cloud_bottom=cloud_bottom,
            bullish_cloud=bullish_cloud,
            above_cloud=above_cloud,
            below_cloud=below_cloud,
        )

    def get_signals(
        self,
        close: np.ndarray,
        result: IchimokuResult,
    ) -> dict:
        """
        获取交易信号

        经典 Ichimoku 信号:
        1. TK 交叉: Tenkan 穿越 Kijun
        2. 云带突破: 价格穿越云带
        3. Chikou 确认: 迟行线相对于价格的位置
        """
        n = len(close)

        # TK 交叉
        tk_cross_up = np.zeros(n, dtype=bool)
        tk_cross_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(result.tenkan[i]) and not np.isnan(result.kijun[i]):
                if not np.isnan(result.tenkan[i-1]) and not np.isnan(result.kijun[i-1]):
                    tk_cross_up[i] = (result.tenkan[i] > result.kijun[i] and
                                      result.tenkan[i-1] <= result.kijun[i-1])
                    tk_cross_down[i] = (result.tenkan[i] < result.kijun[i] and
                                        result.tenkan[i-1] >= result.kijun[i-1])

        # 云带突破
        cloud_breakout_up = np.zeros(n, dtype=bool)
        cloud_breakout_down = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not np.isnan(result.cloud_top[i]) and not np.isnan(result.cloud_bottom[i]):
                # 从云带下方突破到上方
                if result.above_cloud[i] and not result.above_cloud[i-1]:
                    cloud_breakout_up[i] = True
                # 从云带上方跌破到下方
                if result.below_cloud[i] and not result.below_cloud[i-1]:
                    cloud_breakout_down[i] = True

        # 综合买入信号: TK 金叉 + 价格在云带上方 + 看涨云
        strong_buy = tk_cross_up & result.above_cloud & result.bullish_cloud

        # 综合卖出信号: TK 死叉 + 价格在云带下方 + 看跌云
        strong_sell = tk_cross_down & result.below_cloud & ~result.bullish_cloud

        return {
            "tk_cross_up": tk_cross_up,
            "tk_cross_down": tk_cross_down,
            "cloud_breakout_up": cloud_breakout_up,
            "cloud_breakout_down": cloud_breakout_down,
            "strong_buy": strong_buy,
            "strong_sell": strong_sell,
        }


class IchimokuPyneCore:
    """PyneCore 兼容的 Ichimoku 实现"""

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ):
        self.indicator = IchimokuIndicator(
            tenkan_period=tenkan_period,
            kijun_period=kijun_period,
            senkou_b_period=senkou_b_period,
            displacement=displacement,
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
            "tenkan": result.tenkan,
            "kijun": result.kijun,
            "senkou_a": result.senkou_a,
            "senkou_b": result.senkou_b,
            "chikou": result.chikou,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_ICHIMOKU = '''
//@version=5
indicator("Ichimoku Cloud", overlay=true)

// 参数
conversionPeriods = input.int(9, "Tenkan Period")
basePeriods = input.int(26, "Kijun Period")
laggingSpan2Periods = input.int(52, "Senkou B Period")
displacement = input.int(26, "Displacement")

// Donchian 中线函数
donchian(len) => math.avg(ta.lowest(len), ta.highest(len))

// 计算
tenkanSen = donchian(conversionPeriods)
kijunSen = donchian(basePeriods)
senkouSpanA = math.avg(tenkanSen, kijunSen)
senkouSpanB = donchian(laggingSpan2Periods)

// 绘图
plot(tenkanSen, "Tenkan", color=color.blue)
plot(kijunSen, "Kijun", color=color.red)
plot(close, offset=-displacement, color=color.green, title="Chikou")

p1 = plot(senkouSpanA, offset=displacement, color=color.green, title="Senkou A")
p2 = plot(senkouSpanB, offset=displacement, color=color.red, title="Senkou B")
fill(p1, p2, color=senkouSpanA > senkouSpanB ? color.new(color.green, 90) : color.new(color.red, 90))
'''


def main():
    """测试 Ichimoku 指标"""
    print("=" * 60)
    print("Ichimoku Cloud - Python Implementation")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    n = 150

    # 模拟价格数据 (带有趋势)
    base_price = 100.0
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.randn(n) * 0.5)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 1.0
    low = close - np.abs(np.random.randn(n)) * 1.0

    # 计算 Ichimoku
    indicator = IchimokuIndicator()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: Tenkan={indicator.tenkan_period}, Kijun={indicator.kijun_period}")
    print(f"      Senkou B={indicator.senkou_b_period}, Displacement={indicator.displacement}")
    print(f"数据点数: {n}")
    print()

    print("最后 15 个数据点:")
    print("-" * 100)
    print(f"{'Bar':<5} {'Close':<10} {'Tenkan':<10} {'Kijun':<10} {'SpanA':<10} {'SpanB':<10} {'Position':<15}")
    print("-" * 100)

    for i in range(n - 15, n):
        tenkan_str = f"{result.tenkan[i]:.2f}" if not np.isnan(result.tenkan[i]) else "NaN"
        kijun_str = f"{result.kijun[i]:.2f}" if not np.isnan(result.kijun[i]) else "NaN"
        span_a_str = f"{result.senkou_a[i]:.2f}" if not np.isnan(result.senkou_a[i]) else "NaN"
        span_b_str = f"{result.senkou_b[i]:.2f}" if not np.isnan(result.senkou_b[i]) else "NaN"

        if result.above_cloud[i]:
            pos = "Above Cloud"
        elif result.below_cloud[i]:
            pos = "Below Cloud"
        else:
            pos = "In Cloud"

        print(f"{i:<5} {close[i]:<10.2f} {tenkan_str:<10} {kijun_str:<10} {span_a_str:<10} {span_b_str:<10} {pos:<15}")

    print()
    signals = indicator.get_signals(close, result)
    print("信号统计:")
    print(f"  TK 金叉: {np.sum(signals['tk_cross_up'])}")
    print(f"  TK 死叉: {np.sum(signals['tk_cross_down'])}")
    print(f"  云带向上突破: {np.sum(signals['cloud_breakout_up'])}")
    print(f"  云带向下突破: {np.sum(signals['cloud_breakout_down'])}")
    print(f"  强买信号: {np.sum(signals['strong_buy'])}")
    print(f"  强卖信号: {np.sum(signals['strong_sell'])}")


if __name__ == "__main__":
    main()
