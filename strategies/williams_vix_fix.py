"""
Williams Vix Fix [ChrisMoody] - Python Conversion

核心算法 (Larry Williams 的 VIX 修复):
    wvf = ((highest(close, pd) - low) / highest(close, pd)) * 100

    这是一个"合成VIX"指标，可以应用于任何品种。
    高值表示潜在的市场底部（类似于真实 VIX 在恐慌时飙升）

参考:
    - https://www.tradingview.com/script/og7JPrRA-CM-Williams-Vix-Fix/
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class WilliamsVixFixResult:
    """Williams Vix Fix 计算结果"""
    wvf: np.ndarray              # Williams Vix Fix 值
    highest_close: np.ndarray     # 最高收盘价
    bb_upper: np.ndarray          # 布林带上轨
    bb_lower: np.ndarray          # 布林带下轨
    bb_middle: np.ndarray         # 布林带中轨
    range_high: np.ndarray        # 区间最高
    is_alert: np.ndarray          # 警报信号 (WVF > 上轨)
    is_extreme: np.ndarray        # 极端信号


class WilliamsVixFix:
    """
    Williams Vix Fix [ChrisMoody]

    这是 Larry Williams 创造的"合成VIX"指标，可以应用于任何品种的图表。
    它测量当前低点距离回望期内最高收盘价的距离。

    高 WVF 值表示:
        - 市场恐慌
        - 潜在的底部
        - 类似于真实 VIX 在市场崩盘时的行为

    Parameters:
        pd: 回望周期 (标准差最高点) - 默认 22
        bb_period: 布林带周期 - 默认 20
        bb_mult: 布林带标准差乘数 - 默认 2.0
        lb: 最高/最低点回望周期 - 默认 50
        ph: 百分比最高阈值 - 默认 0.85
        pl: 百分比最低阈值 - 默认 1.01
    """

    def __init__(
        self,
        pd: int = 22,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        lb: int = 50,
        ph: float = 0.85,
        pl: float = 1.01,
    ):
        self.pd = pd
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.lb = lb
        self.ph = ph
        self.pl = pl

    def _highest(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算滚动最高值"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.max(data[i - period + 1:i + 1])
        return result

    def _lowest(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算滚动最低值"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.min(data[i - period + 1:i + 1])
        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算 SMA"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid_data = data[i - period + 1:i + 1]
            valid_data = valid_data[~np.isnan(valid_data)]
            if len(valid_data) >= period:
                result[i] = np.mean(valid_data)
        return result

    def _stdev(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算标准差 (population, ddof=0)"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            valid_data = data[i - period + 1:i + 1]
            valid_data = valid_data[~np.isnan(valid_data)]
            if len(valid_data) >= period:
                result[i] = np.std(valid_data, ddof=0)
        return result

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> WilliamsVixFixResult:
        """
        计算 Williams Vix Fix

        算法:
        1. 计算回望期内的最高收盘价
        2. WVF = ((highest_close - low) / highest_close) * 100
        3. 计算 WVF 的布林带
        4. 确定警报和极端信号

        Returns:
            WilliamsVixFixResult 包含所有计算结果
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Step 1: 计算最高收盘价
        highest_close = self._highest(close, self.pd)

        # Step 2: 计算 WVF
        # WVF = ((highest_close - low) / highest_close) * 100
        wvf = np.where(
            highest_close != 0,
            ((highest_close - low) / highest_close) * 100,
            0
        )

        # Step 3: 计算布林带
        bb_middle = self._sma(wvf, self.bb_period)
        bb_stdev = self._stdev(wvf, self.bb_period)
        bb_upper = bb_middle + (self.bb_mult * bb_stdev)
        bb_lower = bb_middle - (self.bb_mult * bb_stdev)

        # Step 4: 计算区间最高/最低
        range_high = self._highest(wvf, self.lb) * self.ph

        # Step 5: 确定信号
        # 警报: WVF 穿越布林带上轨
        is_alert = wvf >= bb_upper

        # 极端: WVF 超过区间最高
        is_extreme = wvf >= range_high

        return WilliamsVixFixResult(
            wvf=wvf,
            highest_close=highest_close,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_middle=bb_middle,
            range_high=range_high,
            is_alert=is_alert,
            is_extreme=is_extreme,
        )

    def get_signals(self, result: WilliamsVixFixResult) -> dict:
        """
        获取交易信号

        潜在买入信号:
            - WVF 突破布林带上轨 (恐慌)
            - WVF 回落 (恐慌消退)
        """
        n = len(result.wvf)

        # 检测 WVF 从高位回落 (潜在买入)
        buy_signal = np.zeros(n, dtype=bool)
        for i in range(1, n):
            # 从警报状态回落
            if result.is_alert[i - 1] and not result.is_alert[i]:
                buy_signal[i] = True

        return {
            "buy": buy_signal,
            "alert": result.is_alert,
            "extreme": result.is_extreme,
        }


class WilliamsVixFixPyneCore:
    """
    PyneCore 兼容的 Williams Vix Fix 实现
    """

    def __init__(
        self,
        pd: int = 22,
        bb_period: int = 20,
        bb_mult: float = 2.0,
    ):
        self.indicator = WilliamsVixFix(
            pd=pd,
            bb_period=bb_period,
            bb_mult=bb_mult,
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
            "wvf": result.wvf,
            "bb_upper": result.bb_upper,
            "bb_middle": result.bb_middle,
            "bb_lower": result.bb_lower,
            "is_alert": result.is_alert,
            "is_extreme": result.is_extreme,
        }, index=close.index)


# ============================================================
# Pine Script 对照版本 (用于验证)
# ============================================================
PINE_SCRIPT_WILLIAMS_VIX_FIX = '''
//@version=5
indicator("Williams Vix Fix", overlay=false)

// 参数
pd = input.int(22, "LookBack Period Standard Deviation High")
bession = input.int(20, "Bollinger Band SMA Period")
mult = input.float(2.0, "Bollinger Band Standard Deviation")
lb = input.int(50, "Look Back Period Percentile High")
ph = input.float(0.85, "Highest Percentile")
pl = input.float(1.01, "Lowest Percentile")

// Williams Vix Fix 计算
wvf = ((ta.highest(close, pd) - low) / ta.highest(close, pd)) * 100

// 布林带
sDev = mult * ta.stdev(wvf, bession)
midLine = ta.sma(wvf, bession)
lowerBand = midLine - sDev
upperBand = midLine + sDev

// 区间计算
rangeHigh = ta.highest(wvf, lb) * ph

// 颜色条件
col = wvf >= upperBand or wvf >= rangeHigh ? color.lime :
      wvf < upperBand and wvf > lowerBand ? color.gray : color.red

// 绘图
plot(wvf, title="Williams Vix Fix", style=plot.style_histogram, linewidth=4, color=col)
plot(upperBand, "Upper Band", color=color.aqua)
plot(rangeHigh, "Range High", color=color.orange)
'''


def main():
    """测试 Williams Vix Fix 指标"""
    print("=" * 60)
    print("Williams Vix Fix [ChrisMoody] - Python Implementation")
    print("=" * 60)

    # 生成测试数据 (模拟市场崩盘场景)
    np.random.seed(42)
    n = 100

    # 模拟价格数据 (包含一个下跌阶段)
    base_price = 100.0
    close = [base_price]
    for i in range(1, n):
        # 在中间创造一个下跌
        if 40 <= i <= 60:
            change = -np.random.uniform(0.005, 0.02)
        else:
            change = np.random.randn() * 0.01

        close.append(close[-1] * (1 + change))

    close = np.array(close)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.015)

    # 计算 WVF
    indicator = WilliamsVixFix()
    result = indicator.calculate(high, low, close)

    # 显示结果
    print(f"\n参数: pd={indicator.pd}, bb_period={indicator.bb_period}")
    print(f"数据点数: {n}")
    print()

    print("最后 20 个数据点:")
    print("-" * 80)
    print(f"{'Bar':<5} {'Close':<10} {'WVF':<10} {'BB Upper':<10} {'Alert':<8} {'Extreme':<8}")
    print("-" * 80)

    for i in range(n - 20, n):
        wvf_val = result.wvf[i]
        bb_upper = result.bb_upper[i]

        wvf_str = f"{wvf_val:.2f}" if not np.isnan(wvf_val) else "NaN"
        bb_str = f"{bb_upper:.2f}" if not np.isnan(bb_upper) else "NaN"
        alert_str = "YES" if result.is_alert[i] else ""
        extreme_str = "YES" if result.is_extreme[i] else ""

        print(f"{i:<5} {close[i]:<10.2f} {wvf_str:<10} {bb_str:<10} {alert_str:<8} {extreme_str:<8}")

    print()
    print("信号统计:")
    signals = indicator.get_signals(result)
    print(f"  警报信号 (WVF >= BB Upper): {np.sum(result.is_alert)}")
    print(f"  极端信号 (WVF >= Range High): {np.sum(result.is_extreme)}")
    print(f"  买入信号 (从警报回落): {np.sum(signals['buy'])}")

    # 找出高 WVF 的位置
    high_wvf_mask = result.wvf > np.nanpercentile(result.wvf, 90)
    high_wvf_indices = np.where(high_wvf_mask)[0]
    if len(high_wvf_indices) > 0:
        print(f"\n高 WVF 区间 (>90th percentile): bars {high_wvf_indices[0]}-{high_wvf_indices[-1]}")


if __name__ == "__main__":
    main()
