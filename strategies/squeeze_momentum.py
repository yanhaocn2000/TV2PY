"""
Squeeze Momentum Indicator [LazyBear] - TradingView to Python Conversion

原始指标: https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
作者: LazyBear (基于 John Carter's TTM Squeeze)

指标逻辑:
1. Squeeze 检测: Bollinger Bands 在 Keltner Channel 内部 = 低波动率 (黑点)
2. Squeeze 释放: BB 突破 KC = 准备爆发 (灰点)
3. 动量: 使用线性回归计算动量方向和强度
4. 信号: Squeeze 释放后，根据动量方向入场

Pine Script 原始代码:
```
//@version=4
study(shorttitle="SQZMOM_LB", title="Squeeze Momentum Indicator [LazyBear]", overlay=false)

length = input(20, title="BB Length")
mult = input(2.0,title="BB MultFactor")
lengthKC=input(20, title="KC Length")
multKC = input(1.5, title="KC MultFactor")

useTrueRange = input(true, title="Use TrueRange (KC)")

// Calculate BB
source = close
basis = sma(source, length)
dev = multKC * stdev(source, length)
upperBB = basis + dev
lowerBB = basis - dev

// Calculate KC
ma = sma(source, lengthKC)
range = useTrueRange ? tr : (high - low)
rangema = sma(range, lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
noSqz  = (sqzOn == false) and (sqzOff == false)

val = linreg(source - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)),sma(close,lengthKC)), lengthKC,0)

bcolor = iff( val > 0,
            iff( val > nz(val[1]), lime, green),
            iff( val < nz(val[1]), red, maroon))
scolor = noSqz ? blue : sqzOn ? black : gray
plot(val, color=bcolor, style=plot.style_histogram, linewidth=4)
plot(0, color=scolor, style=plot.style_cross, linewidth=2)
```
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import numpy as np
import pandas as pd


class SqueezeState(Enum):
    """Squeeze 状态"""
    SQUEEZE_ON = "squeeze_on"    # BB 在 KC 内部 (黑点) - 低波动率
    SQUEEZE_OFF = "squeeze_off"  # BB 突破 KC (灰点) - 准备爆发
    NO_SQUEEZE = "no_squeeze"    # 中间状态 (蓝点)


class MomentumDirection(Enum):
    """动量方向"""
    STRONG_UP = "strong_up"      # 上涨加速 (浅绿)
    WEAK_UP = "weak_up"          # 上涨减速 (深绿)
    WEAK_DOWN = "weak_down"      # 下跌减速 (深红)
    STRONG_DOWN = "strong_down"  # 下跌加速 (浅红)


@dataclass
class SqueezeMomentumParams:
    """Squeeze Momentum 指标参数"""
    # Bollinger Bands 参数
    bb_length: int = 20
    bb_mult: float = 2.0

    # Keltner Channel 参数
    kc_length: int = 20
    kc_mult: float = 1.5
    use_true_range: bool = True


@dataclass
class SqueezeMomentumResult:
    """Squeeze Momentum 计算结果"""
    # 动量值 (柱状图)
    momentum: np.ndarray

    # Squeeze 状态
    squeeze_on: np.ndarray   # True = Squeeze 中 (黑点)
    squeeze_off: np.ndarray  # True = Squeeze 释放 (灰点)
    no_squeeze: np.ndarray   # True = 无 Squeeze (蓝点)

    # 动量方向
    momentum_increasing: np.ndarray  # True = 动量增强

    # Bollinger Bands
    bb_upper: np.ndarray
    bb_basis: np.ndarray
    bb_lower: np.ndarray

    # Keltner Channel
    kc_upper: np.ndarray
    kc_basis: np.ndarray
    kc_lower: np.ndarray


class SqueezeMomentumIndicator:
    """
    Squeeze Momentum Indicator [LazyBear]

    完全对齐 TradingView 实现
    """

    def __init__(self, params: Optional[SqueezeMomentumParams] = None):
        self.params = params or SqueezeMomentumParams()

    def calculate(self, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray) -> SqueezeMomentumResult:
        """
        计算 Squeeze Momentum 指标

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组

        Returns:
            SqueezeMomentumResult 计算结果
        """
        n = len(close)
        source = close  # 默认使用收盘价

        # ============================================================
        # 1. 计算 Bollinger Bands
        # Pine: basis = sma(source, length)
        #       dev = mult * stdev(source, length)
        # ============================================================
        bb_basis = self._sma(source, self.params.bb_length)
        bb_dev = self.params.bb_mult * self._stdev(source, self.params.bb_length)
        bb_upper = bb_basis + bb_dev
        bb_lower = bb_basis - bb_dev

        # ============================================================
        # 2. 计算 Keltner Channel
        # Pine: ma = sma(source, lengthKC)
        #       range = useTrueRange ? tr : (high - low)
        #       rangema = sma(range, lengthKC)
        # 注意: LazyBear 版本使用 SMA 而不是 EMA!
        # ============================================================
        kc_basis = self._sma(source, self.params.kc_length)

        if self.params.use_true_range:
            range_val = self._true_range(high, low, close)
        else:
            range_val = high - low

        range_ma = self._sma(range_val, self.params.kc_length)
        kc_upper = kc_basis + range_ma * self.params.kc_mult
        kc_lower = kc_basis - range_ma * self.params.kc_mult

        # ============================================================
        # 3. 检测 Squeeze 状态
        # Pine: sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
        #       sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
        #       noSqz  = (sqzOn == false) and (sqzOff == false)
        # ============================================================
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
        no_squeeze = ~squeeze_on & ~squeeze_off

        # ============================================================
        # 4. 计算动量值 (核心!)
        # Pine: val = linreg(source - avg(avg(highest(high, lengthKC),
        #                                      lowest(low, lengthKC)),
        #                                 sma(close, lengthKC)),
        #                    lengthKC, 0)
        #
        # 拆解:
        #   highest_high = highest(high, lengthKC)
        #   lowest_low = lowest(low, lengthKC)
        #   donchian_mid = avg(highest_high, lowest_low)
        #   sma_close = sma(close, lengthKC)
        #   midline = avg(donchian_mid, sma_close)
        #   delta = source - midline
        #   val = linreg(delta, lengthKC, 0)
        # ============================================================

        # 计算 Donchian Channel 中线
        highest_high = self._highest(high, self.params.kc_length)
        lowest_low = self._lowest(low, self.params.kc_length)
        donchian_mid = (highest_high + lowest_low) / 2

        # 计算 SMA
        sma_close = self._sma(close, self.params.kc_length)

        # 计算中线 (Donchian 中线 和 SMA 的平均)
        midline = (donchian_mid + sma_close) / 2

        # 计算 delta (收盘价与中线的偏离)
        delta = source - midline

        # 线性回归
        momentum = self._linreg(delta, self.params.kc_length, 0)

        # ============================================================
        # 5. 计算动量方向
        # Pine: val > nz(val[1]) = 增强, val < nz(val[1]) = 减弱
        # ============================================================
        momentum_prev = np.roll(momentum, 1)
        momentum_prev[0] = 0
        momentum_increasing = momentum > momentum_prev

        return SqueezeMomentumResult(
            momentum=momentum,
            squeeze_on=squeeze_on,
            squeeze_off=squeeze_off,
            no_squeeze=no_squeeze,
            momentum_increasing=momentum_increasing,
            bb_upper=bb_upper,
            bb_basis=bb_basis,
            bb_lower=bb_lower,
            kc_upper=kc_upper,
            kc_basis=kc_basis,
            kc_lower=kc_lower
        )

    def _sma(self, data: np.ndarray, length: int) -> np.ndarray:
        """Simple Moving Average"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.mean(data[i - length + 1:i + 1])
        return result

    def _stdev(self, data: np.ndarray, length: int) -> np.ndarray:
        """Standard Deviation (population, ddof=0)"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            result[i] = np.std(window, ddof=0)
        return result

    def _true_range(self, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray) -> np.ndarray:
        """True Range"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        return np.maximum(np.maximum(tr1, tr2), tr3)

    def _highest(self, data: np.ndarray, length: int) -> np.ndarray:
        """Highest value over period (rolling max)"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.max(data[i - length + 1:i + 1])
        return result

    def _lowest(self, data: np.ndarray, length: int) -> np.ndarray:
        """Lowest value over period (rolling min)"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.min(data[i - length + 1:i + 1])
        return result

    def _linreg(self, data: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
        """
        Linear Regression Value

        Pine Script linreg(source, length, offset) 返回线性回归线在 offset 位置的值
        offset=0 表示当前 bar 的回归值

        计算方法:
        y = mx + b
        其中 x 是 bar 索引 (0 到 length-1), y 是数据值
        返回 x = length - 1 - offset 时的 y 值
        """
        result = np.full_like(data, np.nan, dtype=float)

        for i in range(length - 1, len(data)):
            # 获取窗口数据
            window = data[i - length + 1:i + 1]

            # 跳过包含 NaN 的窗口
            if np.any(np.isnan(window)):
                continue

            # x 值: 0, 1, 2, ..., length-1
            x = np.arange(length)

            # 计算线性回归 y = mx + b
            # m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
            # b = ȳ - m * x̄
            x_mean = np.mean(x)
            y_mean = np.mean(window)

            numerator = np.sum((x - x_mean) * (window - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator == 0:
                result[i] = y_mean
            else:
                m = numerator / denominator
                b = y_mean - m * x_mean

                # 返回 x = length - 1 - offset 时的值
                target_x = length - 1 - offset
                result[i] = m * target_x + b

        return result

    def get_signal(self, result: SqueezeMomentumResult) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取交易信号

        信号逻辑:
        - 买入: Squeeze 释放 (squeeze_off) + 动量为正且增强
        - 卖出: Squeeze 释放 (squeeze_off) + 动量为负且增强

        Returns:
            (buy_signals, sell_signals) 两个布尔数组
        """
        # Squeeze 从 on 变为 off (释放)
        squeeze_release = np.zeros(len(result.momentum), dtype=bool)
        for i in range(1, len(result.squeeze_on)):
            if result.squeeze_on[i-1] and result.squeeze_off[i]:
                squeeze_release[i] = True

        # 买入: Squeeze 释放 + 动量 > 0
        buy_signals = squeeze_release & (result.momentum > 0)

        # 卖出: Squeeze 释放 + 动量 < 0
        sell_signals = squeeze_release & (result.momentum < 0)

        return buy_signals, sell_signals

    def get_momentum_color(self, result: SqueezeMomentumResult) -> List[MomentumDirection]:
        """
        获取动量颜色/方向

        Pine Script:
        bcolor = iff(val > 0,
                     iff(val > nz(val[1]), lime, green),   # 正: 增强=浅绿, 减弱=深绿
                     iff(val < nz(val[1]), red, maroon))   # 负: 增强=浅红, 减弱=深红
        """
        colors = []
        for i in range(len(result.momentum)):
            val = result.momentum[i]
            increasing = result.momentum_increasing[i]

            if np.isnan(val):
                colors.append(MomentumDirection.WEAK_UP)
            elif val > 0:
                if increasing:
                    colors.append(MomentumDirection.STRONG_UP)   # lime
                else:
                    colors.append(MomentumDirection.WEAK_UP)     # green
            else:
                if not increasing:  # val < nz(val[1]) means val decreased
                    colors.append(MomentumDirection.STRONG_DOWN) # red
                else:
                    colors.append(MomentumDirection.WEAK_DOWN)   # maroon

        return colors

    def get_squeeze_color(self, result: SqueezeMomentumResult) -> List[SqueezeState]:
        """
        获取 Squeeze 状态颜色

        Pine Script:
        scolor = noSqz ? blue : sqzOn ? black : gray
        """
        states = []
        for i in range(len(result.squeeze_on)):
            if result.no_squeeze[i]:
                states.append(SqueezeState.NO_SQUEEZE)    # blue
            elif result.squeeze_on[i]:
                states.append(SqueezeState.SQUEEZE_ON)    # black
            else:
                states.append(SqueezeState.SQUEEZE_OFF)   # gray

        return states


def squeeze_momentum(df: pd.DataFrame,
                     bb_length: int = 20,
                     bb_mult: float = 2.0,
                     kc_length: int = 20,
                     kc_mult: float = 1.5,
                     use_true_range: bool = True) -> pd.DataFrame:
    """
    计算 Squeeze Momentum 指标

    Args:
        df: OHLCV DataFrame (需要 high, low, close 列)
        bb_length: Bollinger Bands 周期
        bb_mult: BB 标准差倍数
        kc_length: Keltner Channel 周期
        kc_mult: KC 倍数
        use_true_range: 是否使用 True Range

    Returns:
        添加了指标列的 DataFrame
    """
    params = SqueezeMomentumParams(
        bb_length=bb_length,
        bb_mult=bb_mult,
        kc_length=kc_length,
        kc_mult=kc_mult,
        use_true_range=use_true_range
    )

    indicator = SqueezeMomentumIndicator(params)
    result = indicator.calculate(
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    # 添加结果到 DataFrame
    df_result = df.copy()
    df_result['sqz_momentum'] = result.momentum
    df_result['sqz_on'] = result.squeeze_on
    df_result['sqz_off'] = result.squeeze_off
    df_result['sqz_no'] = result.no_squeeze
    df_result['sqz_mom_increasing'] = result.momentum_increasing

    # 添加 BB 和 KC
    df_result['bb_upper'] = result.bb_upper
    df_result['bb_basis'] = result.bb_basis
    df_result['bb_lower'] = result.bb_lower
    df_result['kc_upper'] = result.kc_upper
    df_result['kc_basis'] = result.kc_basis
    df_result['kc_lower'] = result.kc_lower

    return df_result


# ============================================================
# TradingView 验证脚本生成
# ============================================================

def generate_tv_validation_script() -> str:
    """生成 TradingView 验证脚本"""
    return '''
//@version=5
indicator("Squeeze Momentum Validation", overlay=false)

// 参数
length = input.int(20, "BB Length")
mult = input.float(2.0, "BB MultFactor")
lengthKC = input.int(20, "KC Length")
multKC = input.float(1.5, "KC MultFactor")
useTrueRange = input.bool(true, "Use TrueRange (KC)")

// Calculate BB
source = close
basis = ta.sma(source, length)
dev = mult * ta.stdev(source, length)
upperBB = basis + dev
lowerBB = basis - dev

// Calculate KC (注意: LazyBear 原版使用 SMA!)
ma = ta.sma(source, lengthKC)
range_val = useTrueRange ? ta.tr : (high - low)
rangema = ta.sma(range_val, lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

// Squeeze states
sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
noSqz  = (sqzOn == false) and (sqzOff == false)

// Momentum calculation
highest_high = ta.highest(high, lengthKC)
lowest_low = ta.lowest(low, lengthKC)
donchian_mid = (highest_high + lowest_low) / 2
sma_close = ta.sma(close, lengthKC)
midline = (donchian_mid + sma_close) / 2
delta = source - midline
val = ta.linreg(delta, lengthKC, 0)

// Colors
bcolor = val > 0 ? (val > nz(val[1]) ? color.lime : color.green) : (val < nz(val[1]) ? color.red : color.maroon)
scolor = noSqz ? color.blue : sqzOn ? color.black : color.gray

// Plot
plot(val, "Momentum", color=bcolor, style=plot.style_histogram, linewidth=4)
plot(0, "Zero", color=scolor, style=plot.style_cross, linewidth=2)

// 验证输出 (在数据窗口显示)
plot(upperBB, "BB_Upper", display=display.data_window)
plot(lowerBB, "BB_Lower", display=display.data_window)
plot(upperKC, "KC_Upper", display=display.data_window)
plot(lowerKC, "KC_Lower", display=display.data_window)
plot(midline, "Midline", display=display.data_window)
plot(delta, "Delta", display=display.data_window)
'''


if __name__ == "__main__":
    # 示例用法
    import pandas as pd

    # 生成测试数据
    np.random.seed(42)
    n = 100

    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    # 模拟价格走势 (有 squeeze 和爆发)
    base_price = 100
    prices = [base_price]

    for i in range(1, n):
        if 30 <= i <= 50:
            # Squeeze 期间 - 低波动
            change = np.random.randn() * 0.002
        elif 50 < i <= 60:
            # 爆发期间 - 高波动上涨
            change = np.random.randn() * 0.01 + 0.005
        else:
            # 正常波动
            change = np.random.randn() * 0.005

        prices.append(prices[-1] * (1 + change))

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.randn(n) * 0.003))
    low = close * (1 - np.abs(np.random.randn(n) * 0.003))

    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
    }, index=dates)

    # 计算指标
    result_df = squeeze_momentum(df)

    # 统计
    squeeze_on_count = result_df['sqz_on'].sum()
    squeeze_off_count = result_df['sqz_off'].sum()

    print("=" * 60)
    print("Squeeze Momentum Indicator [LazyBear]")
    print("=" * 60)
    print(f"数据点: {n}")
    print(f"Squeeze On (黑点): {squeeze_on_count}")
    print(f"Squeeze Off (灰点): {squeeze_off_count}")
    print()

    # 显示最后 10 行
    print("最后 10 根 bar:")
    print(result_df[['close', 'sqz_momentum', 'sqz_on', 'sqz_off']].tail(10).to_string())
    print()

    # 生成验证脚本
    print("=" * 60)
    print("TradingView 验证脚本:")
    print("=" * 60)
    print(generate_tv_validation_script())
