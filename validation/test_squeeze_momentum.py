"""
Squeeze Momentum Indicator 验证测试

验证 Python 实现与 TradingView 的对齐精度
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_momentum import (
    SqueezeMomentumIndicator,
    SqueezeMomentumParams,
    squeeze_momentum
)


# ============================================================
# TradingView 参考实现 (用于验证)
# ============================================================

class TradingViewSqueezeMomentum:
    """
    TradingView Squeeze Momentum 参考实现

    严格按照 LazyBear 原版 Pine Script 实现
    """

    @staticmethod
    def sma(data: np.ndarray, length: int) -> np.ndarray:
        """SMA - TradingView ta.sma()"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.mean(data[i - length + 1:i + 1])
        return result

    @staticmethod
    def stdev(data: np.ndarray, length: int) -> np.ndarray:
        """STDEV - TradingView ta.stdev() 使用 population stdev"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            result[i] = np.std(window, ddof=0)
        return result

    @staticmethod
    def tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range - TradingView ta.tr"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(np.maximum(tr1, tr2), tr3)

    @staticmethod
    def highest(data: np.ndarray, length: int) -> np.ndarray:
        """Highest - TradingView ta.highest()"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.max(data[i - length + 1:i + 1])
        return result

    @staticmethod
    def lowest(data: np.ndarray, length: int) -> np.ndarray:
        """Lowest - TradingView ta.lowest()"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.min(data[i - length + 1:i + 1])
        return result

    @staticmethod
    def linreg(data: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
        """Linear Regression - TradingView ta.linreg()"""
        result = np.full_like(data, np.nan, dtype=float)

        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            if np.any(np.isnan(window)):
                continue

            x = np.arange(length)
            x_mean = np.mean(x)
            y_mean = np.mean(window)

            numerator = np.sum((x - x_mean) * (window - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator == 0:
                result[i] = y_mean
            else:
                m = numerator / denominator
                b = y_mean - m * x_mean
                target_x = length - 1 - offset
                result[i] = m * target_x + b

        return result

    @staticmethod
    def calculate(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  bb_length: int = 20, bb_mult: float = 2.0,
                  kc_length: int = 20, kc_mult: float = 1.5,
                  use_true_range: bool = True) -> dict:
        """
        计算 Squeeze Momentum (严格按照 Pine Script)

        Pine Script:
        ```
        source = close
        basis = sma(source, length)
        dev = mult * stdev(source, length)
        upperBB = basis + dev
        lowerBB = basis - dev

        ma = sma(source, lengthKC)
        range = useTrueRange ? tr : (high - low)
        rangema = sma(range, lengthKC)
        upperKC = ma + rangema * multKC
        lowerKC = ma - rangema * multKC

        sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
        noSqz  = (sqzOn == false) and (sqzOff == false)

        val = linreg(source - avg(avg(highest(high, lengthKC),
                                       lowest(low, lengthKC)),
                                  sma(close, lengthKC)),
                     lengthKC, 0)
        ```
        """
        tv = TradingViewSqueezeMomentum
        source = close

        # Bollinger Bands
        basis = tv.sma(source, bb_length)
        dev = bb_mult * tv.stdev(source, bb_length)
        upperBB = basis + dev
        lowerBB = basis - dev

        # Keltner Channel
        ma = tv.sma(source, kc_length)
        if use_true_range:
            range_val = tv.tr(high, low, close)
        else:
            range_val = high - low
        rangema = tv.sma(range_val, kc_length)
        upperKC = ma + rangema * kc_mult
        lowerKC = ma - rangema * kc_mult

        # Squeeze states
        sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
        sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
        noSqz = ~sqzOn & ~sqzOff

        # Momentum
        highest_high = tv.highest(high, kc_length)
        lowest_low = tv.lowest(low, kc_length)
        donchian_mid = (highest_high + lowest_low) / 2
        sma_close = tv.sma(close, kc_length)
        midline = (donchian_mid + sma_close) / 2
        delta = source - midline
        val = tv.linreg(delta, kc_length, 0)

        return {
            'momentum': val,
            'squeeze_on': sqzOn,
            'squeeze_off': sqzOff,
            'no_squeeze': noSqz,
            'bb_upper': upperBB,
            'bb_lower': lowerBB,
            'kc_upper': upperKC,
            'kc_lower': lowerKC,
            'midline': midline,
            'delta': delta
        }


# ============================================================
# 验证测试
# ============================================================

def generate_test_data(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(seed)

    # 模拟有 squeeze 特征的价格走势
    base_price = 100
    prices = [base_price]

    for i in range(1, n):
        if 30 <= i <= 50:
            # Squeeze 期间 - 极低波动
            change = np.random.randn() * 0.001
        elif 50 < i <= 60:
            # 爆发期间 - 高波动
            change = np.random.randn() * 0.015 + 0.008
        else:
            # 正常波动
            change = np.random.randn() * 0.005

        prices.append(prices[-1] * (1 + change))

    close = np.array(prices)
    volatility = np.abs(np.random.randn(n) * 0.003) + 0.001
    high = close * (1 + volatility)
    low = close * (1 - volatility)

    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    return pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    }, index=dates)


def validate_component(name: str, our_values: np.ndarray,
                       ref_values: np.ndarray, tolerance: float = 1e-10) -> dict:
    """验证单个组件"""
    mask = ~(np.isnan(our_values) | np.isnan(ref_values))
    our_valid = our_values[mask]
    ref_valid = ref_values[mask]

    if len(our_valid) == 0:
        return {'name': name, 'count': 0, 'max_error': 0, 'passed': True}

    errors = np.abs(our_valid - ref_valid)
    max_error = float(np.max(errors))
    mean_error = float(np.mean(errors))

    # 百分比误差
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = np.where(ref_valid != 0, np.abs(errors / ref_valid) * 100, 0)
    max_pct_error = float(np.max(pct_errors))

    passed = max_error <= tolerance

    return {
        'name': name,
        'count': len(our_valid),
        'max_error': max_error,
        'mean_error': mean_error,
        'max_pct_error': max_pct_error,
        'passed': passed,
        'tolerance': tolerance
    }


def run_validation():
    """运行完整验证"""
    print("=" * 70)
    print("Squeeze Momentum Indicator 验证测试")
    print("=" * 70)
    print()

    # 生成测试数据
    df = generate_test_data(100)
    print(f"测试数据: {len(df)} 个数据点")
    print()

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # 参数
    params = SqueezeMomentumParams(
        bb_length=20,
        bb_mult=2.0,
        kc_length=20,
        kc_mult=1.5,
        use_true_range=True
    )

    print(f"参数: BB({params.bb_length}, {params.bb_mult}), KC({params.kc_length}, {params.kc_mult})")
    print()

    # 计算 - 我们的实现
    indicator = SqueezeMomentumIndicator(params)
    our_result = indicator.calculate(high, low, close)

    # 计算 - TradingView 参考实现
    tv_result = TradingViewSqueezeMomentum.calculate(
        high, low, close,
        bb_length=params.bb_length,
        bb_mult=params.bb_mult,
        kc_length=params.kc_length,
        kc_mult=params.kc_mult,
        use_true_range=params.use_true_range
    )

    # 验证各组件
    results = []

    # 1. Bollinger Bands
    results.append(validate_component("BB_Upper", our_result.bb_upper, tv_result['bb_upper']))
    results.append(validate_component("BB_Lower", our_result.bb_lower, tv_result['bb_lower']))

    # 2. Keltner Channel
    results.append(validate_component("KC_Upper", our_result.kc_upper, tv_result['kc_upper']))
    results.append(validate_component("KC_Lower", our_result.kc_lower, tv_result['kc_lower']))

    # 3. Momentum (核心!)
    results.append(validate_component("Momentum", our_result.momentum, tv_result['momentum'], tolerance=1e-8))

    # 4. Squeeze 状态
    results.append(validate_component("Squeeze_On",
                                      our_result.squeeze_on.astype(float),
                                      tv_result['squeeze_on'].astype(float)))
    results.append(validate_component("Squeeze_Off",
                                      our_result.squeeze_off.astype(float),
                                      tv_result['squeeze_off'].astype(float)))

    # 打印结果
    print("=" * 70)
    print("验证结果")
    print("=" * 70)
    print()

    all_passed = True
    for r in results:
        status = "✅" if r['passed'] else "❌"
        all_passed = all_passed and r['passed']
        print(f"{status} {r['name']}")
        print(f"   数据点: {r['count']}")
        print(f"   最大误差: {r['max_error']:.2e}")
        if 'max_pct_error' in r:
            print(f"   最大百分比误差: {r['max_pct_error']:.6f}%")
        print()

    print("=" * 70)
    print("总结")
    print("=" * 70)
    passed_count = sum(1 for r in results if r['passed'])
    print(f"通过: {passed_count}/{len(results)}")
    print()

    if all_passed:
        print("✅ 所有组件验证通过!")
        print("   Python 实现与 TradingView 完全对齐")
    else:
        print("❌ 部分组件验证失败")
        print("   需要检查实现")

    return results


def show_sample_values():
    """显示样本值对比"""
    print()
    print("=" * 70)
    print("样本值对比 (最后 10 个数据点)")
    print("=" * 70)
    print()

    df = generate_test_data(100)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # 计算
    indicator = SqueezeMomentumIndicator()
    our_result = indicator.calculate(high, low, close)
    tv_result = TradingViewSqueezeMomentum.calculate(high, low, close)

    print(f"{'Bar':>4} | {'Close':>10} | {'Our_Mom':>12} | {'TV_Mom':>12} | {'Diff':>12} | {'Sqz':>5}")
    print("-" * 70)

    for i in range(-10, 0):
        idx = len(close) + i
        our_mom = our_result.momentum[idx]
        tv_mom = tv_result['momentum'][idx]
        diff = our_mom - tv_mom if not (np.isnan(our_mom) or np.isnan(tv_mom)) else 0

        sqz = "ON" if our_result.squeeze_on[idx] else ("OFF" if our_result.squeeze_off[idx] else "NO")

        print(f"{idx:>4} | {close[idx]:>10.4f} | {our_mom:>12.6f} | {tv_mom:>12.6f} | {diff:>12.2e} | {sqz:>5}")


def generate_pine_script_for_verification():
    """生成用于在 TradingView 验证的 Pine Script"""
    print()
    print("=" * 70)
    print("TradingView 验证脚本")
    print("=" * 70)
    print("""
//@version=5
indicator("Squeeze Momentum Validation", overlay=false)

// 参数 (与 Python 保持一致)
length = 20
mult = 2.0
lengthKC = 20
multKC = 1.5
useTrueRange = true

// Bollinger Bands
source = close
basis = ta.sma(source, length)
dev = mult * ta.stdev(source, length)
upperBB = basis + dev
lowerBB = basis - dev

// Keltner Channel (注意: LazyBear 用 SMA!)
ma = ta.sma(source, lengthKC)
range_val = useTrueRange ? ta.tr : (high - low)
rangema = ta.sma(range_val, lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

// Squeeze
sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
noSqz  = (sqzOn == false) and (sqzOff == false)

// Momentum (关键计算!)
highest_high = ta.highest(high, lengthKC)
lowest_low = ta.lowest(low, lengthKC)
donchian_mid = (highest_high + lowest_low) / 2
sma_close = ta.sma(close, lengthKC)
midline = (donchian_mid + sma_close) / 2
delta = source - midline
val = ta.linreg(delta, lengthKC, 0)

// 颜色
bcolor = val > 0 ? (val > nz(val[1]) ? color.lime : color.green) : (val < nz(val[1]) ? color.red : color.maroon)
scolor = noSqz ? color.blue : sqzOn ? color.black : color.gray

// 绘制
plot(val, "Momentum", color=bcolor, style=plot.style_histogram, linewidth=4)
plot(0, "Zero Line", color=scolor, style=plot.style_cross, linewidth=2)

// 导出数据 (用于对比)
plot(upperBB, "BB_Upper", display=display.data_window)
plot(lowerBB, "BB_Lower", display=display.data_window)
plot(upperKC, "KC_Upper", display=display.data_window)
plot(lowerKC, "KC_Lower", display=display.data_window)
plot(midline, "Midline", display=display.data_window)
plot(delta, "Delta", display=display.data_window)
plot(val, "Momentum", display=display.data_window)
""")


if __name__ == "__main__":
    # 运行验证
    run_validation()

    # 显示样本值
    show_sample_values()

    # 生成 Pine Script
    generate_pine_script_for_verification()
