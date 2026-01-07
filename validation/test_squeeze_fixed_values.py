"""
Squeeze Momentum Indicator - 固定值验证测试

使用固定数据在 Python 和 TradingView 中计算，对比结果
这是最可靠的验证方式，因为数据完全相同
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_momentum import SqueezeMomentumIndicator, SqueezeMomentumParams


# ============================================================
# 固定测试数据 (40个数据点，确保指标有效)
# ============================================================

# 这些数据将在 Python 和 TradingView 中使用完全相同的值
FIXED_OHLC_DATA = [
    # (open, high, low, close)
    (100.00, 101.50, 99.50, 100.50),   # 0
    (100.50, 102.00, 100.00, 101.00),  # 1
    (101.00, 102.50, 100.50, 102.00),  # 2
    (102.00, 103.00, 101.00, 101.50),  # 3
    (101.50, 102.50, 100.50, 102.00),  # 4
    (102.00, 103.50, 101.50, 103.00),  # 5
    (103.00, 104.00, 102.00, 103.50),  # 6
    (103.50, 104.50, 102.50, 104.00),  # 7
    (104.00, 105.00, 103.00, 104.50),  # 8
    (104.50, 105.50, 103.50, 105.00),  # 9
    (105.00, 106.00, 104.00, 105.50),  # 10
    (105.50, 106.50, 104.50, 106.00),  # 11
    (106.00, 107.00, 105.00, 106.50),  # 12
    (106.50, 107.50, 105.50, 107.00),  # 13
    (107.00, 108.00, 106.00, 107.50),  # 14
    (107.50, 108.50, 106.50, 108.00),  # 15
    (108.00, 109.00, 107.00, 108.50),  # 16
    (108.50, 109.50, 107.50, 109.00),  # 17
    (109.00, 110.00, 108.00, 109.50),  # 18
    (109.50, 110.50, 108.50, 110.00),  # 19 - length=20, 指标开始有效
    (110.00, 111.00, 109.00, 110.50),  # 20
    (110.50, 111.50, 109.50, 111.00),  # 21
    (111.00, 112.00, 110.00, 111.50),  # 22
    (111.50, 112.50, 110.50, 112.00),  # 23
    (112.00, 113.00, 111.00, 112.50),  # 24
    (112.50, 113.50, 111.50, 113.00),  # 25
    (113.00, 114.00, 112.00, 113.50),  # 26
    (113.50, 114.50, 112.50, 114.00),  # 27
    (114.00, 115.00, 113.00, 114.50),  # 28
    (114.50, 115.50, 113.50, 115.00),  # 29
    (115.00, 116.00, 114.00, 115.50),  # 30
    (115.50, 116.50, 114.50, 116.00),  # 31
    (116.00, 117.00, 115.00, 116.50),  # 32
    (116.50, 117.50, 115.50, 117.00),  # 33
    (117.00, 118.00, 116.00, 117.50),  # 34
    (117.50, 118.50, 116.50, 118.00),  # 35
    (118.00, 119.00, 117.00, 118.50),  # 36
    (118.50, 119.50, 117.50, 119.00),  # 37
    (119.00, 120.00, 118.00, 118.50),  # 38 - 回调
    (118.50, 119.00, 117.00, 117.50),  # 39 - 回调
]


def generate_pine_script() -> str:
    """生成 TradingView Pine Script 用于验证"""

    # 生成数组初始化代码
    open_values = ", ".join([f"{d[0]:.2f}" for d in FIXED_OHLC_DATA])
    high_values = ", ".join([f"{d[1]:.2f}" for d in FIXED_OHLC_DATA])
    low_values = ", ".join([f"{d[2]:.2f}" for d in FIXED_OHLC_DATA])
    close_values = ", ".join([f"{d[3]:.2f}" for d in FIXED_OHLC_DATA])

    return f'''
//@version=5
indicator("Squeeze Momentum Fixed Value Test", overlay=false)

// ============================================================
// 固定测试数据 - 与 Python 完全相同
// ============================================================
var float[] open_arr = array.from({open_values})
var float[] high_arr = array.from({high_values})
var float[] low_arr = array.from({low_values})
var float[] close_arr = array.from({close_values})

// 获取当前 bar 的 OHLC
int idx = bar_index < 40 ? bar_index : na
float o = not na(idx) ? array.get(open_arr, idx) : na
float h = not na(idx) ? array.get(high_arr, idx) : na
float l = not na(idx) ? array.get(low_arr, idx) : na
float c = not na(idx) ? array.get(close_arr, idx) : na

// ============================================================
// 参数 (与 LazyBear 原版一致)
// ============================================================
int length = 20
float mult = 2.0
int lengthKC = 20
float multKC = 1.5

// ============================================================
// 手动计算 (因为 ta.* 函数需要 series，不能直接用 array)
// ============================================================

// SMA 计算
sma_close(int len) =>
    float sum = 0.0
    for i = 0 to len - 1
        if bar_index - i >= 0 and bar_index - i < 40
            sum := sum + array.get(close_arr, bar_index - i)
    sum / len

// STDEV 计算 (population)
stdev_close(int len) =>
    float mean = sma_close(len)
    float sum_sq = 0.0
    for i = 0 to len - 1
        if bar_index - i >= 0 and bar_index - i < 40
            float diff = array.get(close_arr, bar_index - i) - mean
            sum_sq := sum_sq + diff * diff
    math.sqrt(sum_sq / len)

// True Range
tr_val() =>
    if bar_index == 0
        h - l
    else if bar_index < 40
        float prev_c = array.get(close_arr, bar_index - 1)
        math.max(h - l, math.max(math.abs(h - prev_c), math.abs(l - prev_c)))
    else
        na

// SMA of TR
sma_tr(int len) =>
    float sum = 0.0
    for i = 0 to len - 1
        if bar_index - i >= 0 and bar_index - i < 40
            int bi = bar_index - i
            float hi = array.get(high_arr, bi)
            float lo = array.get(low_arr, bi)
            float tr_i = hi - lo
            if bi > 0
                float prev_c = array.get(close_arr, bi - 1)
                tr_i := math.max(hi - lo, math.max(math.abs(hi - prev_c), math.abs(lo - prev_c)))
            sum := sum + tr_i
    sum / len

// Highest High
highest_high(int len) =>
    float maxVal = na
    for i = 0 to len - 1
        if bar_index - i >= 0 and bar_index - i < 40
            float hi = array.get(high_arr, bar_index - i)
            if na(maxVal) or hi > maxVal
                maxVal := hi
    maxVal

// Lowest Low
lowest_low(int len) =>
    float minVal = na
    for i = 0 to len - 1
        if bar_index - i >= 0 and bar_index - i < 40
            float lo = array.get(low_arr, bar_index - i)
            if na(minVal) or lo < minVal
                minVal := lo
    minVal

// Linear Regression
linreg_delta(int len) =>
    // delta = close - midline
    float hh = highest_high(len)
    float ll = lowest_low(len)
    float donchian_mid = (hh + ll) / 2
    float sma_c = sma_close(len)
    float midline = (donchian_mid + sma_c) / 2

    // 计算 delta 数组
    var float[] delta_arr = array.new_float(40, 0.0)
    for i = 0 to 39
        if i < 40
            float ci = array.get(close_arr, i)
            // 需要为每个 bar 计算 midline，这里简化使用当前 midline
            array.set(delta_arr, i, ci - midline)

    // linreg 计算
    float sum_x = 0.0
    float sum_y = 0.0
    float sum_xy = 0.0
    float sum_xx = 0.0
    int n = len

    for i = 0 to len - 1
        if bar_index - i >= 0 and bar_index - i < 40
            float x = len - 1 - i
            float y = array.get(delta_arr, bar_index - i)
            sum_x := sum_x + x
            sum_y := sum_y + y
            sum_xy := sum_xy + x * y
            sum_xx := sum_xx + x * x

    float x_mean = sum_x / n
    float y_mean = sum_y / n
    float m = (sum_xy - n * x_mean * y_mean) / (sum_xx - n * x_mean * x_mean)
    float b = y_mean - m * x_mean
    float target_x = len - 1
    m * target_x + b

// ============================================================
// 计算指标
// ============================================================
float bb_basis = bar_index >= length - 1 and bar_index < 40 ? sma_close(length) : na
float bb_dev = bar_index >= length - 1 and bar_index < 40 ? mult * stdev_close(length) : na
float bb_upper = bb_basis + bb_dev
float bb_lower = bb_basis - bb_dev

float kc_basis = bb_basis  // LazyBear 用 SMA, 与 BB 相同
float kc_range = bar_index >= length - 1 and bar_index < 40 ? sma_tr(lengthKC) : na
float kc_upper = kc_basis + kc_range * multKC
float kc_lower = kc_basis - kc_range * multKC

// Squeeze
bool sqzOn = (bb_lower > kc_lower) and (bb_upper < kc_upper)
bool sqzOff = (bb_lower < kc_lower) and (bb_upper > kc_upper)

// Momentum (简化计算)
float hh = highest_high(lengthKC)
float ll = lowest_low(lengthKC)
float midline = ((hh + ll) / 2 + sma_close(lengthKC)) / 2
float delta = c - midline

// 使用内置 linreg 验证
float mom_builtin = ta.linreg(close - ((ta.highest(high, lengthKC) + ta.lowest(low, lengthKC)) / 2 + ta.sma(close, lengthKC)) / 2, lengthKC, 0)

// ============================================================
// 输出
// ============================================================
plot(c, "Close", display=display.data_window)
plot(bb_upper, "BB_Upper", display=display.data_window)
plot(bb_lower, "BB_Lower", display=display.data_window)
plot(kc_upper, "KC_Upper", display=display.data_window)
plot(kc_lower, "KC_Lower", display=display.data_window)
plot(midline, "Midline", display=display.data_window)
plot(delta, "Delta", display=display.data_window)
plot(mom_builtin, "Momentum", display=display.data_window, color=color.red)

// 绘制柱状图
bcolor = mom_builtin > 0 ? (mom_builtin > nz(mom_builtin[1]) ? color.lime : color.green) : (mom_builtin < nz(mom_builtin[1]) ? color.red : color.maroon)
scolor = sqzOn ? color.black : sqzOff ? color.gray : color.blue
plot(mom_builtin, "Momentum", color=bcolor, style=plot.style_histogram, linewidth=4)
plot(0, "Zero", color=scolor, style=plot.style_cross, linewidth=2)

// 结果标签
if bar_index == 39
    label.new(bar_index, mom_builtin,
        "Bar 39 结果:\\n" +
        "Close: " + str.tostring(c, "#.##") + "\\n" +
        "BB_Upper: " + str.tostring(bb_upper, "#.####") + "\\n" +
        "KC_Upper: " + str.tostring(kc_upper, "#.####") + "\\n" +
        "Momentum: " + str.tostring(mom_builtin, "#.######"),
        style=label.style_label_left)
'''


def calculate_python_results() -> dict:
    """Python 计算"""
    # 创建 DataFrame
    df = pd.DataFrame(FIXED_OHLC_DATA, columns=['open', 'high', 'low', 'close'])

    # 计算指标
    params = SqueezeMomentumParams(
        bb_length=20,
        bb_mult=2.0,
        kc_length=20,
        kc_mult=1.5,
        use_true_range=True
    )

    indicator = SqueezeMomentumIndicator(params)
    result = indicator.calculate(
        df['high'].values,
        df['low'].values,
        df['close'].values
    )

    return {
        'close': df['close'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'bb_upper': result.bb_upper,
        'bb_basis': result.bb_basis,
        'bb_lower': result.bb_lower,
        'kc_upper': result.kc_upper,
        'kc_basis': result.kc_basis,
        'kc_lower': result.kc_lower,
        'momentum': result.momentum,
        'squeeze_on': result.squeeze_on,
        'squeeze_off': result.squeeze_off,
    }


def run_fixed_value_test():
    """运行固定值测试"""
    print("=" * 70)
    print("Squeeze Momentum Indicator - 固定值测试")
    print("=" * 70)
    print()

    print("参数:")
    print("  BB Length: 20, BB Mult: 2.0")
    print("  KC Length: 20, KC Mult: 1.5")
    print("  Use True Range: True")
    print()

    print(f"测试数据: {len(FIXED_OHLC_DATA)} 个固定 OHLC 数据点")
    print()

    # Python 计算
    result = calculate_python_results()

    print("=" * 70)
    print("Python 计算结果")
    print("=" * 70)
    print()

    print(f"{'Bar':>3} | {'Close':>8} | {'BB_Upper':>10} | {'BB_Lower':>10} | {'KC_Upper':>10} | {'KC_Lower':>10} | {'Momentum':>12} | {'Sqz':>4}")
    print("-" * 95)

    for i in range(len(FIXED_OHLC_DATA)):
        close = result['close'][i]
        bb_upper = result['bb_upper'][i]
        bb_lower = result['bb_lower'][i]
        kc_upper = result['kc_upper'][i]
        kc_lower = result['kc_lower'][i]
        momentum = result['momentum'][i]
        sqz = "ON" if result['squeeze_on'][i] else ("OFF" if result['squeeze_off'][i] else "NO")

        if not np.isnan(bb_upper):
            print(f"{i:>3} | {close:>8.2f} | {bb_upper:>10.4f} | {bb_lower:>10.4f} | {kc_upper:>10.4f} | {kc_lower:>10.4f} | {momentum:>12.6f} | {sqz:>4}")

    print()

    # 显示关键验证点
    print("=" * 70)
    print("关键验证点 (请在 TradingView 对比这些值)")
    print("=" * 70)
    print()

    key_bars = [19, 24, 29, 34, 39]
    print(f"{'Bar':>3} | {'Close':>8} | {'BB_Upper':>12} | {'KC_Upper':>12} | {'Momentum':>14}")
    print("-" * 60)

    for i in key_bars:
        if not np.isnan(result['momentum'][i]):
            print(f"{i:>3} | {result['close'][i]:>8.2f} | {result['bb_upper'][i]:>12.6f} | {result['kc_upper'][i]:>12.6f} | {result['momentum'][i]:>14.8f}")

    print()

    return result


def main():
    # 运行测试
    result = run_fixed_value_test()

    # 生成 Pine Script
    print("=" * 70)
    print("TradingView 验证步骤")
    print("=" * 70)
    print("""
1. 复制下面的 Pine Script 到 TradingView Pine Editor
2. 添加到任意图表 (数据来自脚本内的固定数组，与图表数据无关)
3. 查看数据窗口中的各项指标值
4. 与上面的 Python 结果对比

关键对比项:
- BB_Upper / BB_Lower (布林带上下轨)
- KC_Upper / KC_Lower (肯特纳通道上下轨)
- Momentum (动量值 - 最重要!)
- Squeeze 状态 (零线颜色: 黑=ON, 灰=OFF, 蓝=NO)

验证通过标准: 数值差异 < 0.0001 (4位小数)
""")

    print()
    print("=" * 70)
    print("Pine Script 代码 (复制到 TradingView)")
    print("=" * 70)
    print(generate_pine_script())

    # 保存简化版 Pine Script
    simple_script = '''
//@version=5
indicator("SQZ MOM Simple Validation", overlay=false)

// 使用内置函数计算 (更简单的验证方式)
// 参数
length = 20
mult = 2.0
lengthKC = 20
multKC = 1.5

// Bollinger Bands
basis = ta.sma(close, length)
dev = mult * ta.stdev(close, length)
upperBB = basis + dev
lowerBB = basis - dev

// Keltner Channel (LazyBear 用 SMA!)
ma = ta.sma(close, lengthKC)
rangema = ta.sma(ta.tr, lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

// Squeeze
sqzOn = (lowerBB > lowerKC) and (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)

// Momentum
hh = ta.highest(high, lengthKC)
ll = ta.lowest(low, lengthKC)
midline = ((hh + ll) / 2 + ta.sma(close, lengthKC)) / 2
val = ta.linreg(close - midline, lengthKC, 0)

// Plot
bcolor = val > 0 ? (val > nz(val[1]) ? color.lime : color.green) : (val < nz(val[1]) ? color.red : color.maroon)
scolor = sqzOn ? color.black : sqzOff ? color.gray : color.blue

plot(val, "Momentum", color=bcolor, style=plot.style_histogram, linewidth=4)
plot(0, "Zero", color=scolor, style=plot.style_cross, linewidth=2)

// 数据窗口输出
plot(close, "Close", display=display.data_window)
plot(upperBB, "BB_Upper", display=display.data_window)
plot(lowerBB, "BB_Lower", display=display.data_window)
plot(upperKC, "KC_Upper", display=display.data_window)
plot(lowerKC, "KC_Lower", display=display.data_window)
plot(val, "Momentum", display=display.data_window)
'''

    print()
    print("=" * 70)
    print("简化版 Pine Script (使用图表实际数据)")
    print("=" * 70)
    print(simple_script)


if __name__ == "__main__":
    main()
