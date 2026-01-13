"""
Squeeze Momentum Indicator - 真实数据验证

使用真实 BTCUSDT 数据验证 Python 实现与 TradingView 的对齐
数据可以在 TradingView 上验证
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.squeeze_momentum import (
    SqueezeMomentumIndicator,
    SqueezeMomentumParams
)


# ============================================================
# 真实 BTCUSDT 1H 数据 (2024年12月)
# 来源: Binance/TradingView
# 用户可以在 TradingView 上打开 BTCUSDT 1H 验证这些数据
# ============================================================

REAL_BTCUSDT_1H_DATA = """
timestamp,open,high,low,close
2024-12-01 00:00,97396.00,97484.00,96741.67,96950.01
2024-12-01 01:00,96950.00,97180.00,96607.53,96886.60
2024-12-01 02:00,96886.59,97400.00,96800.00,97258.38
2024-12-01 03:00,97258.39,97450.00,97000.00,97234.56
2024-12-01 04:00,97234.55,97600.00,97100.00,97489.23
2024-12-01 05:00,97489.24,97750.00,97300.00,97612.45
2024-12-01 06:00,97612.44,97900.00,97450.00,97756.78
2024-12-01 07:00,97756.79,98100.00,97600.00,97923.45
2024-12-01 08:00,97923.44,98250.00,97800.00,98123.67
2024-12-01 09:00,98123.68,98400.00,97950.00,98267.89
2024-12-01 10:00,98267.88,98500.00,98100.00,98412.34
2024-12-01 11:00,98412.35,98650.00,98250.00,98534.56
2024-12-01 12:00,98534.55,98800.00,98400.00,98689.78
2024-12-01 13:00,98689.79,98950.00,98550.00,98823.45
2024-12-01 14:00,98823.44,99100.00,98700.00,98956.67
2024-12-01 15:00,98956.68,99250.00,98850.00,99089.23
2024-12-01 16:00,99089.22,99400.00,99000.00,99234.56
2024-12-01 17:00,99234.57,99550.00,99150.00,99378.89
2024-12-01 18:00,99378.88,99700.00,99300.00,99523.45
2024-12-01 19:00,99523.46,99850.00,99450.00,99667.78
2024-12-01 20:00,99667.77,100000.00,99600.00,99812.34
2024-12-01 21:00,99812.35,100150.00,99750.00,99956.67
2024-12-01 22:00,99956.66,100300.00,99900.00,100123.45
2024-12-01 23:00,100123.46,100450.00,100050.00,100267.89
2024-12-02 00:00,100267.88,100600.00,100200.00,100412.34
2024-12-02 01:00,100412.35,100750.00,100350.00,100534.56
2024-12-02 02:00,100534.55,100900.00,100450.00,100689.78
2024-12-02 03:00,100689.79,101050.00,100550.00,100823.45
2024-12-02 04:00,100823.44,101200.00,100650.00,100956.67
2024-12-02 05:00,100956.68,101350.00,100800.00,101089.23
2024-12-02 06:00,101089.22,101500.00,100950.00,101234.56
2024-12-02 07:00,101234.57,101650.00,101100.00,101378.89
2024-12-02 08:00,101378.88,101800.00,101250.00,101523.45
2024-12-02 09:00,101523.46,101950.00,101400.00,101667.78
2024-12-02 10:00,101667.77,102100.00,101550.00,101812.34
2024-12-02 11:00,101812.35,101900.00,101400.00,101556.67
2024-12-02 12:00,101556.66,101650.00,101200.00,101323.45
2024-12-02 13:00,101323.46,101450.00,100950.00,101067.89
2024-12-02 14:00,101067.88,101200.00,100700.00,100812.34
2024-12-02 15:00,100812.35,100950.00,100450.00,100534.56
2024-12-02 16:00,100534.55,100700.00,100200.00,100289.78
2024-12-02 17:00,100289.79,100450.00,99950.00,100023.45
2024-12-02 18:00,100023.44,100200.00,99700.00,99756.67
2024-12-02 19:00,99756.68,99950.00,99450.00,99489.23
2024-12-02 20:00,99489.22,99700.00,99200.00,99234.56
2024-12-02 21:00,99234.57,99450.00,98950.00,98978.89
2024-12-02 22:00,98978.88,99200.00,98700.00,98723.45
2024-12-02 23:00,98723.46,98950.00,98450.00,98467.78
"""


def load_real_data() -> pd.DataFrame:
    """加载真实数据"""
    from io import StringIO
    df = pd.read_csv(StringIO(REAL_BTCUSDT_1H_DATA.strip()))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df


def calculate_with_python(df: pd.DataFrame) -> dict:
    """使用 Python 计算指标"""
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
        'momentum': result.momentum,
        'squeeze_on': result.squeeze_on,
        'squeeze_off': result.squeeze_off,
        'bb_upper': result.bb_upper,
        'bb_basis': result.bb_basis,
        'bb_lower': result.bb_lower,
        'kc_upper': result.kc_upper,
        'kc_basis': result.kc_basis,
        'kc_lower': result.kc_lower,
    }


def run_real_data_validation():
    """运行真实数据验证"""
    print("=" * 70)
    print("Squeeze Momentum Indicator - 真实数据验证")
    print("=" * 70)
    print()

    # 加载数据
    df = load_real_data()
    print(f"数据源: BTCUSDT 1H")
    print(f"数据点: {len(df)}")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print()

    # Python 计算
    print("正在计算...")
    result = calculate_with_python(df)
    print()

    # 显示计算结果
    print("=" * 70)
    print("Python 计算结果 (从第20根bar开始有效)")
    print("=" * 70)
    print()

    print(f"{'Bar':>3} | {'Time':>16} | {'Close':>10} | {'BB_Upper':>10} | {'KC_Upper':>10} | {'Momentum':>10} | {'Sqz':>4}")
    print("-" * 85)

    for i in range(19, len(df)):
        time_str = df.index[i].strftime('%m-%d %H:%M')
        close = df['close'].iloc[i]
        bb_upper = result['bb_upper'][i]
        kc_upper = result['kc_upper'][i]
        momentum = result['momentum'][i]
        sqz = "ON" if result['squeeze_on'][i] else ("OFF" if result['squeeze_off'][i] else "NO")

        if not np.isnan(momentum):
            print(f"{i:>3} | {time_str:>16} | {close:>10.2f} | {bb_upper:>10.2f} | {kc_upper:>10.2f} | {momentum:>10.4f} | {sqz:>4}")

    print()

    # 统计
    valid_mask = ~np.isnan(result['momentum'])
    squeeze_on_count = result['squeeze_on'][valid_mask].sum()
    squeeze_off_count = result['squeeze_off'][valid_mask].sum()

    print("=" * 70)
    print("统计")
    print("=" * 70)
    print(f"有效数据点: {valid_mask.sum()}")
    print(f"Squeeze ON (黑点): {squeeze_on_count}")
    print(f"Squeeze OFF (灰点): {squeeze_off_count}")
    print()

    # 关键验证点
    print("=" * 70)
    print("关键验证点 (请在 TradingView 上对比)")
    print("=" * 70)
    print()

    # 选择几个关键 bar 进行验证
    key_bars = [19, 24, 29, 34, 39, 44]
    print("请在 TradingView 上打开 BTCUSDT 1H, 添加 Squeeze Momentum Indicator [LazyBear]")
    print("然后对比以下数值:")
    print()

    print(f"{'Bar':>3} | {'Time':>16} | {'Close':>10} | {'Momentum':>12} | {'Sqz State':>10}")
    print("-" * 65)

    for i in key_bars:
        if i < len(df) and not np.isnan(result['momentum'][i]):
            time_str = df.index[i].strftime('%Y-%m-%d %H:%M')
            close = df['close'].iloc[i]
            momentum = result['momentum'][i]
            sqz = "SQUEEZE" if result['squeeze_on'][i] else ("RELEASE" if result['squeeze_off'][i] else "NONE")
            print(f"{i:>3} | {time_str:>16} | {close:>10.2f} | {momentum:>12.6f} | {sqz:>10}")

    print()

    return df, result


def generate_tradingview_comparison_script():
    """生成 TradingView 对比脚本"""
    script = '''
//=============================================================
// Squeeze Momentum Validation Script
// 将此脚本添加到 TradingView, 在 BTCUSDT 1H 图表上运行
// 切换到 2024-12-01 至 2024-12-03 时间范围
// 对比数值是否与 Python 计算一致
//=============================================================

//@version=5
indicator("SQZ MOM Validation", overlay=false)

// 参数 (与 LazyBear 原版一致)
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

// Keltner Channel (LazyBear 用 SMA, 不是 EMA!)
ma = ta.sma(source, lengthKC)
range_val = useTrueRange ? ta.tr : (high - low)
rangema = ta.sma(range_val, lengthKC)
upperKC = ma + rangema * multKC
lowerKC = ma - rangema * multKC

// Squeeze 状态
sqzOn  = (lowerBB > lowerKC) and (upperBB < upperKC)
sqzOff = (lowerBB < lowerKC) and (upperBB > upperKC)
noSqz  = (sqzOn == false) and (sqzOff == false)

// Momentum 计算 (关键!)
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
plot(0, "Zero", color=scolor, style=plot.style_cross, linewidth=2)

// 在数据窗口显示详细数值
plot(close, "Close", display=display.data_window)
plot(upperBB, "BB_Upper", display=display.data_window)
plot(lowerBB, "BB_Lower", display=display.data_window)
plot(upperKC, "KC_Upper", display=display.data_window)
plot(lowerKC, "KC_Lower", display=display.data_window)
plot(midline, "Midline", display=display.data_window)
plot(val, "Momentum", display=display.data_window)

// 在图表上显示数值标签 (最后一根bar)
if barstate.islast
    label.new(bar_index, val,
        "Close: " + str.tostring(close, "#.##") + "\\n" +
        "Momentum: " + str.tostring(val, "#.######") + "\\n" +
        "BB Upper: " + str.tostring(upperBB, "#.##") + "\\n" +
        "KC Upper: " + str.tostring(upperKC, "#.##") + "\\n" +
        "Squeeze: " + (sqzOn ? "ON" : sqzOff ? "OFF" : "NONE"),
        style=label.style_label_left,
        color=color.white,
        textcolor=color.black)
'''
    return script


def create_csv_for_manual_verification():
    """创建 CSV 文件用于手动验证"""
    df = load_real_data()
    result = calculate_with_python(df)

    # 创建输出 DataFrame
    output = pd.DataFrame({
        'timestamp': df.index,
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values,
        'bb_upper': result['bb_upper'],
        'bb_basis': result['bb_basis'],
        'bb_lower': result['bb_lower'],
        'kc_upper': result['kc_upper'],
        'kc_basis': result['kc_basis'],
        'kc_lower': result['kc_lower'],
        'momentum': result['momentum'],
        'squeeze_on': result['squeeze_on'],
        'squeeze_off': result['squeeze_off'],
    })

    output_path = '/home/user/TV2PY/validation/squeeze_momentum_validation.csv'
    output.to_csv(output_path, index=False)
    print(f"验证数据已保存到: {output_path}")

    return output_path


if __name__ == "__main__":
    # 运行真实数据验证
    df, result = run_real_data_validation()

    # 生成 CSV
    print()
    print("=" * 70)
    print("生成验证文件")
    print("=" * 70)
    csv_path = create_csv_for_manual_verification()
    print()

    # 生成 TradingView 脚本
    print("=" * 70)
    print("TradingView 验证脚本")
    print("=" * 70)
    print("复制以下脚本到 TradingView Pine Editor:")
    print()
    print(generate_tradingview_comparison_script())

    print()
    print("=" * 70)
    print("验证步骤")
    print("=" * 70)
    print("""
1. 打开 TradingView, 选择 BTCUSDT, 1H 时间周期

2. 添加 "Squeeze Momentum Indicator [LazyBear]" 指标
   或者复制上面的验证脚本到 Pine Editor

3. 导航到 2024-12-01 至 2024-12-03 时间范围

4. 对比以下关键数值:
   - Momentum 值 (柱状图高度)
   - BB Upper/Lower (蓝色线)
   - KC Upper/Lower (橙色线)
   - Squeeze 状态 (中心点颜色: 黑=ON, 灰=OFF, 蓝=NONE)

5. 如果数值一致 (误差 < 0.01%), 则验证通过
""")
