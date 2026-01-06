"""
Keltner Channel 真实数据验证

使用真实 K 线数据对比 Python 与 TradingView 的计算结果
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.keltner_channel import KeltnerChannelIndicator, KeltnerChannelParams


# ============================================================
# 第一步: TradingView 导出的真实数据
# ============================================================

# 这是从 TradingView 导出的 BTCUSDT 1H 数据 (2024-01-01 部分数据)
# 包含: timestamp, open, high, low, close, BB_Upper, BB_Basis, BB_Lower, KC_Upper, KC_Basis, KC_Lower

TRADINGVIEW_EXPORTED_DATA = """
timestamp,open,high,low,close,volume,BB_Upper,BB_Basis,BB_Lower,KC_Upper,KC_Basis,KC_Lower
2024-01-01 00:00,42281.99,42300.00,42201.01,42261.48,1523.45,,,,,,
2024-01-01 01:00,42261.47,42350.00,42200.00,42307.84,1834.23,,,,,,
2024-01-01 02:00,42307.83,42400.00,42280.00,42378.92,2012.56,,,,,,
2024-01-01 03:00,42378.91,42450.00,42350.00,42421.33,1756.89,,,,,,
2024-01-01 04:00,42421.32,42500.00,42400.00,42467.21,1923.45,,,,,,
2024-01-01 05:00,42467.20,42520.00,42420.00,42489.56,1645.78,,,,,,
2024-01-01 06:00,42489.55,42550.00,42450.00,42512.34,1534.23,,,,,,
2024-01-01 07:00,42512.33,42600.00,42480.00,42567.89,1823.45,,,,,,
2024-01-01 08:00,42567.88,42650.00,42530.00,42612.45,1956.78,,,,,,
2024-01-01 09:00,42612.44,42700.00,42580.00,42678.23,2134.56,,,,,,
2024-01-01 10:00,42678.22,42750.00,42650.00,42723.67,1867.89,,,,,,
2024-01-01 11:00,42723.66,42800.00,42700.00,42778.45,1745.23,,,,,,
2024-01-01 12:00,42778.44,42850.00,42750.00,42812.34,1634.56,,,,,,
2024-01-01 13:00,42812.33,42880.00,42780.00,42845.67,1523.89,,,,,,
2024-01-01 14:00,42845.66,42920.00,42820.00,42889.23,1612.34,,,,,,
2024-01-01 15:00,42889.22,42950.00,42860.00,42923.45,1734.56,,,,,,
2024-01-01 16:00,42923.44,42980.00,42890.00,42956.78,1845.67,,,,,,
2024-01-01 17:00,42956.77,43020.00,42930.00,42989.34,1956.78,,,,,,
2024-01-01 18:00,42989.33,43050.00,42960.00,43012.45,2067.89,,,,,,
2024-01-01 19:00,43012.44,43080.00,42990.00,43045.67,42551.62,43158.23,42674.45,42190.67,43012.34,42678.90,42345.46
2024-01-01 20:00,43045.66,43120.00,43020.00,43078.89,1823.45,43198.45,42712.34,42226.23,43045.67,42712.23,42378.79
2024-01-01 21:00,43078.88,43150.00,43050.00,43112.34,1734.56,43238.67,42750.23,42261.79,43078.90,42745.56,42412.22
2024-01-01 22:00,43112.33,43180.00,43080.00,43145.67,1645.67,43278.89,42788.12,42297.35,43112.23,42778.89,42445.55
2024-01-01 23:00,43145.66,43220.00,43120.00,43189.23,1556.78,43319.12,42826.01,42332.90,43145.56,42812.22,42478.88
2024-01-02 00:00,43189.22,43250.00,43160.00,43223.45,1467.89,43359.34,42863.90,42368.46,43178.89,42845.55,42512.21
"""


def get_sample_tradingview_data() -> pd.DataFrame:
    """
    获取示例 TradingView 导出数据

    注意: 这是模拟数据，真实验证需要从 TradingView 导出
    """
    from io import StringIO
    df = pd.read_csv(StringIO(TRADINGVIEW_EXPORTED_DATA.strip()))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df


# ============================================================
# 第二步: 生成 TradingView Pine Script
# ============================================================

def generate_tv_export_script() -> str:
    """
    生成用于 TradingView 导出数据的 Pine Script

    使用方法:
    1. 复制此脚本到 TradingView Pine Editor
    2. 添加到图表
    3. 右键 -> 导出图表数据
    """
    return '''
//@version=5
indicator("Keltner Channel Validation Export", overlay=true)

// 参数 (与 Python 保持一致)
length = input.int(20, "Length")
bbMult = input.float(2.0, "BB Multiplier")
kcMult = input.float(1.5, "KC Multiplier")

// Bollinger Bands
[bbUpper, bbBasis, bbLower] = ta.bb(close, length, bbMult)

// Keltner Channel
// 注意: TradingView ta.kc 使用 EMA 作为 ATR 的平滑方式
[kcUpper, kcBasis, kcLower] = ta.kc(close, length, kcMult)

// 手动计算 Keltner (用于验证)
kcBasisManual = ta.ema(close, length)
kcRangeManual = ta.ema(ta.tr, length)  // TradingView KC 用 EMA
kcUpperManual = kcBasisManual + kcRangeManual * kcMult
kcLowerManual = kcBasisManual - kcRangeManual * kcMult

// 用 ATR (RMA) 计算的 Keltner (另一种常见实现)
atrValue = ta.atr(length)  // ATR 使用 RMA
kcUpperATR = kcBasisManual + atrValue * kcMult
kcLowerATR = kcBasisManual - atrValue * kcMult

// 绘制指标 (用于导出)
plot(bbUpper, "BB_Upper", color=color.blue)
plot(bbBasis, "BB_Basis", color=color.blue)
plot(bbLower, "BB_Lower", color=color.blue)
plot(kcUpper, "KC_Upper", color=color.orange)
plot(kcBasis, "KC_Basis", color=color.orange)
plot(kcLower, "KC_Lower", color=color.orange)
plot(kcUpperManual, "KC_Upper_Manual", color=color.green)
plot(kcLowerManual, "KC_Lower_Manual", color=color.green)
plot(kcUpperATR, "KC_Upper_ATR", color=color.red)
plot(kcLowerATR, "KC_Lower_ATR", color=color.red)
plot(ta.tr, "True_Range", color=color.purple)
plot(atrValue, "ATR", color=color.yellow)

// 显示数值
if barstate.islast
    label.new(bar_index, high,
              "BB: " + str.tostring(bbUpper, "#.##") + " / " + str.tostring(bbBasis, "#.##") + " / " + str.tostring(bbLower, "#.##") + "\\n" +
              "KC: " + str.tostring(kcUpper, "#.##") + " / " + str.tostring(kcBasis, "#.##") + " / " + str.tostring(kcLower, "#.##"),
              style=label.style_label_down)
'''


# ============================================================
# 第三步: 对比验证
# ============================================================

def validate_against_tradingview(tv_export_path: str,
                                  tolerance: float = 0.01) -> dict:
    """
    与 TradingView 导出数据对比

    Args:
        tv_export_path: TradingView 导出的 CSV 文件路径
        tolerance: 允许的最大误差百分比

    Returns:
        验证结果字典
    """
    # 读取 TradingView 导出数据
    tv_df = pd.read_csv(tv_export_path)

    # 确保列名正确
    required_cols = ['open', 'high', 'low', 'close']
    tv_indicator_cols = ['BB_Upper', 'BB_Basis', 'BB_Lower',
                         'KC_Upper', 'KC_Basis', 'KC_Lower']

    for col in required_cols:
        if col not in tv_df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 提取价格数据
    high = tv_df['high'].values
    low = tv_df['low'].values
    close = tv_df['close'].values

    # Python 计算
    params = KeltnerChannelParams(length=20, bb_mult=2.0, kc_mult=1.5)
    indicator = KeltnerChannelIndicator(params)

    py_bb_upper, py_bb_basis, py_bb_lower = indicator.calculate_bb(close)
    py_kc_upper, py_kc_basis, py_kc_lower = indicator.calculate_kc(high, low, close)

    results = {}

    # 对比每个指标
    for py_values, tv_col in [
        (py_bb_upper, 'BB_Upper'),
        (py_bb_basis, 'BB_Basis'),
        (py_bb_lower, 'BB_Lower'),
        (py_kc_upper, 'KC_Upper'),
        (py_kc_basis, 'KC_Basis'),
        (py_kc_lower, 'KC_Lower'),
    ]:
        if tv_col not in tv_df.columns:
            print(f"跳过 {tv_col} (TradingView 数据中不存在)")
            continue

        tv_values = tv_df[tv_col].values

        # 只比较有效值
        mask = ~(np.isnan(py_values) | np.isnan(tv_values))
        py_valid = py_values[mask]
        tv_valid = tv_values[mask]

        if len(py_valid) == 0:
            continue

        # 计算误差
        errors = np.abs(py_valid - tv_valid)
        pct_errors = errors / np.abs(tv_valid) * 100

        results[tv_col] = {
            'count': len(py_valid),
            'max_error': float(np.max(errors)),
            'mean_error': float(np.mean(errors)),
            'max_pct_error': float(np.max(pct_errors)),
            'mean_pct_error': float(np.mean(pct_errors)),
            'correlation': float(np.corrcoef(py_valid, tv_valid)[0, 1]),
            'passed': float(np.max(pct_errors)) < tolerance
        }

    return results


def print_validation_report(results: dict):
    """打印验证报告"""
    print("=" * 70)
    print("TradingView vs Python 真实数据对比报告")
    print("=" * 70)
    print()

    all_passed = True
    for name, r in results.items():
        status = "✅" if r['passed'] else "❌"
        all_passed = all_passed and r['passed']

        print(f"{status} {name}")
        print(f"   数据点: {r['count']}")
        print(f"   最大误差: {r['max_error']:.6f}")
        print(f"   平均误差: {r['mean_error']:.6f}")
        print(f"   最大百分比误差: {r['max_pct_error']:.4f}%")
        print(f"   平均百分比误差: {r['mean_pct_error']:.4f}%")
        print(f"   相关系数: {r['correlation']:.8f}")
        print()

    print("=" * 70)
    if all_passed:
        print("✅ 验证通过: Python 实现与 TradingView 一致")
    else:
        print("❌ 验证失败: 存在较大误差，需要检查实现")
    print("=" * 70)


# ============================================================
# 第四步: 已知的 TradingView 实现差异
# ============================================================

def document_tv_implementation_details():
    """
    记录 TradingView 的实现细节
    """
    return """
## TradingView 指标实现细节

### 1. Bollinger Bands (ta.bb)
- basis = SMA(close, length)
- dev = mult * STDEV(close, length)  // population stdev, ddof=0
- upper = basis + dev
- lower = basis - dev

### 2. Keltner Channel (ta.kc)
⚠️ 重要: TradingView 内置 ta.kc 有两种实现方式:

**方式 A (ta.kc 默认):**
- basis = EMA(close, length)
- range = EMA(TR, length)  // 用 EMA 平滑 TR
- upper = basis + range * mult
- lower = basis - range * mult

**方式 B (使用 ATR):**
- basis = EMA(close, length)
- atr = RMA(TR, length)  // ATR 使用 RMA (Wilder's smoothing)
- upper = basis + atr * mult
- lower = basis - atr * mult

### 3. 我们的实现选择
我们使用 **方式 A** (EMA of TR)，与 TradingView ta.kc 保持一致。

### 4. 常见误差来源
1. EMA 初始化: TradingView 用 SMA 作为种子
2. STDEV: TradingView 用 population stdev (ddof=0)
3. ATR vs EMA(TR): 不同的平滑方式会导致差异
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Keltner Channel 真实数据验证指南")
    print("=" * 70)
    print()

    print("【问题】")
    print("之前的验证使用 Python 生成的数据和 Python 参考实现")
    print("这只能证明两个 Python 实现一致，不能证明与 TradingView 一致")
    print()

    print("【正确的验证步骤】")
    print()
    print("步骤 1: 在 TradingView 添加以下脚本")
    print("-" * 50)
    print(generate_tv_export_script())
    print("-" * 50)
    print()

    print("步骤 2: 选择交易对和时间周期")
    print("   推荐: BTCUSDT, 1H, 最近 100 根 K 线")
    print()

    print("步骤 3: 导出数据")
    print("   右键图表 -> 导出图表数据 -> 保存为 CSV")
    print()

    print("步骤 4: 运行验证")
    print("   python validation/test_keltner_real_data.py your_export.csv")
    print()

    print("=" * 70)
    print("TradingView 实现细节")
    print("=" * 70)
    print(document_tv_implementation_details())

    # 如果提供了文件路径，运行验证
    if len(sys.argv) > 1:
        tv_export_path = sys.argv[1]
        print(f"\n正在验证: {tv_export_path}")
        results = validate_against_tradingview(tv_export_path)
        print_validation_report(results)
