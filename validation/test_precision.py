"""
Python vs TradingView 精度对比测试

用固定数据验证计算是否一致
"""

import pandas as pd
import numpy as np


def test_ema_implementations():
    """
    测试不同 EMA 实现方式

    TradingView EMA 文档说明:
    - 使用 multiplier = 2 / (length + 1)
    - 第一个值用 SMA 作为种子 (根据测试推断)
    """

    # 固定测试数据 (10 个价格)
    prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]
    src = pd.Series(prices)
    length = 3

    print("=" * 60)
    print("EMA 实现对比测试")
    print("=" * 60)
    print(f"输入数据: {prices}")
    print(f"EMA 周期: {length}")
    print()

    # 方法 1: pandas ewm (adjust=False)
    alpha = 2.0 / (length + 1)
    ema_pandas = src.ewm(alpha=alpha, adjust=False).mean()
    print(f"方法1 - pandas ewm(adjust=False):")
    print(f"  alpha = {alpha}")
    print(f"  结果: {ema_pandas.values}")
    print()

    # 方法 2: pandas ewm (adjust=True) - 默认
    ema_pandas_adjust = src.ewm(alpha=alpha, adjust=True).mean()
    print(f"方法2 - pandas ewm(adjust=True):")
    print(f"  结果: {ema_pandas_adjust.values}")
    print()

    # 方法 3: 手动递归 (第一个值 = 第一个价格)
    ema_manual_v1 = [prices[0]]
    for i in range(1, len(prices)):
        ema_val = alpha * prices[i] + (1 - alpha) * ema_manual_v1[-1]
        ema_manual_v1.append(ema_val)
    print(f"方法3 - 手动递归 (seed=第一个价格):")
    print(f"  公式: EMA[i] = α × Price[i] + (1-α) × EMA[i-1]")
    print(f"  结果: {ema_manual_v1}")
    print()

    # 方法 4: 手动递归 (第一个值 = SMA)
    sma_seed = sum(prices[:length]) / length
    ema_manual_v2 = [np.nan] * (length - 1) + [sma_seed]
    for i in range(length, len(prices)):
        ema_val = alpha * prices[i] + (1 - alpha) * ema_manual_v2[-1]
        ema_manual_v2.append(ema_val)
    print(f"方法4 - 手动递归 (seed=SMA):")
    print(f"  SMA seed = {sma_seed}")
    print(f"  结果: {ema_manual_v2}")
    print()

    # 方法 5: 用 span 参数
    ema_span = src.ewm(span=length, adjust=False).mean()
    print(f"方法5 - pandas ewm(span={length}):")
    print(f"  结果: {ema_span.values}")
    print()

    print("=" * 60)
    print("对比分析")
    print("=" * 60)
    print()
    print("pandas ewm(adjust=False) vs 手动递归(seed=第一个价格):")
    print(f"  差异: {np.max(np.abs(np.array(ema_manual_v1) - ema_pandas.values))}")
    print()
    print("关键问题: TradingView 用哪种方式?")
    print("  - 需要从 TradingView 导出实际数据对比")
    print()

    return {
        "pandas_no_adjust": ema_pandas.values,
        "pandas_adjust": ema_pandas_adjust.values,
        "manual_seed_first": ema_manual_v1,
        "manual_seed_sma": ema_manual_v2,
    }


def test_rma_implementations():
    """测试 RMA (Wilder's Smoothing) 实现"""

    prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]
    src = pd.Series(prices)
    length = 3

    print()
    print("=" * 60)
    print("RMA 实现对比测试")
    print("=" * 60)
    print(f"输入数据: {prices}")
    print(f"RMA 周期: {length}")
    print()

    # RMA alpha = 1/length (不是 2/(length+1))
    alpha = 1.0 / length

    # 方法 1: pandas ewm
    rma_pandas = src.ewm(alpha=alpha, adjust=False).mean()
    print(f"方法1 - pandas ewm(alpha=1/n):")
    print(f"  alpha = {alpha}")
    print(f"  结果: {rma_pandas.values}")
    print()

    # 方法 2: 手动
    rma_manual = [prices[0]]
    for i in range(1, len(prices)):
        rma_val = alpha * prices[i] + (1 - alpha) * rma_manual[-1]
        rma_manual.append(rma_val)
    print(f"方法2 - 手动递归:")
    print(f"  结果: {rma_manual}")
    print()


def generate_tv_test_script():
    """生成 TradingView 测试脚本"""

    script = '''
//@version=5
indicator("EMA/RMA 精度测试")

// 使用 close 价格
src = close

// EMA 测试
ema_3 = ta.ema(src, 3)
ema_14 = ta.ema(src, 14)

// RMA 测试
rma_3 = ta.rma(src, 3)
rma_14 = ta.rma(src, 14)

// SMA 对比
sma_3 = ta.sma(src, 3)
sma_14 = ta.sma(src, 14)

// 输出到数据窗口
plot(src, "close")
plot(ema_3, "ema_3")
plot(ema_14, "ema_14")
plot(rma_3, "rma_3")
plot(rma_14, "rma_14")
plot(sma_3, "sma_3")
plot(sma_14, "sma_14")

// 导出: 右键图表 -> 导出图表数据
'''
    return script


if __name__ == "__main__":
    test_ema_implementations()
    test_rma_implementations()

    print()
    print("=" * 60)
    print("TradingView 测试脚本")
    print("=" * 60)
    print(generate_tv_test_script())

    print()
    print("=" * 60)
    print("结论")
    print("=" * 60)
    print("""
要确认 Python 和 TradingView 是否对齐:

1. 在 TradingView 运行上面的脚本
2. 导出数据 (包含 close, ema_3, ema_14, rma_3, rma_14)
3. 用 Python 计算相同数据
4. 对比结果

可能的差异:
- adjust=True vs adjust=False
- 种子值: 第一个价格 vs SMA
- 浮点精度: 通常差异 < 1e-10
""")
