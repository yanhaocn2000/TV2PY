"""
TradingView 对齐的 EMA/RMA 实现

关键发现 (来自 PyneCore 源码分析):
1. EMA 前 length 根 K 线返回 SMA 作为种子值
2. 之后使用递归公式: alpha * src + (1 - alpha) * prev
3. RMA 就是 alpha=1/length 的 EMA
"""

import pandas as pd
import numpy as np


def tv_sma(src: pd.Series, length: int) -> pd.Series:
    """TradingView ta.sma - 简单移动平均"""
    return src.rolling(window=length).mean()


def tv_ema(src: pd.Series, length: int) -> pd.Series:
    """
    TradingView ta.ema - 指数移动平均

    与 TradingView 完全对齐:
    - 前 length 根: 返回 SMA (作为种子)
    - 之后: alpha * src + (1-alpha) * prev
    """
    alpha = 2.0 / (length + 1)

    result = pd.Series(index=src.index, dtype=float)
    sma = src.rolling(window=length).mean()

    for i in range(len(src)):
        if i < length - 1:
            # 前 length-1 根: NA
            result.iloc[i] = np.nan
        elif i == length - 1:
            # 第 length 根: 用 SMA 作为种子
            result.iloc[i] = sma.iloc[i]
        else:
            # 之后: 递归公式
            result.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * result.iloc[i - 1]

    return result


def tv_rma(src: pd.Series, length: int) -> pd.Series:
    """
    TradingView ta.rma - Wilder 移动平均

    与 EMA 相同，只是 alpha = 1/length
    """
    alpha = 1.0 / length

    result = pd.Series(index=src.index, dtype=float)
    sma = src.rolling(window=length).mean()

    for i in range(len(src)):
        if i < length - 1:
            result.iloc[i] = np.nan
        elif i == length - 1:
            result.iloc[i] = sma.iloc[i]
        else:
            result.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * result.iloc[i - 1]

    return result


def compare_implementations():
    """对比不同实现"""
    prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]
    src = pd.Series(prices)
    length = 3

    print("=" * 70)
    print("EMA 实现对比 (length=3)")
    print("=" * 70)
    print(f"输入: {prices}")
    print()

    # 方法 1: 原来的实现 (seed = 第一个价格)
    alpha = 2.0 / (length + 1)
    ema_old = src.ewm(alpha=alpha, adjust=False).mean()

    # 方法 2: TradingView 对齐 (seed = SMA)
    ema_tv = tv_ema(src, length)

    print(f"{'Bar':<5} {'Price':<10} {'旧实现':<15} {'TV对齐':<15} {'差异':<10}")
    print("-" * 60)
    for i in range(len(prices)):
        old_val = ema_old.iloc[i]
        tv_val = ema_tv.iloc[i]
        diff = abs(old_val - tv_val) if not np.isnan(tv_val) else "-"
        tv_str = f"{tv_val:.6f}" if not np.isnan(tv_val) else "NaN"
        diff_str = f"{diff:.6f}" if isinstance(diff, float) else diff
        print(f"{i:<5} {prices[i]:<10} {old_val:<15.6f} {tv_str:<15} {diff_str}")

    print()
    print("=" * 70)
    print("关键区别")
    print("=" * 70)
    print("""
旧实现: Bar 0 就有值 (用第一个价格作为种子)
TV对齐: Bar 0-1 是 NaN, Bar 2 开始有值 (用 SMA 作为种子)

这就是为什么之前对不上的原因！
""")

    # RMA 对比
    print("=" * 70)
    print("RMA 实现对比 (length=3)")
    print("=" * 70)

    alpha_rma = 1.0 / length
    rma_old = src.ewm(alpha=alpha_rma, adjust=False).mean()
    rma_tv = tv_rma(src, length)

    print(f"{'Bar':<5} {'Price':<10} {'旧实现':<15} {'TV对齐':<15}")
    print("-" * 50)
    for i in range(len(prices)):
        old_val = rma_old.iloc[i]
        tv_val = rma_tv.iloc[i]
        tv_str = f"{tv_val:.6f}" if not np.isnan(tv_val) else "NaN"
        print(f"{i:<5} {prices[i]:<10} {old_val:<15.6f} {tv_str:<15}")


if __name__ == "__main__":
    compare_implementations()
