"""
Pine Script 原子级操作模块

从最底层开始构建，确保每一层都与 TradingView 对齐。
"""

from validation.primitives.pine_primitives import (
    # Level 0: 数据类型
    na,
    is_na,
    nz,
    fixnan,
    # Level 1: 历史引用
    series_ref,
    # Level 2: 数学函数
    math,
    PineMath,
    # Level 3: 核心平滑
    alpha_smooth,
    alpha_smooth_manual,
    # Level 4: 基础累加
    cum,
    change,
    rising,
    falling,
    # Level 5: 条件运算
    iff,
    # Level 6: 比较交叉
    crossover,
    crossunder,
    cross,
    # Level 7: 移动平均
    sma,
    ema,
    rma,
)

__all__ = [
    "na", "is_na", "nz", "fixnan",
    "series_ref",
    "math", "PineMath",
    "alpha_smooth", "alpha_smooth_manual",
    "cum", "change", "rising", "falling",
    "iff",
    "crossover", "crossunder", "cross",
    "sma", "ema", "rma",
]
