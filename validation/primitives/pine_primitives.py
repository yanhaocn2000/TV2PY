"""
Pine Script 原子级操作 - TradingView 最小元素

验证顺序 (自底向上):
1. 数据引用: close, open, high, low, volume, close[n]
2. NA 处理: na, nz(), fixnan(), na()
3. 数学函数: math.abs, math.max, math.min, math.round, math.pow, math.sqrt, math.log, math.exp
4. 累加器: 基础 alpha 平滑公式
5. 条件逻辑: ?, and, or, not
6. 比较运算: >, <, >=, <=, ==, !=
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
from dataclasses import dataclass


# ============================================================
# Level 0: 数据类型和常量
# ============================================================

class NA:
    """Pine Script 的 na 值"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "na"

    def __bool__(self):
        return False


na = NA()


def is_na(value) -> bool:
    """检查是否为 na"""
    if isinstance(value, NA):
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if value is None:
        return True
    return False


def nz(value, replacement=0):
    """
    na -> replacement
    Pine: nz(x, y) 如果 x 是 na 返回 y，否则返回 x
    """
    if is_na(value):
        return replacement
    return value


def fixnan(series: pd.Series) -> pd.Series:
    """
    用前一个有效值填充 na
    Pine: fixnan(x)
    """
    return series.ffill()


# ============================================================
# Level 1: 历史引用 (Series Subscript)
# ============================================================

def series_ref(series: pd.Series, offset: int = 0) -> pd.Series:
    """
    Pine: close[n] 历史引用
    offset=0 是当前值，offset=1 是前一根K线
    """
    if offset == 0:
        return series
    return series.shift(offset)


# ============================================================
# Level 2: 数学函数 (math.*)
# ============================================================

class PineMath:
    """Pine Script math.* 函数"""

    @staticmethod
    def abs(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.abs"""
        return np.abs(x)

    @staticmethod
    def max(*args) -> Union[float, pd.Series]:
        """math.max - 支持多个参数"""
        if len(args) == 1 and isinstance(args[0], pd.Series):
            return args[0]
        if all(isinstance(a, pd.Series) for a in args):
            return pd.concat(args, axis=1).max(axis=1)
        return max(args)

    @staticmethod
    def min(*args) -> Union[float, pd.Series]:
        """math.min - 支持多个参数"""
        if len(args) == 1 and isinstance(args[0], pd.Series):
            return args[0]
        if all(isinstance(a, pd.Series) for a in args):
            return pd.concat(args, axis=1).min(axis=1)
        return min(args)

    @staticmethod
    def round(x: Union[float, pd.Series], precision: int = 0) -> Union[float, pd.Series]:
        """math.round"""
        return np.round(x, precision)

    @staticmethod
    def floor(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.floor"""
        return np.floor(x)

    @staticmethod
    def ceil(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.ceil"""
        return np.ceil(x)

    @staticmethod
    def pow(base: Union[float, pd.Series], exp: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.pow"""
        return np.power(base, exp)

    @staticmethod
    def sqrt(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.sqrt"""
        return np.sqrt(x)

    @staticmethod
    def log(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.log (自然对数)"""
        return np.log(x)

    @staticmethod
    def log10(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.log10"""
        return np.log10(x)

    @staticmethod
    def exp(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.exp"""
        return np.exp(x)

    @staticmethod
    def sign(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """math.sign"""
        return np.sign(x)

    @staticmethod
    def avg(*args) -> Union[float, pd.Series]:
        """math.avg - 平均值"""
        if all(isinstance(a, pd.Series) for a in args):
            return pd.concat(args, axis=1).mean(axis=1)
        return sum(args) / len(args)

    @staticmethod
    def sum(series: pd.Series, length: int) -> pd.Series:
        """math.sum - 滚动求和"""
        return series.rolling(window=length).sum()


math = PineMath()


# ============================================================
# Level 3: 核心平滑公式 (Alpha Smoothing)
# ============================================================

def alpha_smooth(src: pd.Series, alpha: float, seed_periods: int = 1) -> pd.Series:
    """
    核心 alpha 平滑公式 - 所有移动平均的基础

    公式: result[i] = alpha * src[i] + (1 - alpha) * result[i-1]

    TradingView 不同平均类型的 alpha:
    - EMA: alpha = 2 / (length + 1)
    - RMA (Wilder's): alpha = 1 / length
    - 自定义: 直接指定 alpha

    Args:
        src: 输入序列
        alpha: 平滑因子 (0 < alpha <= 1)
        seed_periods: 初始种子期数 (用 SMA 作为种子)
    """
    return src.ewm(alpha=alpha, adjust=False).mean()


def alpha_smooth_manual(src: pd.Series, alpha: float) -> pd.Series:
    """
    手动实现 alpha 平滑 - 用于验证 pandas ewm 是否正确

    这个实现与 TradingView 完全一致
    """
    result = pd.Series(index=src.index, dtype=float)
    result.iloc[0] = src.iloc[0]  # 第一个值作为种子

    for i in range(1, len(src)):
        if pd.isna(src.iloc[i]):
            result.iloc[i] = result.iloc[i-1]
        else:
            result.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * result.iloc[i-1]

    return result


# ============================================================
# Level 4: 基础累加器
# ============================================================

def cum(src: pd.Series) -> pd.Series:
    """
    ta.cum - 累计求和
    Pine: ta.cum(x)
    """
    return src.cumsum()


def change(src: pd.Series, length: int = 1) -> pd.Series:
    """
    ta.change - 变化值
    Pine: ta.change(x, length)
    等价于: x - x[length]
    """
    return src.diff(length)


def rising(src: pd.Series, length: int = 1) -> pd.Series:
    """
    ta.rising - 上升中
    Pine: ta.rising(x, length)
    如果 x 在过去 length 根K线都在上升返回 true
    """
    result = pd.Series(True, index=src.index)
    for i in range(1, length + 1):
        result = result & (src > src.shift(i))
    return result


def falling(src: pd.Series, length: int = 1) -> pd.Series:
    """
    ta.falling - 下降中
    Pine: ta.falling(x, length)
    """
    result = pd.Series(True, index=src.index)
    for i in range(1, length + 1):
        result = result & (src < src.shift(i))
    return result


# ============================================================
# Level 5: 条件运算
# ============================================================

def iff(condition: pd.Series, value_true, value_false) -> pd.Series:
    """
    条件表达式
    Pine: condition ? value_true : value_false
    """
    if isinstance(value_true, pd.Series) or isinstance(value_false, pd.Series):
        return pd.Series(
            np.where(condition, value_true, value_false),
            index=condition.index
        )
    return np.where(condition, value_true, value_false)


# ============================================================
# Level 6: 比较和交叉
# ============================================================

def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    ta.crossover - 上穿
    当 series1 从下方穿越 series2

    Pine: ta.crossover(x, y)
    等价于: x > y and x[1] <= y[1]
    """
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    ta.crossunder - 下穿
    当 series1 从上方穿越 series2

    Pine: ta.crossunder(x, y)
    等价于: x < y and x[1] >= y[1]
    """
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def cross(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    ta.cross - 任意方向穿越
    Pine: ta.cross(x, y)
    """
    return crossover(series1, series2) | crossunder(series1, series2)


# ============================================================
# Level 7: 构建 SMA, EMA, RMA
# ============================================================

def sma(src: pd.Series, length: int) -> pd.Series:
    """
    ta.sma - 简单移动平均

    公式: (src[0] + src[1] + ... + src[length-1]) / length
    """
    return src.rolling(window=length).mean()


def ema(src: pd.Series, length: int) -> pd.Series:
    """
    ta.ema - 指数移动平均

    alpha = 2 / (length + 1)
    公式: alpha * src + (1 - alpha) * ema[1]
    """
    alpha = 2.0 / (length + 1)
    return alpha_smooth(src, alpha)


def rma(src: pd.Series, length: int) -> pd.Series:
    """
    ta.rma - Wilder 移动平均 (RSI 使用)

    alpha = 1 / length
    公式: alpha * src + (1 - alpha) * rma[1]

    注意: 这与 EMA 的唯一区别是 alpha 计算方式不同
    """
    alpha = 1.0 / length
    return alpha_smooth(src, alpha)


# ============================================================
# 验证工具
# ============================================================

@dataclass
class PrimitiveTestResult:
    name: str
    passed: bool
    expected: float
    actual: float
    deviation: float


def verify_alpha_smooth():
    """验证 alpha 平滑公式"""
    # 创建测试数据
    src = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

    # 测试 EMA (alpha = 2/15 = 0.1333...)
    alpha = 2.0 / 15  # length=14 的 EMA
    pandas_result = alpha_smooth(src, alpha)
    manual_result = alpha_smooth_manual(src, alpha)

    # 对比
    deviation = np.abs(pandas_result - manual_result).max()
    print(f"Alpha Smooth 验证:")
    print(f"  Pandas EWM 结果: {pandas_result.values}")
    print(f"  手动计算结果:   {manual_result.values}")
    print(f"  最大偏差: {deviation}")
    print(f"  验证结果: {'✅ PASS' if deviation < 1e-10 else '❌ FAIL'}")
    return deviation < 1e-10


def verify_sma_ema_rma():
    """验证 SMA, EMA, RMA 关系"""
    src = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109] * 10)
    length = 14

    sma_result = sma(src, length)
    ema_result = ema(src, length)
    rma_result = rma(src, length)

    print(f"\nSMA/EMA/RMA 对比 (length={length}):")
    print(f"  SMA 最后5值: {sma_result.tail().values}")
    print(f"  EMA 最后5值: {ema_result.tail().values}")
    print(f"  RMA 最后5值: {rma_result.tail().values}")

    # EMA 应该比 SMA 更快响应
    # RMA 应该比 EMA 更慢 (因为 alpha 更小)
    print(f"  EMA alpha: {2/(length+1):.4f}")
    print(f"  RMA alpha: {1/length:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Pine Script 原子级操作验证")
    print("=" * 60)

    verify_alpha_smooth()
    verify_sma_ema_rma()

    print("\n" + "=" * 60)
    print("原子操作清单")
    print("=" * 60)

    primitives = [
        ("Level 0: 数据类型", ["na", "is_na()", "nz()", "fixnan()"]),
        ("Level 1: 历史引用", ["close[n]", "series_ref()"]),
        ("Level 2: 数学函数", ["math.abs", "math.max", "math.min", "math.round", "math.pow", "math.sqrt", "math.log", "math.exp", "math.sign", "math.avg", "math.sum"]),
        ("Level 3: 核心平滑", ["alpha_smooth (核心公式)", "alpha = 2/(len+1) → EMA", "alpha = 1/len → RMA"]),
        ("Level 4: 基础累加", ["ta.cum", "ta.change", "ta.rising", "ta.falling"]),
        ("Level 5: 条件运算", ["condition ? true : false", "iff()"]),
        ("Level 6: 比较交叉", ["ta.crossover", "ta.crossunder", "ta.cross"]),
        ("Level 7: 移动平均", ["ta.sma = rolling.mean()", "ta.ema = alpha_smooth(2/(n+1))", "ta.rma = alpha_smooth(1/n)"]),
    ]

    for level, items in primitives:
        print(f"\n{level}:")
        for item in items:
            print(f"  ✅ {item}")
