"""
Linear Regression Channel - Python Conversion

核心算法:
    使用最小二乘法拟合线性回归线
    y = a + b * x

    其中:
    b = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
    a = (sum(y) - b * sum(x)) / n

通道:
    Upper = Linear Regression + mult * StdDev
    Lower = Linear Regression - mult * StdDev
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class LinearRegressionResult:
    """Linear Regression 计算结果"""
    linreg: np.ndarray         # 线性回归值
    upper: np.ndarray          # 上轨
    lower: np.ndarray          # 下轨
    slope: np.ndarray          # 斜率
    intercept: np.ndarray      # 截距
    r_squared: np.ndarray      # R²


class LinearRegressionChannel:
    """
    Linear Regression Channel

    使用线性回归和标准差构建通道。

    Parameters:
        period: 回望周期 - 默认 100
        mult: 标准差乘数 - 默认 2.0
        offset: 预测偏移 - 默认 0 (当前值)
    """

    def __init__(
        self,
        period: int = 100,
        mult: float = 2.0,
        offset: int = 0,
    ):
        self.period = period
        self.mult = mult
        self.offset = offset

    def _linear_regression(
        self,
        data: np.ndarray,
        period: int,
        offset: int = 0,
    ) -> tuple:
        """
        计算线性回归

        返回: (预测值, 斜率, 截距, R²)
        """
        n = len(data)
        linreg = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        intercept = np.full(n, np.nan)
        r_squared = np.full(n, np.nan)

        for i in range(period - 1, n):
            y = data[i - period + 1:i + 1]
            x = np.arange(period)

            # 计算线性回归参数
            n_points = period
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x ** 2)

            b = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x ** 2)
            a = (sum_y - b * sum_x) / n_points

            slope[i] = b
            intercept[i] = a

            # 预测值 (当前点 + offset)
            predict_x = period - 1 + offset
            linreg[i] = a + b * predict_x

            # R² 计算
            y_pred = a + b * x
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            if ss_tot > 0:
                r_squared[i] = 1 - (ss_res / ss_tot)

        return linreg, slope, intercept, r_squared

    def _stdev(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算标准差"""
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            result[i] = np.std(data[i - period + 1:i + 1], ddof=0)
        return result

    def calculate(self, close: np.ndarray) -> LinearRegressionResult:
        """计算 Linear Regression Channel"""
        close = np.asarray(close, dtype=float)

        # 计算线性回归
        linreg, slope, intercept, r_squared = self._linear_regression(
            close, self.period, self.offset
        )

        # 计算标准差
        stdev = self._stdev(close, self.period)

        # 计算通道
        upper = linreg + self.mult * stdev
        lower = linreg - self.mult * stdev

        return LinearRegressionResult(
            linreg=linreg,
            upper=upper,
            lower=lower,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
        )

    def get_signals(
        self,
        close: np.ndarray,
        result: LinearRegressionResult,
    ) -> dict:
        """获取交易信号"""
        n = len(close)

        # 价格位置
        above_upper = close > result.upper
        below_lower = close < result.lower

        # 斜率方向
        uptrend = result.slope > 0
        downtrend = result.slope < 0

        return {
            "above_upper": above_upper,
            "below_lower": below_lower,
            "uptrend": uptrend,
            "downtrend": downtrend,
        }


class LinearRegressionPyneCore:
    """PyneCore 兼容的 Linear Regression 实现"""

    def __init__(self, period: int = 100, mult: float = 2.0):
        self.indicator = LinearRegressionChannel(period=period, mult=mult)

    def __call__(self, close: pd.Series) -> pd.DataFrame:
        result = self.indicator.calculate(close.values)
        return pd.DataFrame({
            "linreg": result.linreg,
            "upper": result.upper,
            "lower": result.lower,
            "slope": result.slope,
        }, index=close.index)


# TradingView ta.linreg() 兼容函数
def linreg(data: np.ndarray, period: int, offset: int = 0) -> np.ndarray:
    """
    TradingView ta.linreg() 兼容函数

    计算线性回归值，可选偏移

    Parameters:
        data: 输入数据
        period: 回望周期
        offset: 预测偏移 (0=当前, 1=下一个, -1=前一个)
    """
    indicator = LinearRegressionChannel(period=period, offset=offset)
    result = indicator.calculate(data)
    return result.linreg


PINE_SCRIPT_LINREG = '''
//@version=5
indicator("Linear Regression Channel", overlay=true)

length = input.int(100, "Period")
mult = input.float(2.0, "Multiplier")

// 线性回归
linreg_val = ta.linreg(close, length, 0)
linreg_dev = mult * ta.stdev(close, length)

upper = linreg_val + linreg_dev
lower = linreg_val - linreg_dev

plot(linreg_val, "Linear Regression", color=color.yellow)
plot(upper, "Upper", color=color.blue)
plot(lower, "Lower", color=color.blue)
'''


def main():
    print("=" * 60)
    print("Linear Regression Channel - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 150

    # 带趋势的数据
    base_price = 100.0
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.randn(n) * 0.5)
    close = base_price + trend + noise

    indicator = LinearRegressionChannel(period=50)
    result = indicator.calculate(close)

    print(f"\n参数: period={indicator.period}, mult={indicator.mult}")
    print(f"\n最后 10 个数据点:")
    print("-" * 70)
    print(f"{'Bar':<5} {'Close':<10} {'LinReg':<10} {'Upper':<10} {'Lower':<10} {'R²':<8}")
    print("-" * 70)

    for i in range(n - 10, n):
        lr_str = f"{result.linreg[i]:.2f}" if not np.isnan(result.linreg[i]) else "NaN"
        up_str = f"{result.upper[i]:.2f}" if not np.isnan(result.upper[i]) else "NaN"
        lo_str = f"{result.lower[i]:.2f}" if not np.isnan(result.lower[i]) else "NaN"
        r2_str = f"{result.r_squared[i]:.3f}" if not np.isnan(result.r_squared[i]) else "NaN"
        print(f"{i:<5} {close[i]:<10.2f} {lr_str:<10} {up_str:<10} {lo_str:<10} {r2_str:<8}")


if __name__ == "__main__":
    main()
