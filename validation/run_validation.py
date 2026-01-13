"""
TV2PY 验证工具 - 对比 TradingView 和 Python 计算结果

使用方法:
1. 在 TradingView 中运行策略并导出数据
2. 将数据放入 validation/data/ 目录
3. 运行此脚本进行对比验证

TradingView 数据导出格式 (CSV):
- timestamp: 时间戳
- open, high, low, close, volume: OHLCV 数据
- ema_fast, ema_slow: 指标值
- long_signal, short_signal: 交易信号 (true/false)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from validation.validators import IndicatorValidator, SignalValidator


def generate_test_data(bars: int = 1000) -> pd.DataFrame:
    """生成测试数据 (模拟 TradingView 导出)"""
    np.random.seed(42)

    # 生成价格数据
    returns = np.random.randn(bars) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=bars, freq="4h"),
        "open": close * (1 + np.random.randn(bars) * 0.001),
        "high": close * (1 + np.abs(np.random.randn(bars)) * 0.01),
        "low": close * (1 - np.abs(np.random.randn(bars)) * 0.01),
        "close": close,
        "volume": np.random.randint(1000, 10000, bars),
    })

    return df


def calculate_python_indicators(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    """使用 Python 计算指标"""
    result = df.copy()

    # EMA 计算 (与 TradingView ta.ema 一致)
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)

    result["ema_fast"] = df["close"].ewm(alpha=alpha_fast, adjust=False).mean()
    result["ema_slow"] = df["close"].ewm(alpha=alpha_slow, adjust=False).mean()

    # 交叉信号
    result["long_signal"] = (result["ema_fast"] > result["ema_slow"]) & (
        result["ema_fast"].shift(1) <= result["ema_slow"].shift(1)
    )
    result["short_signal"] = (result["ema_fast"] < result["ema_slow"]) & (
        result["ema_fast"].shift(1) >= result["ema_slow"].shift(1)
    )

    return result


def simulate_tradingview_export(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    """
    模拟 TradingView 导出数据
    (实际使用时，替换为真实的 TradingView 导出数据)
    """
    result = df.copy()

    # 模拟 TradingView 的 EMA 计算
    # TradingView 和 Python ewm 应该完全一致
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)

    result["tv_ema_fast"] = df["close"].ewm(alpha=alpha_fast, adjust=False).mean()
    result["tv_ema_slow"] = df["close"].ewm(alpha=alpha_slow, adjust=False).mean()

    # 交叉信号
    result["tv_long_signal"] = (result["tv_ema_fast"] > result["tv_ema_slow"]) & (
        result["tv_ema_fast"].shift(1) <= result["tv_ema_slow"].shift(1)
    )
    result["tv_short_signal"] = (result["tv_ema_fast"] < result["tv_ema_slow"]) & (
        result["tv_ema_fast"].shift(1) >= result["tv_ema_slow"].shift(1)
    )

    return result


def run_ema_cross_validation():
    """运行 EMA Cross 策略验证"""
    print("=" * 60)
    print("EMA Cross 策略验证")
    print("=" * 60)
    print()

    # 生成测试数据
    print("生成测试数据...")
    df = generate_test_data(1000)

    # 计算 Python 指标
    print("计算 Python 指标...")
    py_df = calculate_python_indicators(df, fast=12, slow=26)

    # 模拟 TradingView 导出 (实际使用时替换为真实数据)
    print("加载 TradingView 数据 (模拟)...")
    tv_df = simulate_tradingview_export(df, fast=12, slow=26)

    # 验证指标
    print()
    print("验证指标计算...")
    indicator_validator = IndicatorValidator(tolerance=1e-10)

    indicator_validator.validate_series(
        "EMA(12)",
        tv_df["tv_ema_fast"],
        py_df["ema_fast"],
    )

    indicator_validator.validate_series(
        "EMA(26)",
        tv_df["tv_ema_slow"],
        py_df["ema_slow"],
    )

    print(indicator_validator.generate_report())

    # 验证信号
    print()
    print("验证交易信号...")
    signal_validator = SignalValidator(accuracy_threshold=1.0)

    signal_validator.validate_signals(
        "EMA Cross",
        tv_df["tv_long_signal"],
        tv_df["tv_short_signal"],
        py_df["long_signal"],
        py_df["short_signal"],
    )

    print(signal_validator.generate_report())


def run_validation_with_real_data(tv_csv_path: str):
    """
    使用真实 TradingView 导出数据进行验证

    Args:
        tv_csv_path: TradingView 导出的 CSV 文件路径

    CSV 格式要求:
    - timestamp, open, high, low, close, volume
    - ema_fast, ema_slow (或其他指标列)
    - long_signal, short_signal (可选)
    """
    print(f"加载 TradingView 数据: {tv_csv_path}")
    tv_df = pd.read_csv(tv_csv_path)

    # 验证必需列
    required_cols = ["close"]
    for col in required_cols:
        if col not in tv_df.columns:
            raise ValueError(f"缺少必需列: {col}")

    # 计算 Python 指标
    py_df = calculate_python_indicators(tv_df, fast=12, slow=26)

    # 验证指标 (如果 TradingView 数据包含指标列)
    indicator_validator = IndicatorValidator(tolerance=0.01)

    if "ema_fast" in tv_df.columns:
        indicator_validator.validate_series(
            "EMA(12)",
            tv_df["ema_fast"],
            py_df["ema_fast"],
        )

    if "ema_slow" in tv_df.columns:
        indicator_validator.validate_series(
            "EMA(26)",
            tv_df["ema_slow"],
            py_df["ema_slow"],
        )

    if indicator_validator.results:
        print(indicator_validator.generate_report())

    # 验证信号 (如果 TradingView 数据包含信号列)
    if "long_signal" in tv_df.columns and "short_signal" in tv_df.columns:
        signal_validator = SignalValidator(accuracy_threshold=0.95)
        signal_validator.validate_signals(
            "EMA Cross",
            tv_df["long_signal"],
            tv_df["short_signal"],
            py_df["long_signal"],
            py_df["short_signal"],
        )
        print(signal_validator.generate_report())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TV2PY 验证工具")
    parser.add_argument(
        "--tv-data",
        type=str,
        help="TradingView 导出的 CSV 文件路径",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示验证 (使用模拟数据)",
    )

    args = parser.parse_args()

    if args.tv_data:
        run_validation_with_real_data(args.tv_data)
    else:
        # 默认运行演示
        run_ema_cross_validation()
