"""
TV2PY Validation Framework

用于验证 TradingView Pine Script 到 Python 的转换是否正确对齐。

主要功能:
1. IndicatorValidator - 验证指标计算 (EMA, SMA, RSI, MACD, etc.)
2. SignalValidator - 验证交易信号 (入场/出场信号)
3. BacktestValidator - 验证回测结果 (收益、胜率、回撤)

使用方法:
1. 在 TradingView 中运行策略并导出数据 (图表数据 + 指标值)
2. 将导出的 CSV 放入 validation/data/ 目录
3. 运行 python validation/run_validation.py --tv-data <csv_path>
"""

from validation.validators import (
    IndicatorValidator,
    SignalValidator,
    BacktestValidator,
)

__all__ = [
    "IndicatorValidator",
    "SignalValidator",
    "BacktestValidator",
]
