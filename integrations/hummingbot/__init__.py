"""
TV2PY Hummingbot Integration

将 TradingView 指标与 Hummingbot V2 Framework 整合

Hummingbot 是一个开源的加密货币做市和算法交易框架。
本模块提供了将 TV2PY 指标用于 Hummingbot 策略的适配器。

使用方式:
    1. 将此目录复制到 Hummingbot 项目中
    2. 在 Hummingbot 脚本中导入指标适配器
    3. 使用信号生成器获取交易信号

示例:
    from integrations.hummingbot import TV2PYSignalGenerator

    signal_gen = TV2PYSignalGenerator()
    signal = signal_gen.get_signal(candles_df)

参考文档:
    - Hummingbot V2 Framework: https://hummingbot.org/strategies/
    - Hummingbot Scripts: https://hummingbot.org/client/start-stop/
"""

from .indicator_adapter import (
    IndicatorAdapter,
    CandleData,
    IndicatorResult,
)

from .signal_generator import (
    TV2PYSignalGenerator,
    SignalType,
    TradingSignal,
)

__all__ = [
    "IndicatorAdapter",
    "CandleData",
    "IndicatorResult",
    "TV2PYSignalGenerator",
    "SignalType",
    "TradingSignal",
]

__version__ = "1.0.0"
