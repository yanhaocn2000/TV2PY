"""
TPXSlingShotSystem v1.1 - Python/PyneCore 版本

原始 Pine Script 策略转换为 Python，使用 PyneCore 确保与 TradingView 对齐

@pyne
"""
from pynecore.lib import (
    script, input, close, high, low, open, volume, hl2,
    ta, math, bar_index, strategy, color, NA
)
from pynecore.types import Series, Persistent
from typing import Tuple

# =============================================================================
# 策略配置
# =============================================================================

@script.strategy(
    title="TPX SlingShot System v1.1",
    overlay=True,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
    pyramiding=0
)
def main(
    # TPX 设置
    tpx_length: int = input.int("TPX Length", 14, minval=1),
    tpx_smooth: int = input.int("TPX Smoothing (WMA)", 5, minval=1),

    # EMA 设置
    ema_fast: int = input.int("Fast EMA", 55),
    ema_medium: int = input.int("Medium EMA", 89),
    ema_slow: int = input.int("Slow EMA", 200),

    # T3 设置
    t3_length: int = input.int("T3 Length", 8),
    t3_factor: float = input.float("T3 Volume Factor", 0.7),

    # Hull MA 设置
    hma_length: int = input.int("Hull MA Length", 21),

    # 风险管理
    atr_length: int = input.int("ATR Length", 14),
    atr_mult: float = input.float("ATR Multiplier for SL", 2.0),
    trail_atr_mult: float = input.float("Trailing TP ATR Mult", 3.0),

    # 过滤器
    use_rsi_filter: bool = input.bool("Use RSI Filter", True),
    rsi_length: int = input.int("RSI Length", 14),
    rsi_oversold: int = input.int("RSI Oversold", 30),
    rsi_overbought: int = input.int("RSI Overbought", 70),
):
    """
    TPX SlingShot 策略主函数

    交易逻辑:
    - TPX 向上穿越零线 + 价格在趋势均线上方 = 做多
    - TPX 向下穿越零线 + 价格在趋势均线下方 = 做空
    - ATR 止损
    - 追踪止盈
    """

    # =========================================================================
    # 1. TPX 指标计算 (Trading Pressure Index)
    # =========================================================================
    tpx_value = calculate_tpx(close, high, low, tpx_length, tpx_smooth)

    # =========================================================================
    # 2. 移动平均线
    # =========================================================================
    ema_55 = ta.ema(close, ema_fast)
    ema_89 = ta.ema(close, ema_medium)
    ema_200 = ta.ema(close, ema_slow)

    # Tilson T3
    t3_value = calculate_t3(close, t3_length, t3_factor)

    # Hull MA
    hull_ma = ta.hma(close, hma_length)

    # =========================================================================
    # 3. RSI 过滤器
    # =========================================================================
    rsi = ta.rsi(close, rsi_length)

    # =========================================================================
    # 4. ATR 用于止损/止盈
    # =========================================================================
    atr_value = ta.atr(atr_length)

    # =========================================================================
    # 5. 趋势判断
    # =========================================================================
    # 多头趋势: 价格 > EMA200 且 EMA55 > EMA89
    uptrend = close > ema_200 and ema_55 > ema_89

    # 空头趋势: 价格 < EMA200 且 EMA55 < EMA89
    downtrend = close < ema_200 and ema_55 < ema_89

    # =========================================================================
    # 6. 入场信号
    # =========================================================================
    # TPX 穿越零线
    tpx_cross_up = ta.crossover(tpx_value, 0)
    tpx_cross_down = ta.crossunder(tpx_value, 0)

    # RSI 过滤
    rsi_long_ok = (not use_rsi_filter) or (rsi < rsi_overbought)
    rsi_short_ok = (not use_rsi_filter) or (rsi > rsi_oversold)

    # 价格在 T3/Hull 之上/之下
    price_above_t3 = close > t3_value
    price_below_t3 = close < t3_value

    # 完整入场条件
    long_condition = (
        tpx_cross_up and
        uptrend and
        price_above_t3 and
        rsi_long_ok
    )

    short_condition = (
        tpx_cross_down and
        downtrend and
        price_below_t3 and
        rsi_short_ok
    )

    # =========================================================================
    # 7. 交易执行
    # =========================================================================
    if long_condition:
        stop_loss = close - atr_value * atr_mult
        take_profit = close + atr_value * trail_atr_mult
        strategy.entry("Long", strategy.long)
        strategy.exit("Long Exit", "Long", stop=stop_loss, limit=take_profit)

    if short_condition:
        stop_loss = close + atr_value * atr_mult
        take_profit = close - atr_value * trail_atr_mult
        strategy.entry("Short", strategy.short)
        strategy.exit("Short Exit", "Short", stop=stop_loss, limit=take_profit)

    # 返回指标值供可视化/调试
    return {
        "tpx": tpx_value,
        "ema_55": ema_55,
        "ema_89": ema_89,
        "ema_200": ema_200,
        "t3": t3_value,
        "hull_ma": hull_ma,
        "rsi": rsi,
        "atr": atr_value,
    }


# =============================================================================
# 自定义指标函数
# =============================================================================

def calculate_tpx(
    src: Series[float],
    high_src: Series[float],
    low_src: Series[float],
    length: int,
    smooth: int
) -> float:
    """
    TPX (Trading Pressure Index) 指标

    测量买卖压力的强度

    公式:
    - 范围 = highest(high, length) - lowest(low, length)
    - 买压 = close - lowest(low, length)
    - 卖压 = highest(high, length) - close
    - 原始 TPX = (买压 - 卖压) / 范围 * 100
    - TPX = WMA(原始TPX, smooth)
    """
    highest_high = ta.highest(high_src, length)
    lowest_low = ta.lowest(low_src, length)

    range_val = highest_high - lowest_low

    if isinstance(range_val, NA) or range_val == 0:
        return NA(float)

    buying_pressure = src - lowest_low
    selling_pressure = highest_high - src

    raw_tpx = (buying_pressure - selling_pressure) / range_val * 100

    # WMA 平滑
    tpx_smoothed = ta.wma(raw_tpx, smooth)

    return tpx_smoothed


def calculate_t3(src: Series[float], length: int, factor: float) -> float:
    """
    Tilson T3 移动平均线

    使用 6 层 EMA 和体积因子创建超平滑的移动平均

    公式:
    - e1 = EMA(src, length)
    - e2 = EMA(e1, length)
    - ...
    - e6 = EMA(e5, length)
    - c1 = -factor^3
    - c2 = 3*factor^2 + 3*factor^3
    - c3 = -6*factor^2 - 3*factor - 3*factor^3
    - c4 = 1 + 3*factor + factor^3 + 3*factor^2
    - T3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
    """
    e1 = ta.ema(src, length)
    e2 = ta.ema(e1, length)
    e3 = ta.ema(e2, length)
    e4 = ta.ema(e3, length)
    e5 = ta.ema(e4, length)
    e6 = ta.ema(e5, length)

    # T3 系数
    c1 = -(factor ** 3)
    c2 = 3 * (factor ** 2) + 3 * (factor ** 3)
    c3 = -6 * (factor ** 2) - 3 * factor - 3 * (factor ** 3)
    c4 = 1 + 3 * factor + (factor ** 3) + 3 * (factor ** 2)

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    return t3


# =============================================================================
# 多时间框架辅助函数 (用于 NautilusTrader 回测)
# =============================================================================

def prepare_mtf_data(data: dict, higher_tf: str = "1D") -> dict:
    """
    为多时间框架分析准备数据

    在 NautilusTrader 中，我们需要预先重采样数据
    而不是使用 request.security

    Args:
        data: 原始 OHLCV 数据 (字典或 DataFrame)
        higher_tf: 高级时间框架 (如 "1D", "4H")

    Returns:
        包含多时间框架数据的字典
    """
    # 这个函数将在回测框架中实现
    # 用于替代 Pine Script 的 request.security
    pass


# =============================================================================
# 用于独立测试的代码
# =============================================================================

if __name__ == "__main__":
    print("TPX SlingShot System v1.1")
    print("=" * 50)
    print("策略已转换为 Python/PyneCore 格式")
    print()
    print("支持的指标:")
    print("  - TPX (Trading Pressure Index)")
    print("  - EMA (55, 89, 200)")
    print("  - Tilson T3")
    print("  - Hull MA")
    print("  - RSI")
    print("  - ATR")
    print()
    print("使用方法:")
    print("  pyne run strategies/tpx_slingshot.py data/ETHUSDT_4h.ohlcv")
