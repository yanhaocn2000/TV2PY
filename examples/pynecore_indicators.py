"""
@pyne

使用 PyneCore 的指标示例 - 与 TradingView 100% 对齐

PyneCore 已经验证过与 TradingView 的对齐 (0.001% 容差)
直接使用即可，无需自己实现底层算法
"""

from pynecore import Series, Persistent
from pynecore.lib import (
    script,
    close, high, low, open, volume, hl2, hlc3, ohlc4,
    ta, plot, color, input,
    strategy
)


@script.indicator(title="TV2PY 多指标面板", overlay=False)
def main(
    # 输入参数 - 与 TradingView input 完全一致
    ema_fast: int = input.int("EMA Fast", 12, minval=1),
    ema_slow: int = input.int("EMA Slow", 26, minval=1),
    rsi_length: int = input.int("RSI Length", 14, minval=1),
    bb_length: int = input.int("BB Length", 20, minval=1),
    bb_mult: float = input.float("BB Mult", 2.0, minval=0.1),
):
    """
    多指标组合示例

    所有指标都使用 PyneCore 的 ta 模块
    与 TradingView 的 ta.* 函数完全对齐
    """

    # ==================== 移动平均 ====================
    # ta.ema - 与 TradingView ta.ema 完全一致
    fast_ema = ta.ema(close, ema_fast)
    slow_ema = ta.ema(close, ema_slow)

    # ta.sma - 简单移动平均
    sma_20 = ta.sma(close, 20)

    # ta.rma - Wilder 移动平均 (RSI 使用的)
    rma_14 = ta.rma(close, 14)

    # ta.wma - 加权移动平均
    wma_10 = ta.wma(close, 10)

    # ta.vwma - 成交量加权移动平均
    vwma_20 = ta.vwma(close, 20)

    # ==================== 动量指标 ====================
    # ta.rsi - 相对强弱指数
    rsi = ta.rsi(close, rsi_length)

    # ta.macd - MACD
    macd_line, signal_line, histogram = ta.macd(close, 12, 26, 9)

    # ta.stoch - 随机指标
    stoch_k = ta.stoch(close, high, low, 14)
    stoch_d = ta.sma(stoch_k, 3)

    # ta.cci - 商品通道指数
    cci = ta.cci(high, low, close, 20)

    # ta.mom - 动量
    momentum = ta.mom(close, 10)

    # ta.roc - 变化率
    roc = ta.roc(close, 10)

    # ==================== 波动率指标 ====================
    # ta.atr - 平均真实波幅
    atr = ta.atr(14)

    # ta.bb - 布林带
    bb_middle, bb_upper, bb_lower = ta.bb(close, bb_length, bb_mult)

    # ta.kc - 肯特纳通道
    kc_middle, kc_upper, kc_lower = ta.kc(close, 20, 1.5)

    # ==================== 趋势指标 ====================
    # ta.adx - 平均趋向指数
    plus_di, minus_di, adx = ta.dmi(14, 14)

    # ta.supertrend - 超级趋势
    supertrend, direction = ta.supertrend(3, 10)

    # ta.sar - 抛物线 SAR
    sar = ta.sar(0.02, 0.02, 0.2)

    # ==================== 交叉信号 ====================
    # ta.crossover / ta.crossunder - 与 TradingView 一致
    golden_cross = ta.crossover(fast_ema, slow_ema)
    death_cross = ta.crossunder(fast_ema, slow_ema)

    # ==================== 输出 ====================
    plot(rsi, "RSI", color=color.purple)
    plot(50, "RSI 中线", color=color.gray)

    return {
        "ema_fast": fast_ema,
        "ema_slow": slow_ema,
        "rsi": rsi,
        "macd": macd_line,
        "signal": signal_line,
        "atr": atr,
        "adx": adx,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
    }


# ==================== 策略示例 ====================
@script.strategy(title="EMA Cross Strategy", overlay=True)
def ema_cross_strategy(
    fast_length: int = input.int("Fast Length", 12),
    slow_length: int = input.int("Slow Length", 26),
):
    """
    EMA 交叉策略 - 与 TradingView 策略逻辑一致
    """
    fast_ema = ta.ema(close, fast_length)
    slow_ema = ta.ema(close, slow_length)

    # 交叉信号
    long_signal = ta.crossover(fast_ema, slow_ema)
    short_signal = ta.crossunder(fast_ema, slow_ema)

    # 交易逻辑 - 与 TradingView strategy.* 一致
    if long_signal:
        strategy.entry("Long", strategy.long)

    if short_signal:
        strategy.close("Long")

    # 绘制
    plot(fast_ema, "Fast EMA", color=color.blue)
    plot(slow_ema, "Slow EMA", color=color.red)


if __name__ == "__main__":
    print("""
PyneCore 指标示例

使用方法:
1. 下载数据
   pyne data download ccxt --symbol "BINANCE:ETH/USDT"

2. 运行指标
   pyne run examples/pynecore_indicators.py data.ohlcv

3. 运行策略回测
   pyne run examples/pynecore_indicators.py data.ohlcv --strategy

所有指标都与 TradingView 对齐，无需担心计算差异！
""")
