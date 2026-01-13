"""
@pyne

RSI 超买超卖策略 - 完整示例

这个策略展示如何:
1. 使用 PyneCore 的 ta.rsi (与 TradingView 对齐)
2. 设置策略参数
3. 执行交易逻辑
4. 输出到图表
"""

from pynecore import Series, Persistent
from pynecore.lib import (
    script,
    close, high, low,
    ta, plot, hline, fill, color, input,
    strategy
)


@script.strategy(
    title="RSI Strategy",
    overlay=False,
    default_qty_type=strategy.percent_of_equity,
    default_qty_value=100,
)
def main(
    # RSI 参数
    rsi_length: int = input.int("RSI Length", 14, minval=1, maxval=50),
    rsi_overbought: int = input.int("Overbought", 70, minval=50, maxval=100),
    rsi_oversold: int = input.int("Oversold", 30, minval=0, maxval=50),

    # 止损止盈
    use_sl: bool = input.bool("Use Stop Loss", True),
    sl_percent: float = input.float("Stop Loss %", 2.0, minval=0.1, maxval=10.0),
    use_tp: bool = input.bool("Use Take Profit", True),
    tp_percent: float = input.float("Take Profit %", 4.0, minval=0.1, maxval=20.0),
):
    """
    RSI 超买超卖策略

    入场:
    - RSI < oversold (超卖) → 做多
    - RSI > overbought (超买) → 平仓

    出场:
    - 止损/止盈
    - 反向信号
    """

    # ==================== 计算指标 ====================
    # ta.rsi 与 TradingView 完全一致
    rsi = ta.rsi(close, rsi_length)

    # ==================== 交易信号 ====================
    # 使用 Persistent 变量记录状态
    in_position: Persistent[bool] = False

    # 超卖买入信号
    long_signal = ta.crossover(rsi, rsi_oversold) if not in_position else False

    # 超买卖出信号
    exit_signal = ta.crossunder(rsi, rsi_overbought) if in_position else False

    # ==================== 执行交易 ====================
    if long_signal:
        strategy.entry("Long", strategy.long)
        in_position = True

        # 设置止损止盈
        if use_sl:
            strategy.exit(
                "SL/TP",
                from_entry="Long",
                stop=close * (1 - sl_percent / 100) if use_sl else None,
                limit=close * (1 + tp_percent / 100) if use_tp else None,
            )

    if exit_signal:
        strategy.close("Long")
        in_position = False

    # ==================== 图表输出 ====================
    # RSI 线
    plot(rsi, "RSI", color=color.purple, linewidth=2)

    # 超买超卖水平线
    h_overbought = hline(rsi_overbought, "Overbought", color=color.red, linestyle=hline.style_dashed)
    h_oversold = hline(rsi_oversold, "Oversold", color=color.green, linestyle=hline.style_dashed)
    h_middle = hline(50, "Middle", color=color.gray, linestyle=hline.style_dotted)

    # 填充超买超卖区域
    fill(h_overbought, h_middle, color=color.new(color.red, 90))
    fill(h_oversold, h_middle, color=color.new(color.green, 90))

    # 信号标记
    # plotshape(long_signal, "Buy", shape.triangleup, location.belowbar, color.green)
    # plotshape(exit_signal, "Sell", shape.triangledown, location.abovebar, color.red)


if __name__ == "__main__":
    print("""
RSI 策略使用说明
================

1. 安装 PyneCore:
   pip install pynesys-pynecore[cli]

2. 下载数据:
   pyne data download ccxt --symbol "BINANCE:ETH/USDT" --timeframe 4h

3. 运行策略回测:
   pyne run examples/rsi_strategy.py ccxt_BINANCE_ETH_USDT_4h.ohlcv

4. 查看结果:
   策略会输出交易记录、收益曲线等

核心优势:
- ta.rsi 与 TradingView 100% 对齐
- 回测结果可与 TradingView 对比验证
- 无需担心指标计算差异
""")
