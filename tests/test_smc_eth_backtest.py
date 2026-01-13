"""
Smart Money Concepts - ETH Backtest with Yahoo Finance Data

下载 ETH 数据并运行 SMC 分析
"""

import sys
sys.path.insert(0, '/home/user/TV2PY')

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

from strategies.smart_money_concepts import (
    SmartMoneyConcepts,
    StructureType,
    OrderBlockType,
    FVGType,
)


def generate_eth_like_data(n_bars: int = 1825, interval: str = "4h") -> pd.DataFrame:
    """
    生成类似 ETH 的模拟数据 (5年 = 1825天, 或 4H = ~10950 bars)

    模拟 ETH 的历史特征:
    - 2020: ~$130 -> ~$750 (牛市起步)
    - 2021: ~$750 -> ~$4800 (牛市高峰)
    - 2022: ~$4800 -> ~$1000 (熊市)
    - 2023: ~$1000 -> ~$2500 (震荡恢复)
    - 2024: ~$2500 -> ~$4000 (牛市)
    """
    print(f"Generating ETH-like simulated data ({n_bars} bars, {interval})...")

    np.random.seed(42)  # 可重复性

    # 时间范围
    if interval == "4h":
        n_bars = n_bars * 6  # 每天 6 个 4H K线
        freq = "4h"
    else:
        freq = "1D"

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq=freq)

    # 基础价格趋势 (模拟 ETH 周期)
    t = np.linspace(0, 1, n_bars)

    # 多周期叠加
    # 1. 长期趋势 (牛熊周期)
    bull_bear = np.zeros(n_bars)
    phase1 = int(n_bars * 0.3)   # 2020-2021.5: 上涨
    phase2 = int(n_bars * 0.55)  # 2021.5-2022: 下跌
    phase3 = int(n_bars * 0.75)  # 2022-2023: 底部
    phase4 = n_bars              # 2023-2024: 恢复

    bull_bear[:phase1] = np.linspace(0, 3, phase1)           # 130 -> 4800
    bull_bear[phase1:phase2] = np.linspace(3, 1.5, phase2 - phase1)  # 4800 -> 1000
    bull_bear[phase2:phase3] = np.linspace(1.5, 1.8, phase3 - phase2)  # 震荡
    bull_bear[phase3:] = np.linspace(1.8, 2.5, phase4 - phase3)  # 恢复

    # 2. 中期波动 (周期性)
    medium_cycle = 0.3 * np.sin(t * 20 * np.pi)

    # 3. 短期噪声 (随机游走)
    noise = np.cumsum(np.random.randn(n_bars) * 0.02)
    noise = noise - np.linspace(noise[0], noise[-1], n_bars)  # 去趋势

    # 组合
    log_price = np.log(130) + bull_bear + medium_cycle + noise
    close = np.exp(log_price)

    # 生成 OHLC
    daily_volatility = 0.03 + 0.02 * np.abs(np.sin(t * 10 * np.pi))  # 波动率变化
    daily_range = close * daily_volatility

    high = close + np.abs(np.random.randn(n_bars)) * daily_range * 0.6
    low = close - np.abs(np.random.randn(n_bars)) * daily_range * 0.6
    open_ = np.roll(close, 1) + np.random.randn(n_bars) * daily_range * 0.2
    open_[0] = close[0]

    # 确保 OHLC 关系正确
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # 成交量 (价格上涨时增加)
    base_volume = 10_000_000_000  # 100亿
    price_change = np.diff(close, prepend=close[0])
    volume = base_volume + np.abs(price_change / close) * base_volume * 50
    volume += np.abs(np.random.randn(n_bars)) * base_volume * 0.3

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    df.index.name = "Date"

    print(f"Generated {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    return df


def download_eth_data(period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    下载 ETH 数据 (直接使用 Yahoo Finance API)
    如果网络不可用，使用模拟数据
    """
    print(f"Attempting to download ETH-USD data ({period}, {interval})...")

    try:
        # 计算时间范围
        end_time = int(time.time())
        if period == "5y":
            start_time = end_time - 5 * 365 * 24 * 60 * 60
        elif period == "2y":
            start_time = end_time - 2 * 365 * 24 * 60 * 60
        elif period == "1y":
            start_time = end_time - 365 * 24 * 60 * 60
        else:
            start_time = end_time - 5 * 365 * 24 * 60 * 60

        # Yahoo Finance API URL
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/ETH-USD"
        params = {
            "period1": start_time,
            "period2": end_time,
            "interval": interval,
            "events": "history",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        if "chart" not in data or "result" not in data["chart"]:
            raise ValueError(f"Invalid response")

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        df = pd.DataFrame({
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        }, index=pd.to_datetime(timestamps, unit='s'))

        df.index.name = "Date"

        print(f"Downloaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        return df

    except Exception as e:
        print(f"Network error: {e}")
        print("Using simulated ETH-like data instead...")
        return generate_eth_like_data(n_bars=1825, interval=interval)


def run_smc_analysis(df: pd.DataFrame) -> dict:
    """运行 SMC 分析"""
    print("\n" + "=" * 70)
    print("Running Smart Money Concepts Analysis on ETH-USD")
    print("=" * 70)

    # 准备数据
    open_ = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values

    # 运行 SMC
    smc = SmartMoneyConcepts(
        swing_length=10,  # 日线用更大的周期
        ob_lookback=50,
        fvg_filter=0.5,   # 0.5% 最小缺口
    )

    result = smc.calculate(open_, high, low, close, volume)

    return {
        'smc': smc,
        'result': result,
        'df': df,
        'close': close,
    }


def print_analysis_summary(analysis: dict):
    """打印分析摘要"""
    result = analysis['result']
    df = analysis['df']
    close = analysis['close']
    high = df['High'].values
    low = df['Low'].values

    n = len(close)

    print(f"\n数据范围 / Data Range:")
    print(f"  开始: {df.index[0]}")
    print(f"  结束: {df.index[-1]}")
    print(f"  总K线数: {n}")
    print(f"  价格范围: ${low.min():.2f} - ${high.max():.2f}")

    # Swing Points
    swing_high_count = np.sum(~np.isnan(result.swing_highs))
    swing_low_count = np.sum(~np.isnan(result.swing_lows))
    print(f"\nSwing Points 摆动点:")
    print(f"  Swing Highs: {swing_high_count}")
    print(f"  Swing Lows: {swing_low_count}")

    # Order Blocks
    print(f"\nOrder Blocks 订单块:")
    print(f"  总数: {len(result.order_blocks)}")
    bullish_ob = [ob for ob in result.order_blocks if ob.type == OrderBlockType.BULLISH]
    bearish_ob = [ob for ob in result.order_blocks if ob.type == OrderBlockType.BEARISH]
    print(f"  看涨 OB: {len(bullish_ob)} (活跃: {len(result.active_bullish_ob)})")
    print(f"  看跌 OB: {len(bearish_ob)} (活跃: {len(result.active_bearish_ob)})")

    # 显示最近的活跃 OB
    if result.active_bullish_ob:
        print(f"\n  最近的看涨订单块 / Recent Bullish OBs:")
        for ob in result.active_bullish_ob[-5:]:
            date = df.index[ob.start_index] if ob.start_index < len(df) else "N/A"
            print(f"    [{date}] ${ob.bottom:.2f} - ${ob.top:.2f}")

    if result.active_bearish_ob:
        print(f"\n  最近的看跌订单块 / Recent Bearish OBs:")
        for ob in result.active_bearish_ob[-5:]:
            date = df.index[ob.start_index] if ob.start_index < len(df) else "N/A"
            print(f"    [{date}] ${ob.bottom:.2f} - ${ob.top:.2f}")

    # Fair Value Gaps
    print(f"\nFair Value Gaps 公允价值缺口:")
    print(f"  总数: {len(result.fvg_list)}")
    bullish_fvg = [fvg for fvg in result.fvg_list if fvg.type == FVGType.BULLISH]
    bearish_fvg = [fvg for fvg in result.fvg_list if fvg.type == FVGType.BEARISH]
    print(f"  看涨 FVG: {len(bullish_fvg)} (活跃: {len(result.active_bullish_fvg)})")
    print(f"  看跌 FVG: {len(bearish_fvg)} (活跃: {len(result.active_bearish_fvg)})")

    # 显示最近的活跃 FVG
    if result.active_bullish_fvg:
        print(f"\n  最近的看涨 FVG / Recent Bullish FVGs:")
        for fvg in result.active_bullish_fvg[-5:]:
            date = df.index[fvg.index] if fvg.index < len(df) else "N/A"
            print(f"    [{date}] ${fvg.bottom:.2f} - ${fvg.top:.2f} (填补: {fvg.fill_percent*100:.1f}%)")

    if result.active_bearish_fvg:
        print(f"\n  最近的看跌 FVG / Recent Bearish FVGs:")
        for fvg in result.active_bearish_fvg[-5:]:
            date = df.index[fvg.index] if fvg.index < len(df) else "N/A"
            print(f"    [{date}] ${fvg.bottom:.2f} - ${fvg.top:.2f} (填补: {fvg.fill_percent*100:.1f}%)")

    # Structure Breaks
    print(f"\nStructure Breaks 结构突破:")
    bos_list = [sb for sb in result.structure_breaks if sb.type == StructureType.BOS]
    choch_list = [sb for sb in result.structure_breaks if sb.type == StructureType.CHOCH]
    print(f"  BOS (趋势延续): {len(bos_list)}")
    print(f"  CHoCH (趋势反转): {len(choch_list)}")

    # 最近的结构变化
    print(f"\n  最近的结构变化 / Recent Structure Changes:")
    for sb in result.structure_breaks[-10:]:
        date = df.index[sb.index] if sb.index < len(df) else "N/A"
        direction = "↑ Bullish" if sb.direction == 1 else "↓ Bearish"
        print(f"    [{date}] {sb.type.value.upper()} {direction} @ ${sb.price:.2f}")

    # Liquidity Zones
    print(f"\nLiquidity Zones 流动性区域:")
    buy_side = [z for z in result.liquidity_zones if z.type == 'buy_side']
    sell_side = [z for z in result.liquidity_zones if z.type == 'sell_side']
    print(f"  Buy Side (Equal Highs): {len(buy_side)}")
    print(f"  Sell Side (Equal Lows): {len(sell_side)}")

    # Current State
    print(f"\n" + "=" * 70)
    print(f"当前状态 / Current State (最新K线)")
    print("=" * 70)

    current_price = close[-1]
    current_trend = result.trend[-1]
    trend_str = "↑ Bullish 看涨" if current_trend == 1 else "↓ Bearish 看跌" if current_trend == -1 else "→ Neutral 中性"

    print(f"  当前价格 / Current Price: ${current_price:.2f}")
    print(f"  当前趋势 / Current Trend: {trend_str}")
    print(f"  Equilibrium: ${result.equilibrium[-1]:.2f}")
    print(f"  Premium Zone (溢价区): > ${result.premium_zone[-1]:.2f}")
    print(f"  Discount Zone (折价区): < ${result.discount_zone[-1]:.2f}")

    # 判断当前位置
    if current_price > result.premium_zone[-1]:
        zone = "🔴 Premium Zone (溢价区 - 考虑卖出)"
    elif current_price < result.discount_zone[-1]:
        zone = "🟢 Discount Zone (折价区 - 考虑买入)"
    else:
        zone = "🟡 Equilibrium Zone (平衡区)"
    print(f"  当前区域 / Current Zone: {zone}")

    # 交易信号
    smc = analysis['smc']
    signals = smc.get_signals(result, close, n - 1)

    print(f"\n交易信号 / Trading Signals:")
    if signals['near_bullish_ob']:
        print(f"  🟢 接近看涨订单块 - 潜在支撑")
    if signals['near_bearish_ob']:
        print(f"  🔴 接近看跌订单块 - 潜在阻力")
    if signals['near_bullish_fvg']:
        print(f"  🟢 在看涨 FVG 区域 - 潜在支撑")
    if signals['near_bearish_fvg']:
        print(f"  🔴 在看跌 FVG 区域 - 潜在阻力")

    if signals['recent_choch']:
        choch = signals['recent_choch']
        direction = "Bullish" if choch.direction == 1 else "Bearish"
        print(f"  ⚠️ 最近 CHoCH: {direction} @ ${choch.price:.2f}")

    if signals['recent_bos']:
        bos = signals['recent_bos']
        direction = "Bullish" if bos.direction == 1 else "Bearish"
        print(f"  ℹ️ 最近 BOS: {direction} @ ${bos.price:.2f}")


def run_simple_backtest(analysis: dict):
    """运行简单回测"""
    print(f"\n" + "=" * 70)
    print("简单回测 / Simple Backtest")
    print("=" * 70)

    result = analysis['result']
    df = analysis['df']
    close = analysis['close']
    smc = analysis['smc']

    n = len(close)

    # 策略: 在折价区买入看涨 OB/FVG，在溢价区卖出看跌 OB/FVG
    trades = []
    position = 0  # 0=无持仓, 1=多头
    entry_price = 0
    entry_date = None

    for i in range(100, n):  # 跳过预热期
        signals = smc.get_signals(result, close, i)
        current_price = close[i]
        current_date = df.index[i]

        # 入场条件
        if position == 0:
            # 多头入场: 折价区 + 看涨信号
            if (signals['in_discount'] and
                (signals['near_bullish_ob'] or signals['near_bullish_fvg']) and
                signals['trend'] >= 0):
                position = 1
                entry_price = current_price
                entry_date = current_date

        # 出场条件
        elif position == 1:
            # 止损: 跌破 3%
            if current_price < entry_price * 0.97:
                pnl = (current_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'type': 'stop_loss'
                })
                position = 0

            # 止盈: 溢价区 + 看跌信号
            elif (signals['in_premium'] and
                  (signals['near_bearish_ob'] or signals['near_bearish_fvg'])):
                pnl = (current_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'type': 'take_profit'
                })
                position = 0

            # 趋势反转出场
            elif signals['recent_choch'] and signals['recent_choch'].direction == -1:
                pnl = (current_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'type': 'choch_exit'
                })
                position = 0

    # 统计结果
    if trades:
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(winning_trades) / total_trades * 100

        total_pnl = sum(t['pnl'] for t in trades)
        avg_pnl = total_pnl / total_trades

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        print(f"\n回测结果 / Backtest Results:")
        print(f"  总交易次数 / Total Trades: {total_trades}")
        print(f"  盈利交易 / Winning Trades: {len(winning_trades)}")
        print(f"  亏损交易 / Losing Trades: {len(losing_trades)}")
        print(f"  胜率 / Win Rate: {win_rate:.1f}%")
        print(f"  总收益 / Total PnL: {total_pnl:.2f}%")
        print(f"  平均收益 / Avg PnL: {avg_pnl:.2f}%")
        print(f"  平均盈利 / Avg Win: {avg_win:.2f}%")
        print(f"  平均亏损 / Avg Loss: {avg_loss:.2f}%")

        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            print(f"  盈亏比 / Profit Factor: {profit_factor:.2f}")

        # 按类型统计
        print(f"\n出场类型统计 / Exit Type Stats:")
        exit_types = {}
        for t in trades:
            exit_types[t['type']] = exit_types.get(t['type'], 0) + 1
        for etype, count in exit_types.items():
            print(f"    {etype}: {count}")

        # 显示最近交易
        print(f"\n最近交易 / Recent Trades:")
        for t in trades[-10:]:
            emoji = "✅" if t['pnl'] > 0 else "❌"
            print(f"  {emoji} [{t['entry_date'].strftime('%Y-%m-%d')}] "
                  f"${t['entry_price']:.2f} → ${t['exit_price']:.2f} "
                  f"({t['pnl']:+.2f}%) [{t['type']}]")
    else:
        print("  没有产生交易信号")


def main():
    print("=" * 70)
    print("Smart Money Concepts - ETH-USD Backtest")
    print("=" * 70)

    # 下载数据 (5年日线)
    # Note: Yahoo Finance 对 4H 数据有限制，使用日线代替
    df = download_eth_data(period="5y", interval="1d")

    # 过滤无效数据
    df = df.dropna()

    # 运行分析
    analysis = run_smc_analysis(df)

    # 打印摘要
    print_analysis_summary(analysis)

    # 运行回测
    run_simple_backtest(analysis)

    print("\n" + "=" * 70)
    print("分析完成 / Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
