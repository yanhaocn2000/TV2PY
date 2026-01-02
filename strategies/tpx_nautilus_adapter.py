"""
TPX SlingShot 策略 - NautilusTrader 适配器

将 PyneCore 策略信号转换为 NautilusTrader 订单

这个文件展示如何将转换后的 Pine Script 策略
集成到 NautilusTrader 回测引擎中
"""
from decimal import Decimal
from typing import Optional

import pandas as pd
import numpy as np

# NautilusTrader 导入 (需要安装 nautilus_trader)
try:
    from nautilus_trader.core.datetime import dt_to_unix_nanos
    from nautilus_trader.model.identifiers import InstrumentId, TraderId, StrategyId
    from nautilus_trader.model.data import Bar, BarType
    from nautilus_trader.model.enums import OrderSide, PositionSide
    from nautilus_trader.model.orders import MarketOrder
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.config import StrategyConfig
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    print("警告: nautilus_trader 未安装，仅可使用简化回测模式")


# =============================================================================
# TradingView 对齐的指标实现 (使用 numpy 向量化)
# =============================================================================

def tv_ema(src: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView 对齐的 EMA 实现

    关键点:
    - 使用 SMA 作为种子值 (前 length 个 bar)
    - alpha = 2 / (length + 1)
    """
    alpha = 2.0 / (length + 1)
    result = np.full_like(src, np.nan, dtype=float)

    # 计算 SMA 作为种子
    for i in range(length - 1, len(src)):
        if i == length - 1:
            # 使用 SMA 作为第一个值
            result[i] = np.mean(src[:length])
        else:
            result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]

    return result


def tv_wma(src: np.ndarray, length: int) -> np.ndarray:
    """TradingView 对齐的 WMA 实现"""
    result = np.full_like(src, np.nan, dtype=float)
    weights = np.arange(1, length + 1)
    denom = weights.sum()

    for i in range(length - 1, len(src)):
        window = src[i - length + 1:i + 1]
        result[i] = np.sum(window * weights) / denom

    return result


def tv_rma(src: np.ndarray, length: int) -> np.ndarray:
    """TradingView 对齐的 RMA (Wilder's MA) 实现"""
    alpha = 1.0 / length
    result = np.full_like(src, np.nan, dtype=float)

    for i in range(length - 1, len(src)):
        if i == length - 1:
            result[i] = np.mean(src[:length])
        else:
            result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]

    return result


def tv_rsi(src: np.ndarray, length: int) -> np.ndarray:
    """TradingView 对齐的 RSI 实现"""
    delta = np.diff(src, prepend=src[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = tv_rma(gain, length)
    avg_loss = tv_rma(loss, length)

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def tv_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """TradingView 对齐的 ATR 实现"""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )

    return tv_rma(tr, length)


def tv_hma(src: np.ndarray, length: int) -> np.ndarray:
    """TradingView 对齐的 Hull MA 实现"""
    half_length = length // 2
    sqrt_length = int(np.sqrt(length))

    wma_half = tv_wma(src, half_length)
    wma_full = tv_wma(src, length)

    raw_hma = 2 * wma_half - wma_full

    return tv_wma(raw_hma, sqrt_length)


def calculate_tpx(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    length: int,
    smooth: int
) -> np.ndarray:
    """TPX 指标向量化实现"""
    result = np.full_like(close, np.nan, dtype=float)

    for i in range(length - 1, len(close)):
        # 计算区间内的最高/最低
        highest_high = np.max(high[i - length + 1:i + 1])
        lowest_low = np.min(low[i - length + 1:i + 1])

        range_val = highest_high - lowest_low
        if range_val == 0:
            result[i] = 0
        else:
            buying_pressure = close[i] - lowest_low
            selling_pressure = highest_high - close[i]
            result[i] = (buying_pressure - selling_pressure) / range_val * 100

    # WMA 平滑
    return tv_wma(result, smooth)


def calculate_t3(src: np.ndarray, length: int, factor: float) -> np.ndarray:
    """Tilson T3 向量化实现"""
    e1 = tv_ema(src, length)
    e2 = tv_ema(e1, length)
    e3 = tv_ema(e2, length)
    e4 = tv_ema(e3, length)
    e5 = tv_ema(e4, length)
    e6 = tv_ema(e5, length)

    c1 = -(factor ** 3)
    c2 = 3 * (factor ** 2) + 3 * (factor ** 3)
    c3 = -6 * (factor ** 2) - 3 * factor - 3 * (factor ** 3)
    c4 = 1 + 3 * factor + (factor ** 3) + 3 * (factor ** 2)

    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


# =============================================================================
# 简化版回测 (不依赖 NautilusTrader)
# =============================================================================

class TPXSlingShotBacktest:
    """
    TPX SlingShot 策略简化回测

    用于快速验证策略逻辑，不需要完整的 NautilusTrader 环境
    """

    def __init__(
        self,
        tpx_length: int = 14,
        tpx_smooth: int = 5,
        ema_fast: int = 55,
        ema_medium: int = 89,
        ema_slow: int = 200,
        t3_length: int = 8,
        t3_factor: float = 0.7,
        hma_length: int = 21,
        atr_length: int = 14,
        atr_mult: float = 2.0,
        trail_atr_mult: float = 3.0,
        use_rsi_filter: bool = True,
        rsi_length: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1%
    ):
        self.params = {
            'tpx_length': tpx_length,
            'tpx_smooth': tpx_smooth,
            'ema_fast': ema_fast,
            'ema_medium': ema_medium,
            'ema_slow': ema_slow,
            't3_length': t3_length,
            't3_factor': t3_factor,
            'hma_length': hma_length,
            'atr_length': atr_length,
            'atr_mult': atr_mult,
            'trail_atr_mult': trail_atr_mult,
            'use_rsi_filter': use_rsi_filter,
            'rsi_length': rsi_length,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
        }
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, df: pd.DataFrame) -> dict:
        """
        运行回测

        Args:
            df: OHLCV DataFrame，需要包含 columns: timestamp, open, high, low, close, volume

        Returns:
            回测结果字典
        """
        # 提取价格数据
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_ = df['open'].values

        n = len(close)

        # 计算指标
        tpx = calculate_tpx(
            close, high, low,
            self.params['tpx_length'],
            self.params['tpx_smooth']
        )

        ema_55 = tv_ema(close, self.params['ema_fast'])
        ema_89 = tv_ema(close, self.params['ema_medium'])
        ema_200 = tv_ema(close, self.params['ema_slow'])

        t3 = calculate_t3(close, self.params['t3_length'], self.params['t3_factor'])

        rsi = tv_rsi(close, self.params['rsi_length'])

        atr = tv_atr(high, low, close, self.params['atr_length'])

        # 趋势判断
        uptrend = (close > ema_200) & (ema_55 > ema_89)
        downtrend = (close < ema_200) & (ema_55 < ema_89)

        # TPX 穿越
        tpx_prev = np.roll(tpx, 1)
        tpx_prev[0] = np.nan
        tpx_cross_up = (tpx > 0) & (tpx_prev <= 0)
        tpx_cross_down = (tpx < 0) & (tpx_prev >= 0)

        # RSI 过滤
        if self.params['use_rsi_filter']:
            rsi_long_ok = rsi < self.params['rsi_overbought']
            rsi_short_ok = rsi > self.params['rsi_oversold']
        else:
            rsi_long_ok = np.ones(n, dtype=bool)
            rsi_short_ok = np.ones(n, dtype=bool)

        # 价格位置
        price_above_t3 = close > t3
        price_below_t3 = close < t3

        # 信号
        long_signal = tpx_cross_up & uptrend & price_above_t3 & rsi_long_ok
        short_signal = tpx_cross_down & downtrend & price_below_t3 & rsi_short_ok

        # 模拟交易
        trades = []
        position = 0  # 0: 无仓位, 1: 多头, -1: 空头
        entry_price = 0
        entry_idx = 0
        stop_loss = 0
        take_profit = 0
        highest_since_entry = 0  # 用于追踪止盈
        lowest_since_entry = float('inf')

        equity = self.initial_capital
        equity_curve = [equity]

        # 止损止盈模式
        use_trailing_tp = self.params.get('use_trailing_tp', True)  # 默认使用追踪止盈
        use_trailing_sl = self.params.get('use_trailing_sl', False)  # 默认不使用追踪止损
        trailing_activation_pct = self.params.get('trailing_activation_pct', 1.5)  # 追踪止盈激活阈值

        for i in range(1, n):
            # 更新持仓期间的最高/最低价
            if position == 1:
                highest_since_entry = max(highest_since_entry, high[i])

                # 追踪止损: 止损跟随最高价
                if use_trailing_sl and not np.isnan(atr[i]):
                    new_sl = highest_since_entry - atr[i] * self.params['atr_mult']
                    stop_loss = max(stop_loss, new_sl)

                # 追踪止盈: 价格涨幅超过阈值后激活
                if use_trailing_tp:
                    if highest_since_entry > entry_price * (1 + trailing_activation_pct / 100):
                        # 追踪止盈 = 最高价 - ATR × 倍数
                        new_tp = highest_since_entry - atr[i] * self.params['trail_atr_mult'] * 0.5
                        if take_profit == float('inf') or new_tp > take_profit:
                            take_profit = new_tp

            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, low[i])

                # 追踪止损
                if use_trailing_sl and not np.isnan(atr[i]):
                    new_sl = lowest_since_entry + atr[i] * self.params['atr_mult']
                    stop_loss = min(stop_loss, new_sl)

                # 追踪止盈
                if use_trailing_tp:
                    if lowest_since_entry < entry_price * (1 - trailing_activation_pct / 100):
                        new_tp = lowest_since_entry + atr[i] * self.params['trail_atr_mult'] * 0.5
                        if take_profit == 0 or new_tp < take_profit:
                            take_profit = new_tp

            # 检查止损/止盈
            if position == 1:  # 多头
                if low[i] <= stop_loss:
                    # 止损
                    pnl = (stop_loss - entry_price) / entry_price - self.commission
                    equity *= (1 + pnl)
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': 'long',
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'pnl_pct': pnl * 100,
                        'exit_reason': 'stop_loss'
                    })
                    position = 0
                elif take_profit != float('inf') and high[i] >= take_profit:
                    # 止盈
                    pnl = (take_profit - entry_price) / entry_price - self.commission
                    equity *= (1 + pnl)
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': 'long',
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'pnl_pct': pnl * 100,
                        'exit_reason': 'take_profit'
                    })
                    position = 0

            elif position == -1:  # 空头
                if high[i] >= stop_loss:
                    # 止损
                    pnl = (entry_price - stop_loss) / entry_price - self.commission
                    equity *= (1 + pnl)
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': 'short',
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'pnl_pct': pnl * 100,
                        'exit_reason': 'stop_loss'
                    })
                    position = 0
                elif take_profit != 0 and low[i] <= take_profit:
                    # 止盈
                    pnl = (entry_price - take_profit) / entry_price - self.commission
                    equity *= (1 + pnl)
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'side': 'short',
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'pnl_pct': pnl * 100,
                        'exit_reason': 'take_profit'
                    })
                    position = 0

            # 新信号
            if position == 0:
                if long_signal[i] and not np.isnan(atr[i]):
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                    highest_since_entry = high[i]
                    # 固定 ATR 止损
                    stop_loss = entry_price - atr[i] * self.params['atr_mult']
                    # 追踪止盈模式: 初始设为无穷大
                    if use_trailing_tp:
                        take_profit = float('inf')
                    else:
                        take_profit = entry_price + atr[i] * self.params['trail_atr_mult']
                    equity *= (1 - self.commission)  # 入场手续费

                elif short_signal[i] and not np.isnan(atr[i]):
                    position = -1
                    entry_price = close[i]
                    entry_idx = i
                    lowest_since_entry = low[i]
                    # 固定 ATR 止损
                    stop_loss = entry_price + atr[i] * self.params['atr_mult']
                    # 追踪止盈模式
                    if use_trailing_tp:
                        take_profit = 0
                    else:
                        take_profit = entry_price - atr[i] * self.params['trail_atr_mult']
                    equity *= (1 - self.commission)

            equity_curve.append(equity)

        # 计算统计
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
        losing_trades = total_trades - winning_trades

        if total_trades > 0:
            win_rate = winning_trades / total_trades * 100
            avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]) if losing_trades > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) * 100

        # 夏普比率 (假设无风险利率为 0)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)  # 4H bars
        else:
            sharpe = 0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': equity,
            'total_return_pct': (equity / self.initial_capital - 1) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,
            'trades': trades,
            'equity_curve': equity_curve,
            'indicators': {
                'tpx': tpx,
                'ema_55': ema_55,
                'ema_89': ema_89,
                'ema_200': ema_200,
                't3': t3,
                'rsi': rsi,
                'atr': atr,
            }
        }


# =============================================================================
# NautilusTrader 策略 (完整版)
# =============================================================================

if NAUTILUS_AVAILABLE:

    class TPXSlingShotConfig(StrategyConfig):
        """TPX SlingShot 策略配置"""
        instrument_id: str
        tpx_length: int = 14
        tpx_smooth: int = 5
        ema_fast: int = 55
        ema_medium: int = 89
        ema_slow: int = 200
        t3_length: int = 8
        t3_factor: float = 0.7
        hma_length: int = 21
        atr_length: int = 14
        atr_mult: float = 2.0
        trail_atr_mult: float = 3.0
        use_rsi_filter: bool = True
        rsi_length: int = 14
        rsi_oversold: int = 30
        rsi_overbought: int = 70


    class TPXSlingShotStrategy(Strategy):
        """
        TPX SlingShot 策略 - NautilusTrader 版本

        完整的事件驱动策略实现
        """

        def __init__(self, config: TPXSlingShotConfig):
            super().__init__(config)
            self.instrument_id = InstrumentId.from_str(config.instrument_id)
            self.config = config

            # 存储历史数据用于指标计算
            self.closes = []
            self.highs = []
            self.lows = []
            self.opens = []

            # 缓存计算结果
            self._last_tpx = None
            self._last_ema_55 = None
            self._last_ema_89 = None
            self._last_ema_200 = None
            self._last_t3 = None
            self._last_rsi = None
            self._last_atr = None

        def on_start(self):
            """策略启动时调用"""
            self.log.info("TPX SlingShot Strategy started")

        def on_bar(self, bar: Bar):
            """
            处理新的 K 线数据

            这是策略的核心逻辑
            """
            # 更新历史数据
            self.closes.append(float(bar.close))
            self.highs.append(float(bar.high))
            self.lows.append(float(bar.low))
            self.opens.append(float(bar.open))

            # 确保有足够的数据
            min_bars = max(
                self.config.ema_slow,
                self.config.tpx_length + self.config.tpx_smooth,
                self.config.t3_length * 6,  # T3 需要 6 层 EMA
            )

            if len(self.closes) < min_bars + 1:
                return

            # 转换为 numpy 数组计算指标
            close_arr = np.array(self.closes)
            high_arr = np.array(self.highs)
            low_arr = np.array(self.lows)

            # 计算指标 (只取最后几个值以提高效率)
            tpx = calculate_tpx(
                close_arr, high_arr, low_arr,
                self.config.tpx_length,
                self.config.tpx_smooth
            )

            ema_55 = tv_ema(close_arr, self.config.ema_fast)
            ema_89 = tv_ema(close_arr, self.config.ema_medium)
            ema_200 = tv_ema(close_arr, self.config.ema_slow)
            t3 = calculate_t3(close_arr, self.config.t3_length, self.config.t3_factor)
            rsi = tv_rsi(close_arr, self.config.rsi_length)
            atr = tv_atr(high_arr, low_arr, close_arr, self.config.atr_length)

            # 当前值
            curr_close = close_arr[-1]
            curr_tpx = tpx[-1]
            prev_tpx = tpx[-2] if len(tpx) > 1 else np.nan
            curr_ema_55 = ema_55[-1]
            curr_ema_89 = ema_89[-1]
            curr_ema_200 = ema_200[-1]
            curr_t3 = t3[-1]
            curr_rsi = rsi[-1]
            curr_atr = atr[-1]

            # 检查 NaN
            if any(np.isnan([curr_tpx, prev_tpx, curr_ema_200, curr_t3, curr_atr])):
                return

            # 趋势判断
            uptrend = curr_close > curr_ema_200 and curr_ema_55 > curr_ema_89
            downtrend = curr_close < curr_ema_200 and curr_ema_55 < curr_ema_89

            # TPX 穿越
            tpx_cross_up = curr_tpx > 0 and prev_tpx <= 0
            tpx_cross_down = curr_tpx < 0 and prev_tpx >= 0

            # RSI 过滤
            rsi_long_ok = (not self.config.use_rsi_filter) or (curr_rsi < self.config.rsi_overbought)
            rsi_short_ok = (not self.config.use_rsi_filter) or (curr_rsi > self.config.rsi_oversold)

            # 价格位置
            price_above_t3 = curr_close > curr_t3
            price_below_t3 = curr_close < curr_t3

            # 交易逻辑
            long_condition = tpx_cross_up and uptrend and price_above_t3 and rsi_long_ok
            short_condition = tpx_cross_down and downtrend and price_below_t3 and rsi_short_ok

            # 获取当前仓位
            position = self.portfolio.positions.get(self.instrument_id)

            if long_condition and (position is None or position.side == PositionSide.SHORT):
                # 平空开多
                if position is not None:
                    self.close_position(self.instrument_id)

                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=self.instrument.make_qty(1),  # 简化处理
                )
                self.submit_order(order)

                # 设置止损止盈
                stop_loss = curr_close - curr_atr * self.config.atr_mult
                take_profit = curr_close + curr_atr * self.config.trail_atr_mult
                self.log.info(f"LONG entry at {curr_close}, SL: {stop_loss}, TP: {take_profit}")

            elif short_condition and (position is None or position.side == PositionSide.LONG):
                # 平多开空
                if position is not None:
                    self.close_position(self.instrument_id)

                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=self.instrument.make_qty(1),
                )
                self.submit_order(order)

                stop_loss = curr_close + curr_atr * self.config.atr_mult
                take_profit = curr_close - curr_atr * self.config.trail_atr_mult
                self.log.info(f"SHORT entry at {curr_close}, SL: {stop_loss}, TP: {take_profit}")

        def on_stop(self):
            """策略停止时调用"""
            self.log.info("TPX SlingShot Strategy stopped")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("TPX SlingShot System - NautilusTrader 适配器")
    print("=" * 60)
    print()

    # 检查是否有示例数据
    data_path = "/home/user/TV2PY/data/ETHUSDT_4h.csv"

    if os.path.exists(data_path):
        print(f"加载数据: {data_path}")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])

        print(f"数据范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
        print(f"总 K 线数: {len(df)}")
        print()

        # 运行简化回测
        print("运行简化回测...")
        backtest = TPXSlingShotBacktest()
        results = backtest.run(df)

        print()
        print("=" * 60)
        print("回测结果")
        print("=" * 60)
        print(f"初始资金:     ${results['initial_capital']:,.2f}")
        print(f"最终权益:     ${results['final_equity']:,.2f}")
        print(f"总收益率:     {results['total_return_pct']:.2f}%")
        print(f"最大回撤:     {results['max_drawdown_pct']:.2f}%")
        print(f"夏普比率:     {results['sharpe_ratio']:.2f}")
        print()
        print(f"总交易次数:   {results['total_trades']}")
        print(f"胜率:         {results['win_rate_pct']:.1f}%")
        print(f"平均盈利:     {results['avg_win_pct']:.2f}%")
        print(f"平均亏损:     {results['avg_loss_pct']:.2f}%")

    else:
        print(f"未找到数据文件: {data_path}")
        print("请先下载数据或修改数据路径")
        print()
        print("示例用法:")
        print("  1. 下载数据: python scripts/download_binance.py")
        print("  2. 运行回测: python strategies/tpx_nautilus_adapter.py")
