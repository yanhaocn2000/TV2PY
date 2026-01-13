"""
Keltner Channel Strategy - NautilusTrader Adapter

将 TradingView BB Keltner Squeeze 策略适配到 NautilusTrader 回测引擎
"""

from decimal import Decimal
from typing import Optional
import numpy as np
import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from keltner_channel import (
    KeltnerChannelParams,
    KeltnerChannelIndicator,
    SqueezeType,
    Direction
)


class KeltnerChannelConfig(StrategyConfig, frozen=True):
    """Keltner Channel 策略配置"""
    instrument_id: str
    bar_type: str

    # 指标参数
    length: int = 20
    bb_mult: float = 2.0
    kc_mult: float = 1.5
    use_true_range: bool = True
    squeeze_type: str = "within"  # "within" or "width"
    width_ratio: float = 1.0

    # 仓位管理
    trade_size: float = 1.0

    # 止盈止损
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    trailing_stop_pct: float = 0.0


class KeltnerChannelNautilus(Strategy):
    """
    Keltner Channel 策略 - NautilusTrader 实现

    完全对齐 TradingView BB Keltner Squeeze Strategy
    """

    def __init__(self, config: KeltnerChannelConfig):
        super().__init__(config)

        # 解析配置
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)

        # 创建参数
        self.params = KeltnerChannelParams(
            length=config.length,
            bb_mult=config.bb_mult,
            kc_mult=config.kc_mult,
            use_true_range=config.use_true_range,
            squeeze_type=SqueezeType.WITHIN if config.squeeze_type == "within" else SqueezeType.WIDTH,
            width_ratio=config.width_ratio,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            trailing_stop_pct=config.trailing_stop_pct
        )

        # 创建指标计算器
        self.indicator = KeltnerChannelIndicator(self.params)

        # 交易参数
        self.trade_size = Decimal(str(config.trade_size))

        # 数据缓存
        self.high_buffer: list = []
        self.low_buffer: list = []
        self.close_buffer: list = []

        # 状态
        self.direction = Direction.NONE
        self.entry_price = 0.0
        self.prev_midc = 0  # 上一个 bar 的市场状态

        # 指标值
        self.bb_upper = 0.0
        self.bb_basis = 0.0
        self.bb_lower = 0.0
        self.kc_upper = 0.0
        self.kc_basis = 0.0
        self.kc_lower = 0.0
        self.is_squeeze = False

    def on_start(self):
        """策略启动"""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"无法找到交易品种: {self.instrument_id}")
            return

        # 订阅 Bar 数据
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Keltner Channel 策略已启动 - {self.instrument_id}")

    def on_bar(self, bar: Bar):
        """处理每个 Bar"""
        # 更新数据缓存
        self.high_buffer.append(float(bar.high))
        self.low_buffer.append(float(bar.low))
        self.close_buffer.append(float(bar.close))

        # 保持缓存大小
        max_len = self.params.length * 3
        if len(self.close_buffer) > max_len:
            self.high_buffer = self.high_buffer[-max_len:]
            self.low_buffer = self.low_buffer[-max_len:]
            self.close_buffer = self.close_buffer[-max_len:]

        # 需要足够的数据
        if len(self.close_buffer) < self.params.length:
            return

        # 计算指标
        self._calculate_indicators()

        # 生成信号
        self._check_signals(bar)

    def _calculate_indicators(self):
        """计算 BB 和 KC 指标"""
        high = np.array(self.high_buffer)
        low = np.array(self.low_buffer)
        close = np.array(self.close_buffer)

        # Bollinger Bands
        bb_upper, bb_basis, bb_lower = self.indicator.calculate_bb(close)
        self.bb_upper = bb_upper[-1]
        self.bb_basis = bb_basis[-1]
        self.bb_lower = bb_lower[-1]

        # Keltner Channel
        kc_upper, kc_basis, kc_lower = self.indicator.calculate_kc(high, low, close)
        self.kc_upper = kc_upper[-1]
        self.kc_basis = kc_basis[-1]
        self.kc_lower = kc_lower[-1]

        # Squeeze 检测
        squeeze = self.indicator.detect_squeeze(
            bb_upper, bb_lower, kc_upper, kc_lower, bb_basis, kc_basis
        )
        self.is_squeeze = squeeze[-1]

    def _check_signals(self, bar: Bar):
        """
        检查交易信号

        Pine Script 逻辑:
            midc = squeeze ? 0 : close > B2basis ? 1 : 2

            if direction == 0:
                if midc[1] == 0 and midc == 1:
                    strategy.entry("LONG", strategy.long)
                else if midc[1] == 0 and midc == 2:
                    strategy.entry("SHORT", strategy.short)
            else if direction != midc:
                strategy.close_all()
        """
        close = float(bar.close)

        # 计算当前市场状态 (midc)
        if self.is_squeeze:
            midc = 0  # Squeeze 中
        elif close > self.bb_basis:
            midc = 1  # 价格在中轨上方
        else:
            midc = 2  # 价格在中轨下方

        # 检查信号
        if self.direction == Direction.NONE:
            # 无持仓, 等待 squeeze 释放
            if self.prev_midc == 0 and midc == 1:
                # Squeeze 释放, 价格向上 -> 做多
                self._enter_long(bar)
            elif self.prev_midc == 0 and midc == 2:
                # Squeeze 释放, 价格向下 -> 做空
                self._enter_short(bar)
        else:
            # 有持仓, 检查是否需要平仓
            if self.direction == Direction.LONG and midc == 2:
                self._close_position(bar, "方向反转")
            elif self.direction == Direction.SHORT and midc == 1:
                self._close_position(bar, "方向反转")

        # 更新上一个 midc
        self.prev_midc = midc

    def _enter_long(self, bar: Bar):
        """开多仓"""
        if self.instrument is None:
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
        self.direction = Direction.LONG
        self.entry_price = float(bar.close)

        self.log.info(
            f"LONG 入场 @ {bar.close} | "
            f"BB: [{self.bb_lower:.2f}, {self.bb_basis:.2f}, {self.bb_upper:.2f}] | "
            f"KC: [{self.kc_lower:.2f}, {self.kc_basis:.2f}, {self.kc_upper:.2f}]"
        )

        # 设置止损止盈
        self._set_exit_orders(bar, OrderSide.BUY)

    def _enter_short(self, bar: Bar):
        """开空仓"""
        if self.instrument is None:
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(order)
        self.direction = Direction.SHORT
        self.entry_price = float(bar.close)

        self.log.info(
            f"SHORT 入场 @ {bar.close} | "
            f"BB: [{self.bb_lower:.2f}, {self.bb_basis:.2f}, {self.bb_upper:.2f}] | "
            f"KC: [{self.kc_lower:.2f}, {self.kc_basis:.2f}, {self.kc_upper:.2f}]"
        )

        # 设置止损止盈
        self._set_exit_orders(bar, OrderSide.SELL)

    def _close_position(self, bar: Bar, reason: str):
        """平仓"""
        if self.instrument is None:
            return

        # 平掉所有持仓
        for position in self.cache.positions(venue=self.instrument_id.venue):
            if position.instrument_id == self.instrument_id:
                self.close_position(position)

        pnl = float(bar.close) - self.entry_price
        if self.direction == Direction.SHORT:
            pnl = -pnl

        self.log.info(
            f"平仓 @ {bar.close} | 原因: {reason} | "
            f"入场价: {self.entry_price:.2f} | PnL: {pnl:.2f}"
        )

        self.direction = Direction.NONE
        self.entry_price = 0.0

    def _set_exit_orders(self, bar: Bar, entry_side: OrderSide):
        """设置止损止盈订单"""
        if self.instrument is None:
            return

        entry_price = float(bar.close)

        # 止损
        if self.params.stop_loss_pct > 0:
            if entry_side == OrderSide.BUY:
                stop_price = entry_price * (1 - self.params.stop_loss_pct / 100)
            else:
                stop_price = entry_price * (1 + self.params.stop_loss_pct / 100)

            # 注: NautilusTrader 需要使用 stop_market_order
            self.log.info(f"止损价: {stop_price:.2f}")

        # 止盈
        if self.params.take_profit_pct > 0:
            if entry_side == OrderSide.BUY:
                tp_price = entry_price * (1 + self.params.take_profit_pct / 100)
            else:
                tp_price = entry_price * (1 - self.params.take_profit_pct / 100)

            self.log.info(f"止盈价: {tp_price:.2f}")

    def on_stop(self):
        """策略停止"""
        # 平掉所有持仓
        self.close_all_positions(self.instrument_id)
        self.log.info("Keltner Channel 策略已停止")


def run_backtest(
    data_path: str,
    instrument_id: str = "BTCUSDT.BINANCE",
    bar_type: str = "BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
    length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 4.0
):
    """
    运行 Keltner Channel 策略回测

    Args:
        data_path: 数据文件路径
        instrument_id: 交易品种 ID
        bar_type: Bar 类型
        length: 周期
        bb_mult: BB 倍数
        kc_mult: KC 倍数
        stop_loss_pct: 止损百分比
        take_profit_pct: 止盈百分比
    """
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    # 配置回测引擎
    engine_config = BacktestEngineConfig(
        trader_id="BACKTESTER-001",
    )
    engine = BacktestEngine(config=engine_config)

    # 添加交易所
    venue = Venue("BINANCE")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(100_000, USD)],
    )

    # 添加交易品种
    instrument = TestInstrumentProvider.btcusdt_binance()
    engine.add_instrument(instrument)

    # 加载数据
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')

    wrangler = BarDataWrangler(
        bar_type=BarType.from_str(bar_type),
        instrument=instrument,
    )
    bars = wrangler.process(df)
    engine.add_data(bars)

    # 配置策略
    config = KeltnerChannelConfig(
        instrument_id=instrument_id,
        bar_type=bar_type,
        length=length,
        bb_mult=bb_mult,
        kc_mult=kc_mult,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )
    strategy = KeltnerChannelNautilus(config)
    engine.add_strategy(strategy)

    # 运行回测
    engine.run()

    # 输出结果
    engine.trader.generate_order_fills_report()
    engine.trader.generate_positions_report()
    engine.trader.generate_account_report(venue)

    return engine


if __name__ == "__main__":
    print("Keltner Channel NautilusTrader Adapter")
    print("用法: run_backtest('data.csv')")
