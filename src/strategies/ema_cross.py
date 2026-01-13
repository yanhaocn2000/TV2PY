"""
EMA Cross Strategy - A classic trend-following strategy.

This strategy generates buy signals when the fast EMA crosses above the slow EMA,
and sell signals when the fast EMA crosses below the slow EMA.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.ema import ExponentialMovingAverage
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


class EMACrossConfig(StrategyConfig, frozen=True):
    """Configuration for EMA Cross Strategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    fast_ema_period: int = 10
    slow_ema_period: int = 20
    trade_size: Decimal = Decimal("1.0")
    order_id_tag: str = "001"


class EMACrossStrategy(Strategy):
    """
    A simple EMA crossover strategy.

    When the fast EMA crosses above the slow EMA, go long.
    When the fast EMA crosses below the slow EMA, go short.
    """

    def __init__(self, config: EMACrossConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size

        # Indicators
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

        # State
        self.instrument: Instrument | None = None

    def on_start(self) -> None:
        """Called when the strategy is started."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        # Register indicators
        self.register_indicator_for_bars(self.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.bar_type, self.slow_ema)

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)

        self.log.info(f"Strategy started for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a bar is received."""
        self.log.debug(f"Received bar: {bar}")

        # Wait for indicators to warm up
        if not self.fast_ema.initialized or not self.slow_ema.initialized:
            return

        # Check for crossover signals
        fast_value = self.fast_ema.value
        slow_value = self.slow_ema.value

        # Get current position
        position = self.cache.position_for_instrument(self.instrument_id)
        is_flat = position is None or position.is_flat

        # Trading logic
        if fast_value > slow_value:
            # Bullish signal
            if is_flat:
                self._enter_long()
            elif position and position.is_short:
                self._close_position()
                self._enter_long()
        elif fast_value < slow_value:
            # Bearish signal
            if is_flat:
                self._enter_short()
            elif position and position.is_long:
                self._close_position()
                self._enter_short()

    def _enter_long(self) -> None:
        """Enter a long position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Entering LONG position: {order}")

    def _enter_short(self) -> None:
        """Enter a short position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Entering SHORT position: {order}")

    def _close_position(self) -> None:
        """Close all positions for this instrument."""
        self.close_all_positions(self.instrument_id)
        self.log.info("Closing all positions")

    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        self.close_all_positions(self.instrument_id)
        self.log.info("Strategy stopped")

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self.fast_ema.reset()
        self.slow_ema.reset()

    def on_data(self, data: Data) -> None:
        """Called when data is received."""
        pass
