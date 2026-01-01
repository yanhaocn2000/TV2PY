"""
MACD (Moving Average Convergence Divergence) Strategy.

This strategy generates signals based on MACD line crossing the signal line.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class MACDConfig(StrategyConfig, frozen=True):
    """Configuration for MACD Strategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    trade_size: Decimal = Decimal("1.0")
    order_id_tag: str = "001"


class MACDStrategy(Strategy):
    """
    A trend-following strategy based on MACD.

    When MACD line crosses above the signal line, go long.
    When MACD line crosses below the signal line, go short.
    """

    def __init__(self, config: MACDConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size

        # Indicators
        self.macd = MovingAverageConvergenceDivergence(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            signal_period=config.signal_period,
        )

        # State
        self.instrument: Instrument | None = None
        self._prev_macd: float | None = None
        self._prev_signal: float | None = None

    def on_start(self) -> None:
        """Called when the strategy is started."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        # Register indicators
        self.register_indicator_for_bars(self.bar_type, self.macd)

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)

        self.log.info(f"MACD Strategy started for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a bar is received."""
        if not self.macd.initialized:
            return

        macd_value = self.macd.value
        signal_value = self.macd.signal

        # Get current position
        position = self.cache.position_for_instrument(self.instrument_id)
        is_flat = position is None or position.is_flat

        # Check for crossovers
        if self._prev_macd is not None and self._prev_signal is not None:
            # Bullish crossover (MACD crosses above signal)
            prev_diff = self._prev_macd - self._prev_signal
            curr_diff = macd_value - signal_value

            if prev_diff < 0 and curr_diff >= 0:
                # Bullish signal
                if is_flat:
                    self._enter_long()
                elif position and position.is_short:
                    self._close_position()
                    self._enter_long()

            elif prev_diff > 0 and curr_diff <= 0:
                # Bearish signal
                if is_flat:
                    self._enter_short()
                elif position and position.is_long:
                    self._close_position()
                    self._enter_short()

        self._prev_macd = macd_value
        self._prev_signal = signal_value

    def _enter_long(self) -> None:
        """Enter a long position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"MACD bullish crossover - Entering LONG")

    def _enter_short(self) -> None:
        """Enter a short position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"MACD bearish crossover - Entering SHORT")

    def _close_position(self) -> None:
        """Close all positions for this instrument."""
        self.close_all_positions(self.instrument_id)

    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self.macd.reset()
        self._prev_macd = None
        self._prev_signal = None

    def on_data(self, data: Data) -> None:
        pass
