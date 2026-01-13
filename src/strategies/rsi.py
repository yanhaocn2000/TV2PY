"""
RSI (Relative Strength Index) Strategy.

This strategy generates buy signals when RSI is oversold (below threshold),
and sell signals when RSI is overbought (above threshold).
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class RSIConfig(StrategyConfig, frozen=True):
    """Configuration for RSI Strategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    rsi_period: int = 14
    overbought: float = 70.0
    oversold: float = 30.0
    trade_size: Decimal = Decimal("1.0")
    order_id_tag: str = "001"


class RSIStrategy(Strategy):
    """
    A mean-reversion strategy based on RSI.

    When RSI drops below the oversold level, go long (expecting price to rise).
    When RSI rises above the overbought level, go short (expecting price to fall).
    """

    def __init__(self, config: RSIConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size
        self.overbought = config.overbought
        self.oversold = config.oversold

        # Indicators
        self.rsi = RelativeStrengthIndex(config.rsi_period)

        # State
        self.instrument: Instrument | None = None
        self._last_rsi: float | None = None

    def on_start(self) -> None:
        """Called when the strategy is started."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        # Register indicators
        self.register_indicator_for_bars(self.bar_type, self.rsi)

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)

        self.log.info(f"RSI Strategy started for {self.instrument_id}")
        self.log.info(f"Overbought: {self.overbought}, Oversold: {self.oversold}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a bar is received."""
        if not self.rsi.initialized:
            return

        current_rsi = self.rsi.value

        # Get current position
        position = self.cache.position_for_instrument(self.instrument_id)
        is_flat = position is None or position.is_flat

        # Trading logic based on RSI crossovers
        if self._last_rsi is not None:
            # Oversold crossover (RSI crosses above oversold from below)
            if self._last_rsi < self.oversold <= current_rsi:
                if is_flat:
                    self._enter_long()
                elif position and position.is_short:
                    self._close_position()
                    self._enter_long()

            # Overbought crossover (RSI crosses below overbought from above)
            elif self._last_rsi > self.overbought >= current_rsi:
                if is_flat:
                    self._enter_short()
                elif position and position.is_long:
                    self._close_position()
                    self._enter_short()

        self._last_rsi = current_rsi

    def _enter_long(self) -> None:
        """Enter a long position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"RSI oversold - Entering LONG: RSI={self.rsi.value:.2f}")

    def _enter_short(self) -> None:
        """Enter a short position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"RSI overbought - Entering SHORT: RSI={self.rsi.value:.2f}")

    def _close_position(self) -> None:
        """Close all positions for this instrument."""
        self.close_all_positions(self.instrument_id)

    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self.rsi.reset()
        self._last_rsi = None

    def on_data(self, data: Data) -> None:
        pass
