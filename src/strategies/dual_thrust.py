"""
Dual Thrust Strategy.

A classic breakout strategy that uses the range of the previous N bars
to set dynamic entry thresholds.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class DualThrustConfig(StrategyConfig, frozen=True):
    """Configuration for Dual Thrust Strategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    lookback_period: int = 4
    k1: float = 0.5  # Upper threshold multiplier
    k2: float = 0.5  # Lower threshold multiplier
    trade_size: Decimal = Decimal("1.0")
    order_id_tag: str = "001"


class DualThrustStrategy(Strategy):
    """
    Dual Thrust breakout strategy.

    Calculates a range based on the highest high, lowest low,
    highest close, and lowest close over the lookback period.
    Enters long when price breaks above open + k1 * range.
    Enters short when price breaks below open - k2 * range.
    """

    def __init__(self, config: DualThrustConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size
        self.lookback_period = config.lookback_period
        self.k1 = config.k1
        self.k2 = config.k2

        # State
        self.instrument: Instrument | None = None
        self._bars: list[Bar] = []
        self._today_open: float | None = None
        self._upper_band: float | None = None
        self._lower_band: float | None = None

    def on_start(self) -> None:
        """Called when the strategy is started."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        self.subscribe_bars(self.bar_type)
        self.log.info(f"Dual Thrust Strategy started for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a bar is received."""
        self._bars.append(bar)

        # Keep only the bars we need
        if len(self._bars) > self.lookback_period + 1:
            self._bars = self._bars[-(self.lookback_period + 1):]

        # Need enough bars to calculate range
        if len(self._bars) < self.lookback_period + 1:
            return

        # Calculate range from lookback period (excluding current bar)
        lookback_bars = self._bars[:-1]

        hh = max(float(b.high) for b in lookback_bars)  # Highest high
        ll = min(float(b.low) for b in lookback_bars)   # Lowest low
        hc = max(float(b.close) for b in lookback_bars) # Highest close
        lc = min(float(b.close) for b in lookback_bars) # Lowest close

        # Calculate range
        range_value = max(hh - lc, hc - ll)

        # Use previous bar's close as the reference (like session open)
        open_price = float(self._bars[-2].close)

        self._upper_band = open_price + self.k1 * range_value
        self._lower_band = open_price - self.k2 * range_value

        current_price = float(bar.close)

        # Get current position
        position = self.cache.position_for_instrument(self.instrument_id)
        is_flat = position is None or position.is_flat

        # Trading logic
        if current_price > self._upper_band:
            if is_flat:
                self._enter_long()
            elif position and position.is_short:
                self._close_position()
                self._enter_long()

        elif current_price < self._lower_band:
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
        self.log.info(f"Upper band breakout - Entering LONG at {self._upper_band:.2f}")

    def _enter_short(self) -> None:
        """Enter a short position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Lower band breakout - Entering SHORT at {self._lower_band:.2f}")

    def _close_position(self) -> None:
        """Close all positions for this instrument."""
        self.close_all_positions(self.instrument_id)

    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self._bars = []
        self._upper_band = None
        self._lower_band = None

    def on_data(self, data: Data) -> None:
        pass
