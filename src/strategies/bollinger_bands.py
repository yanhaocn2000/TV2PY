"""
Bollinger Bands Strategy.

This strategy generates signals based on price touching or crossing the bands.
Mean reversion approach: buy at lower band, sell at upper band.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.bollinger_bands import BollingerBands
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class BollingerBandsConfig(StrategyConfig, frozen=True):
    """Configuration for Bollinger Bands Strategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    period: int = 20
    num_std: float = 2.0
    trade_size: Decimal = Decimal("1.0")
    order_id_tag: str = "001"


class BollingerBandsStrategy(Strategy):
    """
    A mean-reversion strategy based on Bollinger Bands.

    When price touches the lower band, go long (expecting price to revert to mean).
    When price touches the upper band, go short (expecting price to revert to mean).
    Exit when price crosses the middle band.
    """

    def __init__(self, config: BollingerBandsConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size

        # Indicators
        self.bb = BollingerBands(period=config.period, k=config.num_std)

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
        self.register_indicator_for_bars(self.bar_type, self.bb)

        # Subscribe to bar data
        self.subscribe_bars(self.bar_type)

        self.log.info(f"Bollinger Bands Strategy started for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a bar is received."""
        if not self.bb.initialized:
            return

        close_price = float(bar.close)
        upper_band = self.bb.upper
        middle_band = self.bb.middle
        lower_band = self.bb.lower

        # Get current position
        position = self.cache.position_for_instrument(self.instrument_id)
        is_flat = position is None or position.is_flat

        # Trading logic
        if close_price <= lower_band:
            # Price at lower band - buy signal
            if is_flat:
                self._enter_long()
            elif position and position.is_short:
                self._close_position()
                self._enter_long()

        elif close_price >= upper_band:
            # Price at upper band - sell signal
            if is_flat:
                self._enter_short()
            elif position and position.is_long:
                self._close_position()
                self._enter_short()

        # Exit at middle band
        elif position and not is_flat:
            if position.is_long and close_price >= middle_band:
                self._close_position()
                self.log.info("Price reached middle band - closing long")
            elif position.is_short and close_price <= middle_band:
                self._close_position()
                self.log.info("Price reached middle band - closing short")

    def _enter_long(self) -> None:
        """Enter a long position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Price at lower band - Entering LONG")

    def _enter_short(self) -> None:
        """Enter a short position."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Price at upper band - Entering SHORT")

    def _close_position(self) -> None:
        """Close all positions for this instrument."""
        self.close_all_positions(self.instrument_id)

    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self.bb.reset()

    def on_data(self, data: Data) -> None:
        pass
