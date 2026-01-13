"""
PyneCore Strategy Integration for NautilusTrader.

Provides a base class for creating strategies using Pine Script style syntax.
"""

from decimal import Decimal
from typing import Any

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy

from src.pynecore.bridge import PyneNautilusBridge
from src.pynecore.indicators import PyneIndicators, ta


class PyneStrategyConfig(StrategyConfig, frozen=True):
    """Base configuration for Pyne strategies."""

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal = Decimal("1.0")


class PyneStrategy(Strategy):
    """
    Base class for Pine Script style strategies.

    Provides convenient access to:
    - OHLCV data as numpy arrays
    - Pine Script compatible indicators (ta.*)
    - Simple entry/exit methods

    Example:
        class MyStrategy(PyneStrategy):
            def on_bar(self, bar: Bar):
                super().on_bar(bar)

                # Access data like Pine Script
                fast = ta.ema(self.close, 10)
                slow = ta.ema(self.close, 20)

                if ta.crossover(fast, slow)[-1]:
                    self.entry_long()
                elif ta.crossunder(fast, slow)[-1]:
                    self.close_long()
    """

    def __init__(self, config: PyneStrategyConfig):
        super().__init__(config)

        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size

        # PyneCore bridge
        self._bridge: PyneNautilusBridge | None = None
        self.instrument: Instrument | None = None

        # Indicator module
        self.ta = PyneIndicators

    def on_start(self) -> None:
        """Called when strategy starts."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        self._bridge = PyneNautilusBridge(self, self.instrument_id)
        self.subscribe_bars(self.bar_type)

        self.log.info(f"PyneStrategy started for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """
        Called on each new bar. Override this method to implement strategy logic.

        Always call super().on_bar(bar) first to update data arrays.
        """
        if self._bridge:
            self._bridge.on_bar(bar)

    # =========================================================================
    # Data access (Pine Script style)
    # =========================================================================

    @property
    def close(self):
        """Close price series."""
        return self._bridge.close if self._bridge else []

    @property
    def open(self):
        """Open price series."""
        return self._bridge.open if self._bridge else []

    @property
    def high(self):
        """High price series."""
        return self._bridge.high if self._bridge else []

    @property
    def low(self):
        """Low price series."""
        return self._bridge.low if self._bridge else []

    @property
    def volume(self):
        """Volume series."""
        return self._bridge.volume if self._bridge else []

    @property
    def bar_index(self) -> int:
        """Current bar index (0-based)."""
        return len(self.close) - 1 if len(self.close) > 0 else 0

    # =========================================================================
    # Trading methods (Pine Script style)
    # =========================================================================

    def entry_long(self, qty: Decimal | None = None, comment: str = "") -> None:
        """
        Enter a long position.

        Equivalent to: strategy.entry("Long", strategy.long, qty)
        """
        if self._bridge:
            self._bridge.entry_long(qty or self.trade_size, comment)

    def entry_short(self, qty: Decimal | None = None, comment: str = "") -> None:
        """
        Enter a short position.

        Equivalent to: strategy.entry("Short", strategy.short, qty)
        """
        if self._bridge:
            self._bridge.entry_short(qty or self.trade_size, comment)

    def close_long(self, comment: str = "") -> None:
        """
        Close long position.

        Equivalent to: strategy.close("Long")
        """
        position = self.cache.position_for_instrument(self.instrument_id)
        if position and position.is_long:
            self.close_all_positions(self.instrument_id)
            if comment:
                self.log.info(f"Close long: {comment}")

    def close_short(self, comment: str = "") -> None:
        """
        Close short position.

        Equivalent to: strategy.close("Short")
        """
        position = self.cache.position_for_instrument(self.instrument_id)
        if position and position.is_short:
            self.close_all_positions(self.instrument_id)
            if comment:
                self.log.info(f"Close short: {comment}")

    def close_all(self, comment: str = "") -> None:
        """
        Close all positions.

        Equivalent to: strategy.close_all()
        """
        self.close_all_positions(self.instrument_id)
        if comment:
            self.log.info(f"Close all: {comment}")

    @property
    def position_size(self) -> float:
        """
        Current position size.

        Positive = long, Negative = short, Zero = flat
        """
        position = self.cache.position_for_instrument(self.instrument_id)
        if position is None:
            return 0.0
        return float(position.quantity) if position.is_long else -float(position.quantity)

    @property
    def is_long(self) -> bool:
        """True if currently in a long position."""
        return self.position_size > 0

    @property
    def is_short(self) -> bool:
        """True if currently in a short position."""
        return self.position_size < 0

    @property
    def is_flat(self) -> bool:
        """True if no position."""
        return self.position_size == 0

    def on_stop(self) -> None:
        """Called when strategy stops."""
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        """Called when strategy is reset."""
        if self._bridge:
            self._bridge._bars_data.clear()

    def on_data(self, data: Data) -> None:
        """Called when data is received."""
        pass
