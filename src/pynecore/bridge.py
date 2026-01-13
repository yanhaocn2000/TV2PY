"""
Bridge between PyneCore and NautilusTrader.

This module provides utilities to convert PyneCore signals to NautilusTrader orders.
"""

from decimal import Decimal
from typing import Callable, Any

import pandas as pd
import numpy as np

from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


class PyneNautilusBridge:
    """
    Bridge class to connect PyneCore indicators/signals with NautilusTrader.

    This allows you to:
    1. Use PyneCore's Pine Script compatible indicators
    2. Generate signals using PyneCore logic
    3. Execute trades through NautilusTrader
    """

    def __init__(self, strategy: Strategy, instrument_id: InstrumentId):
        """
        Initialize the bridge.

        Args:
            strategy: NautilusTrader strategy instance
            instrument_id: The instrument to trade
        """
        self.strategy = strategy
        self.instrument_id = instrument_id
        self._bars_data: list[dict] = []

    def on_bar(self, bar: Bar) -> dict[str, float]:
        """
        Process a new bar and return OHLCV data.

        Args:
            bar: NautilusTrader Bar object

        Returns:
            Dictionary with open, high, low, close, volume
        """
        bar_data = {
            "timestamp": bar.ts_event,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        self._bars_data.append(bar_data)
        return bar_data

    def get_series(self, field: str, length: int | None = None) -> np.ndarray:
        """
        Get a series of historical data.

        Args:
            field: Field name (open, high, low, close, volume)
            length: Number of bars to return (None = all)

        Returns:
            Numpy array of values
        """
        if not self._bars_data:
            return np.array([])

        values = [bar[field] for bar in self._bars_data]
        if length is not None:
            values = values[-length:]
        return np.array(values)

    @property
    def close(self) -> np.ndarray:
        """Get close price series."""
        return self.get_series("close")

    @property
    def open(self) -> np.ndarray:
        """Get open price series."""
        return self.get_series("open")

    @property
    def high(self) -> np.ndarray:
        """Get high price series."""
        return self.get_series("high")

    @property
    def low(self) -> np.ndarray:
        """Get low price series."""
        return self.get_series("low")

    @property
    def volume(self) -> np.ndarray:
        """Get volume series."""
        return self.get_series("volume")

    def entry_long(self, qty: Decimal, comment: str = "") -> None:
        """
        Enter a long position (like strategy.entry in Pine Script).

        Args:
            qty: Quantity to buy
            comment: Order comment
        """
        instrument = self.strategy.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        order = self.strategy.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=instrument.make_qty(qty),
        )
        self.strategy.submit_order(order)
        if comment:
            self.strategy.log.info(f"Long entry: {comment}")

    def entry_short(self, qty: Decimal, comment: str = "") -> None:
        """
        Enter a short position (like strategy.entry in Pine Script).

        Args:
            qty: Quantity to sell
            comment: Order comment
        """
        instrument = self.strategy.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        order = self.strategy.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=instrument.make_qty(qty),
        )
        self.strategy.submit_order(order)
        if comment:
            self.strategy.log.info(f"Short entry: {comment}")

    def close_position(self, comment: str = "") -> None:
        """
        Close all positions (like strategy.close_all in Pine Script).

        Args:
            comment: Order comment
        """
        self.strategy.close_all_positions(self.instrument_id)
        if comment:
            self.strategy.log.info(f"Close position: {comment}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert collected bar data to pandas DataFrame.

        Returns:
            DataFrame with OHLCV data
        """
        if not self._bars_data:
            return pd.DataFrame()

        df = pd.DataFrame(self._bars_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
        df = df.set_index("timestamp")
        return df
