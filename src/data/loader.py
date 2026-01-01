"""
Data loading utilities for backtesting.

Provides functions to load and prepare historical data from various sources.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from nautilus_trader.model.data import Bar, BarType, QuoteTick, TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.wranglers import BarDataWrangler, QuoteTickDataWrangler, TradeTickDataWrangler


class DataLoader:
    """Utility class for loading and wrangling market data."""

    @staticmethod
    def load_bars_from_csv(
        file_path: str | Path,
        instrument_id: InstrumentId,
        bar_type: BarType,
        timestamp_column: str = "timestamp",
        open_column: str = "open",
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close",
        volume_column: str = "volume",
    ) -> list[Bar]:
        """
        Load bar data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            instrument_id: The instrument ID.
            bar_type: The bar type.
            timestamp_column: Name of the timestamp column.
            open_column: Name of the open price column.
            high_column: Name of the high price column.
            low_column: Name of the low price column.
            close_column: Name of the close price column.
            volume_column: Name of the volume column.

        Returns:
            List of Bar objects.
        """
        df = pd.read_csv(file_path)

        # Ensure timestamp is datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.set_index(timestamp_column)

        # Rename columns to standard format
        df = df.rename(columns={
            open_column: "open",
            high_column: "high",
            low_column: "low",
            close_column: "close",
            volume_column: "volume",
        })

        wrangler = BarDataWrangler(bar_type=bar_type, instrument=None)
        return wrangler.process(df)

    @staticmethod
    def load_trades_from_csv(
        file_path: str | Path,
        instrument_id: InstrumentId,
        timestamp_column: str = "timestamp",
        price_column: str = "price",
        quantity_column: str = "quantity",
    ) -> list[TradeTick]:
        """
        Load trade tick data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            instrument_id: The instrument ID.
            timestamp_column: Name of the timestamp column.
            price_column: Name of the price column.
            quantity_column: Name of the quantity column.

        Returns:
            List of TradeTick objects.
        """
        df = pd.read_csv(file_path)

        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.set_index(timestamp_column)

        df = df.rename(columns={
            price_column: "price",
            quantity_column: "quantity",
        })

        wrangler = TradeTickDataWrangler(instrument=None)
        return wrangler.process(df)

    @staticmethod
    def load_quotes_from_csv(
        file_path: str | Path,
        instrument_id: InstrumentId,
        timestamp_column: str = "timestamp",
        bid_column: str = "bid",
        ask_column: str = "ask",
        bid_size_column: str = "bid_size",
        ask_size_column: str = "ask_size",
    ) -> list[QuoteTick]:
        """
        Load quote tick data from a CSV file.

        Args:
            file_path: Path to the CSV file.
            instrument_id: The instrument ID.
            timestamp_column: Name of the timestamp column.
            bid_column: Name of the bid price column.
            ask_column: Name of the ask price column.
            bid_size_column: Name of the bid size column.
            ask_size_column: Name of the ask size column.

        Returns:
            List of QuoteTick objects.
        """
        df = pd.read_csv(file_path)

        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.set_index(timestamp_column)

        df = df.rename(columns={
            bid_column: "bid",
            ask_column: "ask",
            bid_size_column: "bid_size",
            ask_size_column: "ask_size",
        })

        wrangler = QuoteTickDataWrangler(instrument=None)
        return wrangler.process(df)

    @staticmethod
    def download_crypto_data(
        symbol: str,
        exchange: str = "binance",
        timeframe: str = "1h",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Download cryptocurrency OHLCV data using ccxt.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT").
            exchange: Exchange name (e.g., "binance").
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d").
            start_date: Start date for data download.
            end_date: End date for data download.

        Returns:
            DataFrame with OHLCV data.
        """
        import ccxt

        exchange_class = getattr(ccxt, exchange)
        ex = exchange_class()

        since = int(start_date.timestamp() * 1000) if start_date else None

        ohlcv = ex.fetch_ohlcv(symbol, timeframe, since=since)

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        if end_date:
            df = df[df.index <= end_date]

        return df

    @staticmethod
    def download_stock_data(
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download stock OHLCV data using yfinance.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL").
            start_date: Start date string (e.g., "2023-01-01").
            end_date: End date string (e.g., "2024-01-01").
            interval: Data interval (e.g., "1m", "1h", "1d").

        Returns:
            DataFrame with OHLCV data.
        """
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        return df[["open", "high", "low", "close", "volume"]]
