#!/usr/bin/env python3
"""
Example: Download Historical Data

This example demonstrates how to download historical market data
from various sources for backtesting.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


def download_crypto_example():
    """Download cryptocurrency data from Binance."""
    print("Downloading BTC/USDT 1-hour data from Binance...")

    loader = DataLoader()

    # Download last 30 days of data
    end_date = datetime.now()
    start_date = datetime(end_date.year, end_date.month - 1, end_date.day)

    try:
        df = loader.download_crypto_data(
            symbol="BTC/USDT",
            exchange="binance",
            timeframe="1h",
            start_date=start_date,
            end_date=end_date,
        )

        print(f"Downloaded {len(df)} candles")
        print(df.head())

        # Save to CSV
        output_path = Path(__file__).parent.parent / "data" / "btcusdt_1h.csv"
        df.to_csv(output_path)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"Error downloading crypto data: {e}")
        print("Make sure ccxt is installed: pip install ccxt")


def download_stock_example():
    """Download stock data using yfinance."""
    print("\nDownloading AAPL daily data...")

    loader = DataLoader()

    try:
        df = loader.download_stock_data(
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
            interval="1d",
        )

        print(f"Downloaded {len(df)} candles")
        print(df.head())

        # Save to CSV
        output_path = Path(__file__).parent.parent / "data" / "aapl_daily.csv"
        df.to_csv(output_path)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"Error downloading stock data: {e}")
        print("Make sure yfinance is installed: pip install yfinance")


if __name__ == "__main__":
    download_crypto_example()
    download_stock_example()
