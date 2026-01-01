#!/usr/bin/env python3
"""
Download historical data from Binance.

This script downloads OHLCV data from Binance and saves it in multiple formats
for use with NautilusTrader backtesting.

Usage:
    python scripts/download_binance_data.py

Requirements:
    pip install ccxt pandas
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import time

import ccxt
import pandas as pd


def download_binance_ohlcv(
    symbol: str = "ETH/USDT",
    timeframe: str = "4h",
    start_date: str = "2019-01-01",
    end_date: str = "2026-01-01",
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Download OHLCV data from Binance.

    Args:
        symbol: Trading pair (e.g., "ETH/USDT", "BTC/USDT")
        timeframe: Candle timeframe (e.g., "1m", "1h", "4h", "1d")
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        output_dir: Directory to save output files

    Returns:
        DataFrame with OHLCV data
    """
    print(f"{'=' * 60}")
    print(f"Binance Data Downloader")
    print(f"{'=' * 60}")
    print(f"Symbol:     {symbol}")
    print(f"Timeframe:  {timeframe}")
    print(f"Start:      {start_date}")
    print(f"End:        {end_date}")
    print(f"{'=' * 60}")

    # Initialize Binance exchange
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    # Calculate timeframe in milliseconds
    timeframe_ms = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
    }.get(timeframe, 60 * 60 * 1000)

    # Binance limit is 1000 candles per request
    limit = 1000

    all_ohlcv = []
    current_ts = start_ts
    request_count = 0

    print(f"\nDownloading data...")

    while current_ts < end_ts:
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ts,
                limit=limit,
            )

            if not ohlcv:
                print(f"No more data available")
                break

            all_ohlcv.extend(ohlcv)

            # Update timestamp for next request
            last_ts = ohlcv[-1][0]
            current_ts = last_ts + timeframe_ms

            request_count += 1

            # Progress update
            current_date = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            print(f"  [{request_count}] Downloaded up to {current_date.strftime('%Y-%m-%d %H:%M')} "
                  f"({len(all_ohlcv)} candles)")

            # Rate limiting - be nice to the API
            time.sleep(0.1)

        except ccxt.RateLimitExceeded:
            print("Rate limit exceeded, waiting 60 seconds...")
            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

    if not all_ohlcv:
        print("No data downloaded!")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"])

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")

    # Sort by time
    df = df.sort_index()

    # Filter to requested date range
    df = df[start_date:end_date]

    print(f"\n{'=' * 60}")
    print(f"Download Complete!")
    print(f"{'=' * 60}")
    print(f"Total candles: {len(df)}")
    print(f"Date range:    {df.index[0]} to {df.index[-1]}")
    print(f"{'=' * 60}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    symbol_clean = symbol.replace("/", "")
    filename_base = f"{symbol_clean}_{timeframe}_{start_date}_{end_date}"

    # Save as CSV
    csv_path = output_path / f"{filename_base}.csv"
    df.to_csv(csv_path)
    print(f"\nSaved CSV:     {csv_path}")

    # Save as Parquet (more efficient)
    try:
        parquet_path = output_path / f"{filename_base}.parquet"
        df.to_parquet(parquet_path)
        print(f"Saved Parquet: {parquet_path}")
    except Exception as e:
        print(f"Could not save Parquet (install pyarrow): {e}")

    # Print data summary
    print(f"\nData Summary:")
    print(f"  Open:   {df['open'].min():.2f} - {df['open'].max():.2f}")
    print(f"  High:   {df['high'].min():.2f} - {df['high'].max():.2f}")
    print(f"  Low:    {df['low'].min():.2f} - {df['low'].max():.2f}")
    print(f"  Close:  {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"  Volume: {df['volume'].min():.2f} - {df['volume'].max():.2f}")

    return df


def main():
    """Main entry point."""
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Download ETH/USDT 4H data from 2019 to 2026
    df = download_binance_ohlcv(
        symbol="ETH/USDT",
        timeframe="4h",
        start_date="2019-01-01",
        end_date="2026-01-01",
        output_dir="data",
    )

    return df


if __name__ == "__main__":
    main()
