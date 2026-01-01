#!/usr/bin/env python3
"""
Generate realistic sample ETH/USDT data for testing.

Since Binance API is not accessible, this script generates
realistic price data based on historical ETH patterns.

Usage:
    python scripts/generate_sample_data.py
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd


def generate_eth_data(
    start_date: str = "2019-01-01",
    end_date: str = "2026-01-01",
    timeframe_hours: int = 4,
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Generate realistic ETH/USDT OHLCV data.

    Uses geometric Brownian motion with regime changes to simulate
    realistic crypto price movements.
    """
    print(f"{'=' * 60}")
    print(f"ETH/USDT Sample Data Generator")
    print(f"{'=' * 60}")
    print(f"Timeframe:  {timeframe_hours}H")
    print(f"Start:      {start_date}")
    print(f"End:        {end_date}")
    print(f"{'=' * 60}")

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Generate timestamps
    timestamps = []
    current = start
    while current < end:
        timestamps.append(current)
        current += timedelta(hours=timeframe_hours)

    n_bars = len(timestamps)
    print(f"\nGenerating {n_bars} bars...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Historical ETH price milestones (approximate)
    # 2019-01: ~$130, 2020-01: ~$130, 2021-01: ~$730
    # 2021-11: ~$4800 (ATH), 2022-06: ~$1000, 2023-01: ~$1200
    # 2024-01: ~$2300, 2024-12: ~$3500

    # Start price
    price = 130.0

    # Generate price series with regime changes
    prices = []
    volumes = []

    # Define regimes (approximate date ranges and drift/volatility)
    regimes = [
        # (end_date, daily_drift, daily_volatility, base_volume)
        ("2019-07-01", 0.002, 0.05, 500000),   # Early 2019, slight up
        ("2020-01-01", -0.001, 0.04, 400000),  # Late 2019, consolidation
        ("2020-03-15", -0.01, 0.08, 800000),   # COVID crash
        ("2020-12-31", 0.003, 0.04, 600000),   # Recovery
        ("2021-05-15", 0.008, 0.06, 1500000),  # Bull run to ATH
        ("2021-07-01", -0.005, 0.07, 1000000), # Summer correction
        ("2021-11-15", 0.006, 0.05, 1200000),  # New ATH
        ("2022-06-01", -0.004, 0.06, 900000),  # Bear market
        ("2022-12-31", -0.001, 0.04, 500000),  # Consolidation
        ("2023-06-01", 0.002, 0.04, 600000),   # Slow recovery
        ("2023-12-31", 0.003, 0.05, 700000),   # Continued recovery
        ("2024-06-01", 0.004, 0.05, 900000),   # Bull momentum
        ("2024-12-31", 0.003, 0.04, 1000000),  # Consolidation
        ("2026-01-01", 0.002, 0.04, 800000),   # Future projection
    ]

    regime_idx = 0
    regime_end = datetime.strptime(regimes[0][0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    daily_drift = regimes[0][1]
    daily_vol = regimes[0][2]
    base_volume = regimes[0][3]

    # Adjust for 4H timeframe (6 bars per day)
    bars_per_day = 24 / timeframe_hours
    drift = daily_drift / bars_per_day
    vol = daily_vol / np.sqrt(bars_per_day)

    for i, ts in enumerate(timestamps):
        # Check for regime change
        while ts >= regime_end and regime_idx < len(regimes) - 1:
            regime_idx += 1
            regime_end = datetime.strptime(regimes[regime_idx][0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            daily_drift = regimes[regime_idx][1]
            daily_vol = regimes[regime_idx][2]
            base_volume = regimes[regime_idx][3]
            drift = daily_drift / bars_per_day
            vol = daily_vol / np.sqrt(bars_per_day)

        # Generate return using GBM
        ret = drift + vol * np.random.randn()
        price = price * (1 + ret)

        # Ensure minimum price
        price = max(price, 50)

        prices.append(price)

        # Generate volume with some randomness
        vol_mult = np.exp(0.5 * np.random.randn())
        volumes.append(base_volume * vol_mult)

        # Progress update
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{n_bars} bars, price: ${price:.2f}")

    # Convert to numpy arrays
    closes = np.array(prices)

    # Generate OHLC from close prices
    # Add some intra-bar volatility
    intra_vol = 0.01  # 1% typical range

    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    # High and low with some randomness
    range_mult = np.abs(np.random.randn(n_bars)) * intra_vol + intra_vol
    highs = np.maximum(opens, closes) * (1 + range_mult)
    lows = np.minimum(opens, closes) * (1 - range_mult)

    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    print(f"\n{'=' * 60}")
    print(f"Generation Complete!")
    print(f"{'=' * 60}")
    print(f"Total bars: {len(df)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"{'=' * 60}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save files
    filename_base = f"ETHUSDT_4h_{start_date}_{end_date}"

    # Save as CSV
    csv_path = output_path / f"{filename_base}.csv"
    df.to_csv(csv_path)
    print(f"\nSaved CSV:     {csv_path}")

    # Save as Parquet
    try:
        parquet_path = output_path / f"{filename_base}.parquet"
        df.to_parquet(parquet_path)
        print(f"Saved Parquet: {parquet_path}")
    except Exception as e:
        print(f"Could not save Parquet: {e}")

    # Print statistics
    print(f"\nData Statistics:")
    print(f"  Mean Close:  ${df['close'].mean():.2f}")
    print(f"  Std Close:   ${df['close'].std():.2f}")
    print(f"  Min Close:   ${df['close'].min():.2f}")
    print(f"  Max Close:   ${df['close'].max():.2f}")
    print(f"  Mean Volume: {df['volume'].mean():,.0f}")

    return df


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Generate ETH 4H data
    df = generate_eth_data(
        start_date="2019-01-01",
        end_date="2026-01-01",
        timeframe_hours=4,
        output_dir="data",
    )

    return df


if __name__ == "__main__":
    main()
