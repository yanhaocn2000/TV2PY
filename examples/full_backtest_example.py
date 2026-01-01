#!/usr/bin/env python3
"""
Example: Complete Backtest Workflow

This example demonstrates a complete backtesting workflow including:
1. Downloading data
2. Setting up the backtest engine
3. Running the backtest
4. Analyzing results
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_data() -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    import numpy as np

    np.random.seed(42)

    # Generate 1000 bars of sample data
    n_bars = 1000
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Random walk for price
    returns = np.random.randn(n_bars) * 0.002
    price = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = {
        "timestamp": dates,
        "open": price * (1 + np.random.randn(n_bars) * 0.001),
        "high": price * (1 + abs(np.random.randn(n_bars) * 0.003)),
        "low": price * (1 - abs(np.random.randn(n_bars) * 0.003)),
        "close": price,
        "volume": np.random.randint(1000, 10000, n_bars),
    }

    df = pd.DataFrame(data)
    df = df.set_index("timestamp")

    return df


def run_complete_backtest():
    """Run a complete backtest example."""

    print("=" * 70)
    print("                    COMPLETE BACKTEST EXAMPLE")
    print("=" * 70)
    print()

    # Step 1: Create or load data
    print("Step 1: Loading market data...")
    print("-" * 40)

    df = create_sample_data()
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")
    print()

    # Save sample data
    data_path = Path(__file__).parent.parent / "data" / "sample_data.csv"
    df.to_csv(data_path)
    print(f"Sample data saved to: {data_path}")
    print()

    # Step 2: Configure backtest
    print("Step 2: Configuring backtest...")
    print("-" * 40)

    config = {
        "venue": "BINANCE",
        "instrument": "BTCUSDT",
        "strategy": "EMACross",
        "params": {
            "fast_ema_period": 10,
            "slow_ema_period": 20,
            "trade_size": 0.1,
        },
        "starting_balance": 10000,
    }

    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Step 3: Run backtest (simulated results)
    print("Step 3: Running backtest...")
    print("-" * 40)

    # Simulate equity curve based on strategy logic
    import numpy as np

    # Simple simulation: EMA crossover on the sample data
    fast_ema = df["close"].ewm(span=config["params"]["fast_ema_period"]).mean()
    slow_ema = df["close"].ewm(span=config["params"]["slow_ema_period"]).mean()

    # Generate signals
    signal = (fast_ema > slow_ema).astype(int)
    signal_diff = signal.diff()

    # Calculate returns (simplified)
    returns = df["close"].pct_change()
    strategy_returns = signal.shift(1) * returns

    # Build equity curve
    starting_capital = config["starting_balance"]
    equity_curve = starting_capital * (1 + strategy_returns).cumprod()
    equity_curve = equity_curve.fillna(starting_capital)

    print(f"Backtest completed!")
    print(f"Total bars processed: {len(df)}")
    print()

    # Step 4: Analyze results
    print("Step 4: Analyzing results...")
    print("-" * 40)

    from src.utils.analyzer import PerformanceAnalyzer

    # Create trades DataFrame (simplified)
    entries = df.index[signal_diff == 1].tolist()
    exits = df.index[signal_diff == -1].tolist()

    # Match entries with exits
    trades_list = []
    for i, entry in enumerate(entries):
        # Find next exit after entry
        future_exits = [e for e in exits if e > entry]
        if future_exits:
            exit_time = future_exits[0]
            entry_price = df.loc[entry, "close"]
            exit_price = df.loc[exit_time, "close"]
            pnl = (exit_price - entry_price) * config["params"]["trade_size"]
            trades_list.append({
                "entry_time": entry,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            })

    trades_df = pd.DataFrame(trades_list)

    # Analyze
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.analyze_returns(equity_curve, trades_df)

    # Print report
    report = analyzer.generate_report(metrics)
    print(report)

    # Step 5: Summary
    print("Step 5: Summary")
    print("-" * 40)

    final_value = equity_curve.iloc[-1]
    print(f"Starting Capital:  ${starting_capital:,.2f}")
    print(f"Final Value:       ${final_value:,.2f}")
    print(f"Total Return:      {metrics.total_return:+.2%}")
    print(f"Total Trades:      {metrics.total_trades}")
    print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
    print()

    print("=" * 70)
    print("Backtest complete! This example demonstrates the workflow.")
    print("For real backtesting, use NautilusTrader's BacktestEngine.")
    print("=" * 70)


if __name__ == "__main__":
    run_complete_backtest()
