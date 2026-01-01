#!/usr/bin/env python3
"""
Example: Strategy Optimization using Grid Search

This example demonstrates how to optimize strategy parameters
to find the best configuration.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.optimizer import GridSearchOptimizer


def run_optimization_example():
    """
    Example of grid search optimization.

    In a real scenario, you would:
    1. Load historical data
    2. Define a backtest function that runs NautilusTrader
    3. Run the optimizer
    """

    # Define parameter grid
    param_grid = {
        "fast_ema_period": [5, 10, 15, 20],
        "slow_ema_period": [20, 30, 50, 100],
    }

    # Create optimizer
    optimizer = GridSearchOptimizer(
        param_grid=param_grid,
        objective="sharpe_ratio",
        maximize=True,
    )

    print("Grid Search Optimization Example")
    print("=" * 60)
    print(f"Parameters to optimize: {list(param_grid.keys())}")
    print(f"Total combinations: {len(param_grid['fast_ema_period']) * len(param_grid['slow_ema_period'])}")
    print()

    # In a real scenario, you would define a backtest function like this:
    """
    def run_backtest(params):
        # Configure and run NautilusTrader backtest
        # Return (equity_curve_series, trades_dataframe)

        engine = BacktestEngine(config=...)
        engine.add_venue(...)
        engine.add_instrument(instrument)
        engine.add_data(bars)

        strategy_config = EMACrossConfig(
            instrument_id=instrument.id,
            bar_type=bar_type,
            fast_ema_period=params['fast_ema_period'],
            slow_ema_period=params['slow_ema_period'],
            trade_size=Decimal("0.1"),
        )
        engine.add_strategy(EMACrossStrategy(config=strategy_config))

        engine.run()

        # Extract results
        equity_curve = pd.Series(...)  # From engine results
        trades = pd.DataFrame(...)  # From engine results

        engine.reset()
        engine.dispose()

        return equity_curve, trades

    # Run optimization
    best_result = optimizer.optimize(run_backtest, verbose=True)

    # Get all results as DataFrame
    results_df = optimizer.get_results_dataframe()
    print("\nAll Results:")
    print(results_df)

    # Get top 5 results
    top_5 = optimizer.get_top_n(5)
    print("\nTop 5 Configurations:")
    for i, result in enumerate(top_5, 1):
        print(f"{i}. {result.params} -> Sharpe: {result.score:.4f}")
    """

    print("To run actual optimization:")
    print("1. Load your historical data")
    print("2. Define a run_backtest() function that returns (equity_curve, trades)")
    print("3. Call optimizer.optimize(run_backtest)")
    print()
    print("See the docstring in this file for a complete example.")


if __name__ == "__main__":
    run_optimization_example()
