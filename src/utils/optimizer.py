"""
Strategy Optimizer using Grid Search.

Provides functionality to optimize strategy parameters through backtesting.
"""

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.utils.analyzer import PerformanceAnalyzer, PerformanceMetrics


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""

    params: dict[str, Any]
    metrics: PerformanceMetrics
    score: float


class GridSearchOptimizer:
    """
    Grid search optimizer for trading strategies.

    Exhaustively searches through all parameter combinations
    to find the optimal configuration.
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        objective: str = "sharpe_ratio",
        maximize: bool = True,
    ):
        """
        Initialize the optimizer.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
            objective: Metric to optimize (e.g., 'sharpe_ratio', 'total_return').
            maximize: Whether to maximize (True) or minimize (False) the objective.
        """
        self.param_grid = param_grid
        self.objective = objective
        self.maximize = maximize
        self.results: list[OptimizationResult] = []
        self.analyzer = PerformanceAnalyzer()

    def _generate_param_combinations(self) -> list[dict[str, Any]]:
        """Generate all combinations of parameters."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def optimize(
        self,
        run_backtest: Callable[[dict[str, Any]], tuple[pd.Series, pd.DataFrame | None]],
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Run grid search optimization.

        Args:
            run_backtest: Function that takes params dict and returns
                         (equity_curve, trades_df) tuple.
            verbose: Print progress information.

        Returns:
            Best OptimizationResult found.
        """
        combinations = self._generate_param_combinations()
        total = len(combinations)

        if verbose:
            print(f"Starting grid search with {total} parameter combinations...")
            print(f"Optimizing: {self.objective} ({'max' if self.maximize else 'min'})")
            print("-" * 60)

        self.results = []

        for i, params in enumerate(combinations):
            if verbose:
                print(f"[{i + 1}/{total}] Testing: {params}")

            try:
                equity_curve, trades = run_backtest(params)
                metrics = self.analyzer.analyze_returns(equity_curve, trades)

                # Get the objective value
                score = getattr(metrics, self.objective)

                result = OptimizationResult(
                    params=params,
                    metrics=metrics,
                    score=score,
                )
                self.results.append(result)

                if verbose:
                    print(f"         {self.objective}: {score:.4f}")

            except Exception as e:
                if verbose:
                    print(f"         ERROR: {e}")
                continue

        if not self.results:
            raise ValueError("No successful optimization runs")

        # Find best result
        if self.maximize:
            best = max(self.results, key=lambda r: r.score)
        else:
            best = min(self.results, key=lambda r: r.score)

        if verbose:
            print("-" * 60)
            print(f"Best parameters: {best.params}")
            print(f"Best {self.objective}: {best.score:.4f}")

        return best

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as a DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = []
        for result in self.results:
            row = result.params.copy()
            row["sharpe_ratio"] = result.metrics.sharpe_ratio
            row["total_return"] = result.metrics.total_return
            row["max_drawdown"] = result.metrics.max_drawdown
            row["win_rate"] = result.metrics.win_rate
            row["profit_factor"] = result.metrics.profit_factor
            row["total_trades"] = result.metrics.total_trades
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by objective
        df = df.sort_values(self.objective, ascending=not self.maximize)

        return df

    def get_top_n(self, n: int = 10) -> list[OptimizationResult]:
        """Get top N results."""
        if not self.results:
            return []

        sorted_results = sorted(
            self.results,
            key=lambda r: r.score,
            reverse=self.maximize,
        )

        return sorted_results[:n]


class WalkForwardOptimizer:
    """
    Walk-forward optimization for more robust parameter selection.

    Divides data into training and testing periods, optimizes on training,
    validates on testing, then moves forward.
    """

    def __init__(
        self,
        param_grid: dict[str, list[Any]],
        train_period: int,  # days
        test_period: int,   # days
        objective: str = "sharpe_ratio",
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values.
            train_period: Number of days for training period.
            test_period: Number of days for testing period.
            objective: Metric to optimize.
        """
        self.param_grid = param_grid
        self.train_period = train_period
        self.test_period = test_period
        self.objective = objective
        self.walk_forward_results: list[dict[str, Any]] = []

    def optimize(
        self,
        data: pd.DataFrame,
        run_backtest: Callable[
            [dict[str, Any], pd.DataFrame],
            tuple[pd.Series, pd.DataFrame | None]
        ],
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run walk-forward optimization.

        Args:
            data: Full historical data with datetime index.
            run_backtest: Function that takes (params, data_slice) and returns
                         (equity_curve, trades_df) tuple.
            verbose: Print progress information.

        Returns:
            Summary of walk-forward results.
        """
        total_days = (data.index[-1] - data.index[0]).days
        window_size = self.train_period + self.test_period

        if total_days < window_size:
            raise ValueError(
                f"Not enough data: {total_days} days < {window_size} days required"
            )

        self.walk_forward_results = []
        current_start = data.index[0]

        period = 1
        while True:
            train_end = current_start + pd.Timedelta(days=self.train_period)
            test_end = train_end + pd.Timedelta(days=self.test_period)

            if test_end > data.index[-1]:
                break

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Walk-Forward Period {period}")
                print(f"Train: {current_start.date()} to {train_end.date()}")
                print(f"Test:  {train_end.date()} to {test_end.date()}")
                print("=" * 60)

            # Get data slices
            train_data = data[current_start:train_end]
            test_data = data[train_end:test_end]

            # Optimize on training data
            grid_optimizer = GridSearchOptimizer(
                param_grid=self.param_grid,
                objective=self.objective,
                maximize=True,
            )

            def train_backtest(params):
                return run_backtest(params, train_data)

            best_train = grid_optimizer.optimize(train_backtest, verbose=False)

            if verbose:
                print(f"Best train params: {best_train.params}")
                print(f"Train {self.objective}: {best_train.score:.4f}")

            # Validate on test data
            test_equity, test_trades = run_backtest(best_train.params, test_data)
            analyzer = PerformanceAnalyzer()
            test_metrics = analyzer.analyze_returns(test_equity, test_trades)

            test_score = getattr(test_metrics, self.objective)

            if verbose:
                print(f"Test {self.objective}: {test_score:.4f}")

            self.walk_forward_results.append({
                "period": period,
                "train_start": current_start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "best_params": best_train.params,
                "train_score": best_train.score,
                "test_score": test_score,
                "train_metrics": best_train.metrics,
                "test_metrics": test_metrics,
            })

            # Move forward
            current_start = train_end
            period += 1

        # Summary
        if self.walk_forward_results:
            train_scores = [r["train_score"] for r in self.walk_forward_results]
            test_scores = [r["test_score"] for r in self.walk_forward_results]

            summary = {
                "total_periods": len(self.walk_forward_results),
                "avg_train_score": sum(train_scores) / len(train_scores),
                "avg_test_score": sum(test_scores) / len(test_scores),
                "robustness_ratio": (
                    sum(test_scores) / sum(train_scores)
                    if sum(train_scores) != 0 else 0
                ),
            }

            if verbose:
                print(f"\n{'=' * 60}")
                print("WALK-FORWARD SUMMARY")
                print("=" * 60)
                print(f"Total Periods: {summary['total_periods']}")
                print(f"Avg Train {self.objective}: {summary['avg_train_score']:.4f}")
                print(f"Avg Test {self.objective}: {summary['avg_test_score']:.4f}")
                print(f"Robustness Ratio: {summary['robustness_ratio']:.2%}")

            return summary

        return {}
