"""
Performance Analyzer for backtesting results.

Provides comprehensive analysis of trading strategy performance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    # Returns
    total_return: float
    annual_return: float
    monthly_return: float

    # Risk
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # in days

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float  # in hours

    # Additional
    expectancy: float
    recovery_factor: float


class PerformanceAnalyzer:
    """Analyze backtesting performance results."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
        """
        self.risk_free_rate = risk_free_rate

    def analyze_returns(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame | None = None,
    ) -> PerformanceMetrics:
        """
        Analyze performance from equity curve and trades.

        Args:
            equity_curve: Series of portfolio values indexed by datetime.
            trades: DataFrame with trade information (optional).

        Returns:
            PerformanceMetrics object with all calculated metrics.
        """
        returns = equity_curve.pct_change().dropna()

        # Calculate return metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized metrics
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        monthly_return = (1 + annual_return) ** (1 / 12) - 1

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        max_dd, max_dd_duration = self._calculate_max_drawdown(equity_curve)

        # Risk-adjusted returns
        sharpe = self._calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = self._calculate_sortino_ratio(returns, self.risk_free_rate)
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        trade_stats = self._calculate_trade_stats(trades) if trades is not None else {}

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_return,
            volatility=volatility,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=trade_stats.get("total_trades", 0),
            winning_trades=trade_stats.get("winning_trades", 0),
            losing_trades=trade_stats.get("losing_trades", 0),
            win_rate=trade_stats.get("win_rate", 0),
            profit_factor=trade_stats.get("profit_factor", 0),
            avg_win=trade_stats.get("avg_win", 0),
            avg_loss=trade_stats.get("avg_loss", 0),
            largest_win=trade_stats.get("largest_win", 0),
            largest_loss=trade_stats.get("largest_loss", 0),
            avg_trade_duration=trade_stats.get("avg_trade_duration", 0),
            expectancy=trade_stats.get("expectancy", 0),
            recovery_factor=abs(total_return / max_dd) if max_dd != 0 else 0,
        )

    def _calculate_max_drawdown(
        self, equity_curve: pd.Series
    ) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max

        max_dd = drawdown.min()

        # Calculate duration
        in_drawdown = drawdown < 0
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

        return max_dd, max_dd_duration

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float
    ) -> float:
        """Calculate annualized Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

    def _calculate_trade_stats(self, trades: pd.DataFrame) -> dict[str, Any]:
        """Calculate trading statistics from trade DataFrame."""
        if trades is None or len(trades) == 0:
            return {}

        # Assuming trades has columns: pnl, entry_time, exit_time
        pnl = trades.get("pnl", pd.Series())
        if len(pnl) == 0:
            return {}

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        total_trades = len(pnl)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Calculate trade duration if timestamps available
        avg_duration = 0
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            durations = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 3600
            avg_duration = durations.mean()

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": wins.max() if len(wins) > 0 else 0,
            "largest_loss": losses.min() if len(losses) > 0 else 0,
            "avg_trade_duration": avg_duration,
            "expectancy": expectancy,
        }

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a text report of performance metrics."""
        report = """
================================================================================
                         BACKTEST PERFORMANCE REPORT
================================================================================

RETURNS
-------
  Total Return:      {total_return:>10.2%}
  Annual Return:     {annual_return:>10.2%}
  Monthly Return:    {monthly_return:>10.2%}

RISK
----
  Volatility (Ann.): {volatility:>10.2%}
  Max Drawdown:      {max_drawdown:>10.2%}
  Max DD Duration:   {max_dd_duration:>10} days

RISK-ADJUSTED RETURNS
---------------------
  Sharpe Ratio:      {sharpe:>10.2f}
  Sortino Ratio:     {sortino:>10.2f}
  Calmar Ratio:      {calmar:>10.2f}

TRADING STATISTICS
------------------
  Total Trades:      {total_trades:>10}
  Winning Trades:    {winning_trades:>10}
  Losing Trades:     {losing_trades:>10}
  Win Rate:          {win_rate:>10.2%}
  Profit Factor:     {profit_factor:>10.2f}

  Avg Win:           {avg_win:>10.2f}
  Avg Loss:          {avg_loss:>10.2f}
  Largest Win:       {largest_win:>10.2f}
  Largest Loss:      {largest_loss:>10.2f}

  Expectancy:        {expectancy:>10.2f}
  Recovery Factor:   {recovery_factor:>10.2f}

================================================================================
""".format(
            total_return=metrics.total_return,
            annual_return=metrics.annual_return,
            monthly_return=metrics.monthly_return,
            volatility=metrics.volatility,
            max_drawdown=metrics.max_drawdown,
            max_dd_duration=metrics.max_drawdown_duration,
            sharpe=metrics.sharpe_ratio,
            sortino=metrics.sortino_ratio,
            calmar=metrics.calmar_ratio,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss,
            largest_win=metrics.largest_win,
            largest_loss=metrics.largest_loss,
            expectancy=metrics.expectancy,
            recovery_factor=metrics.recovery_factor,
        )
        return report
