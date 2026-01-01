"""
TV2PY - TradingView Pine Script to Python conversion library.

This library provides Python implementations of TradingView's Pine Script
strategy functions with exact algorithmic matching for consistent backtesting
results between TradingView and Python-based backtesting engines like Nautilus.
"""

from .strategy_exit import StrategyExit, ExitOrder, ExitType, PositionSide

__version__ = "0.1.0"
__all__ = ["StrategyExit", "ExitOrder", "ExitType", "PositionSide"]
