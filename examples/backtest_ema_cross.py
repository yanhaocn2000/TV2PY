#!/usr/bin/env python3
"""
Example: EMA Cross Strategy Backtest

This example demonstrates how to run a backtest using the EMA Cross strategy
with simulated cryptocurrency data.
"""

import sys
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from src.strategies.ema_cross import EMACrossConfig, EMACrossStrategy


def run_backtest():
    """Run the EMA Cross backtest example."""

    # Configure the backtest engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(log_level="INFO"),
    )

    # Create the backtest engine
    engine = BacktestEngine(config=engine_config)

    # Define venue
    BINANCE = Venue("BINANCE")

    # Add a simulated venue with starting capital
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        base_currency=None,  # Multi-currency account
        starting_balances=[Money(10_000, USDT)],
    )

    # Create instrument (ETH/USDT perpetual future)
    instrument = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(instrument)

    # Configure strategy
    bar_type = BarType.from_str("ETHUSDT.BINANCE-1-MINUTE-LAST-INTERNAL")

    strategy_config = EMACrossConfig(
        instrument_id=instrument.id,
        bar_type=bar_type,
        fast_ema_period=10,
        slow_ema_period=20,
        trade_size=Decimal("0.1"),
    )

    strategy = EMACrossStrategy(config=strategy_config)
    engine.add_strategy(strategy)

    # Note: In a real backtest, you would add historical data here
    # engine.add_data(bars)

    print("=" * 60)
    print("EMA Cross Strategy Backtest")
    print("=" * 60)
    print(f"Instrument: {instrument.id}")
    print(f"Bar Type: {bar_type}")
    print(f"Fast EMA: {strategy_config.fast_ema_period}")
    print(f"Slow EMA: {strategy_config.slow_ema_period}")
    print(f"Trade Size: {strategy_config.trade_size}")
    print("=" * 60)

    # Run the backtest
    # engine.run()

    # Generate reports
    # print("\nAccount Report:")
    # print(engine.trader.generate_account_report(BINANCE))

    # print("\nOrder Fills Report:")
    # print(engine.trader.generate_order_fills_report())

    # print("\nPositions Report:")
    # print(engine.trader.generate_positions_report())

    # Cleanup
    engine.reset()
    engine.dispose()

    print("\nBacktest engine initialized successfully!")
    print("Add your historical data and uncomment engine.run() to execute.")


if __name__ == "__main__":
    run_backtest()
