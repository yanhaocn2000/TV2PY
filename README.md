# TV2PY - TradingView to Python Backtesting

A high-performance backtesting framework powered by [NautilusTrader](https://github.com/nautechsystems/nautilus_trader).

## Features

- Event-driven backtesting engine
- High-performance Rust/Cython core
- Support for multiple asset classes (Crypto, FX, Stocks)
- Built-in technical indicators
- Strategy optimization with grid search
- Live trading ready (same code for backtest and live)

## Requirements

- Python 3.12 - 3.14
- pip or uv package manager

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
TV2PY/
├── src/
│   ├── strategies/      # Custom trading strategies
│   ├── data/            # Data loaders and wranglers
│   └── utils/           # Utility functions
├── examples/            # Example backtests
├── configs/             # Configuration files
├── data/                # Historical data storage
└── notebooks/           # Jupyter notebooks for analysis
```

## Quick Start

```python
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig

# Create backtest engine
engine = BacktestEngine(config=BacktestEngineConfig(
    trader_id="BACKTESTER-001",
))

# Add your strategy and run backtest
# See examples/ for complete examples
```

## Documentation

- [NautilusTrader Docs](https://nautilustrader.io/docs/latest/)
- [Backtest High-Level API](https://nautilustrader.io/docs/latest/getting_started/backtest_high_level/)
- [Backtest Low-Level API](https://nautilustrader.io/docs/latest/getting_started/backtest_low_level/)

## License

MIT
