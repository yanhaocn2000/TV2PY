# TV2PY

TradingView Pine Script to Python conversion library with exact algorithm matching for consistent backtesting results between TradingView and Python-based backtesting engines (like Nautilus Trader).

## Features

- **Exact Algorithm Matching**: Implements TradingView's broker emulator behavior precisely
- **strategy.exit() Implementation**: Full support for all exit order types
  - Take profit / Limit orders
  - Stop loss orders
  - Trailing stop orders with activation conditions
  - Partial exits (by quantity or percentage)
  - OCA (One-Cancels-All) order groups
- **Intrabar Fill Simulation**: Matches TradingView's price path assumption
- **Pine Script v6 Behavior**: Uses whichever price level triggers first when both absolute and relative parameters are specified

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from tv2py import StrategyExit, PositionSide, BarData

# Create strategy exit manager
strategy = StrategyExit(tick_size=0.01)

# Create an exit order (equivalent to Pine Script strategy.exit())
order = strategy.exit(
    id="exit1",
    from_entry="entry1",
    position_side=PositionSide.LONG,
    entry_price=100.0,
    position_qty=10.0,
    profit=500,        # Take profit at 500 ticks (5.0) above entry
    loss=250,          # Stop loss at 250 ticks (2.5) below entry
    trail_points=300,  # Activate trailing at 300 ticks profit
    trail_offset=100,  # Trail 100 ticks (1.0) behind best price
)

# Process bars
bar = BarData(open=101.0, high=106.0, low=100.5, close=105.0)
results = strategy.process_bar(bar)

for result in results:
    print(f"Filled: {result.exit_type.value} at {result.fill_price}")
```

## strategy.exit() Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | str | Required. Order identifier |
| `from_entry` | str | Optional. Entry order identifier |
| `position_side` | PositionSide | LONG or SHORT |
| `entry_price` | float | Entry price of the position |
| `position_qty` | float | Total position quantity |
| `qty` | float | Number of units to exit |
| `qty_percent` | float | Percentage of position to exit (0-100) |
| `profit` | float | Profit target in ticks |
| `limit` | float | Profit target as absolute price |
| `loss` | float | Stop loss in ticks |
| `stop` | float | Stop loss as absolute price |
| `trail_price` | float | Trailing stop activation price |
| `trail_points` | float | Trailing stop activation in ticks profit (0 = immediate) |
| `trail_offset` | float | Trailing stop distance in ticks |
| `oca_name` | str | OCA group name |
| `comment` | str | Default order comment |
| `comment_profit` | str | Comment for profit exit |
| `comment_loss` | str | Comment for loss exit |
| `comment_trailing` | str | Comment for trailing stop exit |
| `when` | bool | Condition to place order |
| `alert_message` | str | Default alert message |
| `alert_profit` | str | Alert for profit exit |
| `alert_loss` | str | Alert for loss exit |
| `alert_trailing` | str | Alert for trailing stop exit |

## Key Algorithms

### Intrabar Price Path

TradingView's broker emulator assumes prices move in a specific order within each bar:

- If `|high - open| <= |low - open|`: Open → High → Low → Close
- If `|low - open| < |high - open|`: Open → Low → High → Close

This determines which order gets filled first when both profit and stop could be hit in the same bar.

### Trailing Stop

1. **Activation**: The trailing stop activates when price reaches either:
   - `trail_price` (absolute price level), OR
   - `entry_price + trail_points * tick_size` (for longs)
   - Whichever triggers first

2. **Trail Calculation**:
   - For LONG: `stop_price = best_price - (trail_offset * tick_size)`
   - For SHORT: `stop_price = best_price + (trail_offset * tick_size)`
   - The stop only moves in the favorable direction, never backwards

3. **Immediate Activation**: Set `trail_points=0` to activate immediately at entry

### Profit/Loss Price Selection (Pine Script v6+)

When both absolute and relative parameters are specified:
- For profit: Uses whichever limit price triggers first
  - LONG: `min(limit, entry_price + profit * tick_size)`
  - SHORT: `max(limit, entry_price - profit * tick_size)`
- For loss: Uses whichever stop price triggers first
  - LONG: `max(stop, entry_price - loss * tick_size)`
  - SHORT: `min(stop, entry_price + loss * tick_size)`

## Usage with Nautilus Trader

```python
from nautilus_trader.model.events import OrderFilled
from tv2py import StrategyExit, PositionSide, BarData

class MyStrategy:
    def __init__(self):
        self.exit_manager = StrategyExit(tick_size=instrument.price_increment)

    def on_order_filled(self, event: OrderFilled):
        if event.order_side == OrderSide.BUY:
            # Create exit orders
            self.exit_manager.exit(
                id=f"exit_{event.client_order_id}",
                position_side=PositionSide.LONG,
                entry_price=float(event.last_px),
                position_qty=float(event.last_qty),
                profit=100,
                loss=50,
            )

    def on_bar(self, bar: Bar):
        bar_data = BarData(
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
        )
        results = self.exit_manager.process_bar(bar_data)

        for result in results:
            # Submit actual exit order to Nautilus
            self.submit_exit_order(result)
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## References

- [TradingView Pine Script Reference - strategy.exit()](https://www.tradingview.com/pine-script-reference/v5/#fun_strategy.exit)
- [TradingView Strategies Documentation](https://www.tradingview.com/pine-script-docs/concepts/strategies/)
- [TradingCode - strategy.exit() Function](https://www.tradingcode.net/tradingview/strategy-exit-function/)

## License

MIT
