"""
Grid Trading Strategy.

Places buy and sell orders at regular price intervals (grid levels).
Profits from price oscillations in ranging markets.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


class GridTradingConfig(StrategyConfig, frozen=True):
    """Configuration for Grid Trading Strategy."""

    instrument_id: InstrumentId
    bar_type: BarType
    grid_levels: int = 10           # Number of grid levels above and below
    grid_spacing_pct: float = 1.0   # Spacing between levels in percent
    base_price: float | None = None # Center price (None = use first bar)
    trade_size: Decimal = Decimal("0.1")
    order_id_tag: str = "001"


class GridTradingStrategy(Strategy):
    """
    Grid trading strategy.

    Creates a grid of price levels and trades when price crosses each level.
    Buy when price drops to a lower level, sell when price rises to upper level.
    """

    def __init__(self, config: GridTradingConfig) -> None:
        super().__init__(config)

        # Configuration
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size
        self.grid_levels = config.grid_levels
        self.grid_spacing_pct = config.grid_spacing_pct / 100.0
        self.base_price = config.base_price

        # State
        self.instrument: Instrument | None = None
        self._grid: list[float] = []
        self._current_level: int = 0
        self._initialized: bool = False

    def on_start(self) -> None:
        """Called when the strategy is started."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument {self.instrument_id}")
            self.stop()
            return

        self.subscribe_bars(self.bar_type)
        self.log.info(f"Grid Trading Strategy started for {self.instrument_id}")

    def _initialize_grid(self, price: float) -> None:
        """Initialize the grid levels centered around the given price."""
        self.base_price = price
        self._grid = []

        # Create grid levels
        for i in range(-self.grid_levels, self.grid_levels + 1):
            level_price = self.base_price * (1 + i * self.grid_spacing_pct)
            self._grid.append(level_price)

        # Sort grid levels
        self._grid.sort()

        # Find current level (closest level below current price)
        self._current_level = 0
        for i, level in enumerate(self._grid):
            if level <= price:
                self._current_level = i

        self._initialized = True

        self.log.info(f"Grid initialized with {len(self._grid)} levels")
        self.log.info(f"Base price: {self.base_price:.2f}")
        self.log.info(f"Grid range: {self._grid[0]:.2f} to {self._grid[-1]:.2f}")
        self.log.info(f"Current level: {self._current_level}")

    def on_bar(self, bar: Bar) -> None:
        """Called when a bar is received."""
        current_price = float(bar.close)

        # Initialize grid on first bar if no base price set
        if not self._initialized:
            if self.base_price is None:
                self._initialize_grid(current_price)
            else:
                self._initialize_grid(self.base_price)
            return

        # Find the level for current price
        new_level = 0
        for i, level in enumerate(self._grid):
            if level <= current_price:
                new_level = i

        # Check for level changes
        if new_level > self._current_level:
            # Price moved up - crossed upper grid levels
            levels_crossed = new_level - self._current_level
            self.log.info(f"Price crossed {levels_crossed} level(s) UP")

            # Sell at each crossed level
            for _ in range(levels_crossed):
                self._sell()

        elif new_level < self._current_level:
            # Price moved down - crossed lower grid levels
            levels_crossed = self._current_level - new_level
            self.log.info(f"Price crossed {levels_crossed} level(s) DOWN")

            # Buy at each crossed level
            for _ in range(levels_crossed):
                self._buy()

        self._current_level = new_level

    def _buy(self) -> None:
        """Execute a buy order."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Grid BUY executed")

    def _sell(self) -> None:
        """Execute a sell order."""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)
        self.log.info(f"Grid SELL executed")

    def on_stop(self) -> None:
        """Called when the strategy is stopped."""
        # Close all positions when stopping
        self.close_all_positions(self.instrument_id)

    def on_reset(self) -> None:
        """Called when the strategy is reset."""
        self._grid = []
        self._current_level = 0
        self._initialized = False

    def on_data(self, data: Data) -> None:
        pass
