"""
Comprehensive tests for TradingView strategy.exit() Python implementation.

These tests verify that the Python implementation matches TradingView's
behavior exactly for consistent backtesting results.
"""

import pytest
from tv2py.strategy_exit import (
    StrategyExit,
    ExitOrder,
    ExitType,
    PositionSide,
    BarData,
    OrderStatus,
    price_to_ticks,
    ticks_to_price,
    percent_to_ticks,
)


class TestBarData:
    """Tests for BarData intrabar path calculation."""

    def test_intrabar_path_high_closer_to_open(self):
        """When high is closer to open: O → H → L → C."""
        bar = BarData(open=100.0, high=101.0, low=98.0, close=99.0)
        # dist_to_high = 1, dist_to_low = 2
        # high is closer, so O → H → L → C
        path = bar.get_intrabar_path()
        assert path == [100.0, 101.0, 98.0, 99.0]

    def test_intrabar_path_low_closer_to_open(self):
        """When low is closer to open: O → L → H → C."""
        bar = BarData(open=100.0, high=103.0, low=99.0, close=101.0)
        # dist_to_high = 3, dist_to_low = 1
        # low is closer, so O → L → H → C
        path = bar.get_intrabar_path()
        assert path == [100.0, 99.0, 103.0, 101.0]

    def test_intrabar_path_equal_distance(self):
        """When equal distance, high path is used."""
        bar = BarData(open=100.0, high=102.0, low=98.0, close=100.0)
        # dist_to_high = 2, dist_to_low = 2
        # Equal, defaults to O → H → L → C
        path = bar.get_intrabar_path()
        assert path == [100.0, 102.0, 98.0, 100.0]


class TestExitOrderBasics:
    """Tests for ExitOrder basic functionality."""

    def test_exit_order_creation(self):
        """Test basic exit order creation."""
        order = ExitOrder(
            id="exit1",
            from_entry="entry1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            tick_size=0.01,
        )
        assert order.id == "exit1"
        assert order.from_entry == "entry1"
        assert order.position_side == PositionSide.LONG
        assert order.status == OrderStatus.PENDING

    def test_qty_calculation_absolute(self):
        """Test exit quantity with absolute qty."""
        order = ExitOrder(
            id="exit1",
            position_qty=10.0,
            qty=5.0,
            tick_size=0.01,
        )
        assert order.get_exit_qty() == 5.0

    def test_qty_calculation_percent(self):
        """Test exit quantity with qty_percent."""
        order = ExitOrder(
            id="exit1",
            position_qty=100.0,
            qty_percent=50.0,
            tick_size=0.01,
        )
        assert order.get_exit_qty() == 50.0

    def test_qty_calculation_full_position(self):
        """Test exit quantity defaults to full position."""
        order = ExitOrder(
            id="exit1",
            position_qty=10.0,
            tick_size=0.01,
        )
        assert order.get_exit_qty() == 10.0

    def test_qty_cannot_exceed_position(self):
        """Test that exit qty cannot exceed position size."""
        order = ExitOrder(
            id="exit1",
            position_qty=5.0,
            qty=10.0,  # More than position
            tick_size=0.01,
        )
        assert order.get_exit_qty() == 5.0

    def test_validation_both_qty_params(self):
        """Cannot specify both qty and qty_percent."""
        with pytest.raises(ValueError):
            ExitOrder(
                id="exit1",
                position_qty=10.0,
                qty=5.0,
                qty_percent=50.0,
                tick_size=0.01,
            )

    def test_validation_trail_offset_without_activation(self):
        """trail_offset requires trail_price or trail_points."""
        with pytest.raises(ValueError):
            ExitOrder(
                id="exit1",
                trail_offset=10,
                tick_size=0.01,
            )

    def test_validation_trail_price_without_offset(self):
        """trail_price requires trail_offset."""
        with pytest.raises(ValueError):
            ExitOrder(
                id="exit1",
                trail_price=105.0,
                tick_size=0.01,
            )


class TestProfitTarget:
    """Tests for profit/limit target calculations."""

    def test_limit_price_long(self):
        """Test limit price for long position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            limit=105.0,
            tick_size=0.01,
        )
        assert order.get_limit_price() == 105.0

    def test_limit_price_short(self):
        """Test limit price for short position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            limit=95.0,
            tick_size=0.01,
        )
        assert order.get_limit_price() == 95.0

    def test_profit_ticks_long(self):
        """Test profit in ticks for long position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            profit=500,  # 500 ticks = 5.0 with tick_size 0.01
            tick_size=0.01,
        )
        assert order.get_limit_price() == 105.0

    def test_profit_ticks_short(self):
        """Test profit in ticks for short position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            profit=500,  # 500 ticks = 5.0
            tick_size=0.01,
        )
        assert order.get_limit_price() == 95.0

    def test_limit_and_profit_long_use_closer(self):
        """
        When both limit and profit specified, use whichever triggers first.
        For longs, lower limit triggers first.
        """
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            limit=110.0,
            profit=300,  # 3.0 → 103.0
            tick_size=0.01,
        )
        # 103.0 is lower (triggers first for long)
        assert order.get_limit_price() == 103.0

    def test_limit_and_profit_short_use_closer(self):
        """
        When both limit and profit specified, use whichever triggers first.
        For shorts, higher limit triggers first.
        """
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            limit=90.0,
            profit=300,  # 3.0 → 97.0
            tick_size=0.01,
        )
        # 97.0 is higher (triggers first for short)
        assert order.get_limit_price() == 97.0


class TestStopLoss:
    """Tests for stop/loss calculations."""

    def test_stop_price_long(self):
        """Test stop price for long position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            stop=95.0,
            tick_size=0.01,
        )
        assert order.get_stop_price() == 95.0

    def test_stop_price_short(self):
        """Test stop price for short position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            stop=105.0,
            tick_size=0.01,
        )
        assert order.get_stop_price() == 105.0

    def test_loss_ticks_long(self):
        """Test loss in ticks for long position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            loss=250,  # 250 ticks = 2.5
            tick_size=0.01,
        )
        assert order.get_stop_price() == 97.5

    def test_loss_ticks_short(self):
        """Test loss in ticks for short position."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            loss=250,  # 250 ticks = 2.5
            tick_size=0.01,
        )
        assert order.get_stop_price() == 102.5

    def test_stop_and_loss_long_use_closer(self):
        """
        When both stop and loss specified, use whichever triggers first.
        For longs, higher stop triggers first.
        """
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            stop=95.0,
            loss=200,  # 2.0 → 98.0
            tick_size=0.01,
        )
        # 98.0 is higher (triggers first for long)
        assert order.get_stop_price() == 98.0

    def test_stop_and_loss_short_use_closer(self):
        """
        When both stop and loss specified, use whichever triggers first.
        For shorts, lower stop triggers first.
        """
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            stop=105.0,
            loss=200,  # 2.0 → 102.0
            tick_size=0.01,
        )
        # 102.0 is lower (triggers first for short)
        assert order.get_stop_price() == 102.0


class TestTrailingStop:
    """Tests for trailing stop functionality."""

    def test_trail_activation_by_price_long(self):
        """Test trailing stop activation by trail_price for long."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            trail_price=105.0,
            trail_offset=50,  # 0.5 ticks
            tick_size=0.01,
        )

        # Price hasn't reached activation yet
        assert not order.trailing_activated
        order.check_trail_activation(104.0)
        assert not order.trailing_activated

        # Price reaches activation
        order.check_trail_activation(105.0)
        assert order.trailing_activated
        assert order.best_price == 105.0
        # Stop is 0.5 below: 105.0 - 0.5 = 104.5
        assert order.trailing_stop_price == 104.5

    def test_trail_activation_by_price_short(self):
        """Test trailing stop activation by trail_price for short."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            trail_price=95.0,
            trail_offset=50,  # 0.5
            tick_size=0.01,
        )

        # Price hasn't reached activation yet
        order.check_trail_activation(96.0)
        assert not order.trailing_activated

        # Price reaches activation
        order.check_trail_activation(95.0)
        assert order.trailing_activated
        assert order.best_price == 95.0
        # Stop is 0.5 above: 95.0 + 0.5 = 95.5
        assert order.trailing_stop_price == 95.5

    def test_trail_activation_by_points_long(self):
        """Test trailing stop activation by trail_points for long."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            trail_points=300,  # 3.0 profit → activate at 103.0
            trail_offset=100,  # 1.0
            tick_size=0.01,
        )

        order.check_trail_activation(102.0)
        assert not order.trailing_activated

        order.check_trail_activation(103.5)
        assert order.trailing_activated
        assert order.best_price == 103.5
        assert order.trailing_stop_price == 102.5

    def test_trail_points_zero_immediate_activation(self):
        """trail_points=0 activates trailing stop immediately."""
        strategy = StrategyExit(tick_size=0.01)
        order = strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            trail_points=0,
            trail_offset=50,  # 0.5
        )

        assert order.trailing_activated
        assert order.best_price == 100.0
        assert order.trailing_stop_price == 99.5

    def test_trailing_stop_follows_price_long(self):
        """Trailing stop follows price higher for long positions."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            trail_points=0,  # Immediate activation
            trail_offset=100,  # 1.0
            tick_size=0.01,
            status=OrderStatus.ACTIVE,
        )
        order._activate_trailing(100.0)

        # Initial: best=100, stop=99
        assert order.trailing_stop_price == 99.0

        # Price rises to 102
        order.update_trailing_stop(102.0)
        assert order.best_price == 102.0
        assert order.trailing_stop_price == 101.0

        # Price drops to 101.5 (stop doesn't move down)
        order.update_trailing_stop(101.5)
        assert order.best_price == 102.0
        assert order.trailing_stop_price == 101.0

        # Price rises to 103
        order.update_trailing_stop(103.0)
        assert order.best_price == 103.0
        assert order.trailing_stop_price == 102.0

    def test_trailing_stop_follows_price_short(self):
        """Trailing stop follows price lower for short positions."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            trail_points=0,
            trail_offset=100,  # 1.0
            tick_size=0.01,
            status=OrderStatus.ACTIVE,
        )
        order._activate_trailing(100.0)

        # Initial: best=100, stop=101
        assert order.trailing_stop_price == 101.0

        # Price drops to 98
        order.update_trailing_stop(98.0)
        assert order.best_price == 98.0
        assert order.trailing_stop_price == 99.0

        # Price rises to 98.5 (stop doesn't move up)
        order.update_trailing_stop(98.5)
        assert order.best_price == 98.0
        assert order.trailing_stop_price == 99.0

    def test_trailing_stop_hit_long(self):
        """Test trailing stop hit detection for long."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            trail_points=0,
            trail_offset=100,
            tick_size=0.01,
            status=OrderStatus.ACTIVE,
        )
        order._activate_trailing(100.0)
        order.update_trailing_stop(105.0)
        # best=105, stop=104

        assert not order.check_trailing_stop_hit(104.5)
        assert order.check_trailing_stop_hit(104.0)
        assert order.check_trailing_stop_hit(103.0)

    def test_trailing_stop_hit_short(self):
        """Test trailing stop hit detection for short."""
        order = ExitOrder(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            trail_points=0,
            trail_offset=100,
            tick_size=0.01,
            status=OrderStatus.ACTIVE,
        )
        order._activate_trailing(100.0)
        order.update_trailing_stop(95.0)
        # best=95, stop=96

        assert not order.check_trailing_stop_hit(95.5)
        assert order.check_trailing_stop_hit(96.0)
        assert order.check_trailing_stop_hit(97.0)


class TestStrategyExitManager:
    """Tests for StrategyExit manager class."""

    def test_create_exit_order(self):
        """Test creating an exit order."""
        strategy = StrategyExit(tick_size=0.01)
        order = strategy.exit(
            id="exit1",
            from_entry="entry1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            profit=500,
            loss=250,
        )

        assert order is not None
        assert order.id == "exit1"
        assert "exit1" in strategy.orders

    def test_when_false_no_order(self):
        """Test when=False prevents order creation."""
        strategy = StrategyExit(tick_size=0.01)
        order = strategy.exit(
            id="exit1",
            when=False,
            position_qty=10.0,
        )

        assert order is None
        assert "exit1" not in strategy.orders

    def test_cancel_order(self):
        """Test order cancellation."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(id="exit1", position_qty=10.0)

        assert strategy.cancel_order("exit1")
        assert "exit1" not in strategy.orders
        assert not strategy.cancel_order("exit1")  # Already cancelled

    def test_oca_group_cancellation(self):
        """Test OCA group: when one fills, others cancel."""
        strategy = StrategyExit(tick_size=0.01)

        # Create two orders in same OCA group
        strategy.exit(
            id="tp",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=105.0,
            oca_name="bracket1",
        )
        strategy.exit(
            id="sl",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            stop=95.0,
            oca_name="bracket1",
        )

        assert len(strategy.orders) == 2

        # Price hits take profit
        bar = BarData(open=104.0, high=106.0, low=103.5, close=105.5)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.PROFIT
        assert results[0].order_id == "tp"
        assert len(strategy.orders) == 0  # Both orders gone


class TestBarProcessing:
    """Tests for bar processing and order fills."""

    def test_profit_target_fill_long(self):
        """Test profit target fill for long position."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=105.0,
        )

        # Bar that reaches limit
        bar = BarData(open=103.0, high=106.0, low=102.0, close=105.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].filled
        assert results[0].exit_type == ExitType.PROFIT
        assert results[0].fill_price == 105.0
        assert results[0].qty == 10.0

    def test_stop_loss_fill_long(self):
        """Test stop loss fill for long position."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            stop=95.0,
        )

        # Bar that reaches stop
        bar = BarData(open=98.0, high=99.0, low=94.0, close=94.5)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.LOSS
        assert results[0].fill_price == 95.0

    def test_profit_target_fill_short(self):
        """Test profit target fill for short position."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            position_qty=10.0,
            limit=95.0,
        )

        bar = BarData(open=97.0, high=98.0, low=94.0, close=95.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.PROFIT
        assert results[0].fill_price == 95.0

    def test_stop_loss_fill_short(self):
        """Test stop loss fill for short position."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.SHORT,
            entry_price=100.0,
            position_qty=10.0,
            stop=105.0,
        )

        bar = BarData(open=102.0, high=106.0, low=101.0, close=105.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.LOSS
        assert results[0].fill_price == 105.0

    def test_trailing_stop_activation_and_fill(self):
        """Test trailing stop activates and fills in same bar."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            trail_points=200,  # Activate at 102
            trail_offset=100,   # 1.0 trail
        )

        # Bar: opens at 101, goes to high 105, then drops to 103
        # Activation at 102, best price becomes 105, trail stop at 104
        # Price drops to 103 but stop at 104, so fills
        bar = BarData(open=101.0, high=105.0, low=103.0, close=103.5)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.TRAILING

    def test_no_fill_when_price_doesnt_reach(self):
        """Test no fill when price doesn't reach targets."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=110.0,
            stop=90.0,
        )

        bar = BarData(open=100.0, high=105.0, low=95.0, close=102.0)
        results = strategy.process_bar(bar)

        assert len(results) == 0
        assert "exit1" in strategy.orders

    def test_intrabar_order_priority(self):
        """Test that intrabar path determines which order fills first."""
        strategy = StrategyExit(tick_size=0.01)

        # Both profit and stop can be hit in this bar
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=103.0,
            stop=97.0,
        )

        # Bar where open is closer to high, so path is O→H→L→C
        # This means profit target hit first
        bar = BarData(open=100.0, high=105.0, low=95.0, close=98.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.PROFIT

    def test_intrabar_stop_hit_first(self):
        """Test stop hit first when low is closer to open."""
        strategy = StrategyExit(tick_size=0.01)

        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=105.0,
            stop=95.0,
        )

        # Bar where low is closer to open than high, so path is O→L→H→C
        # open=99, dist_to_low=|94-99|=5, dist_to_high=|106-99|=7
        # This means stop hit first (price goes to low=94, hitting stop at 95)
        bar = BarData(open=99.0, high=106.0, low=94.0, close=102.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].exit_type == ExitType.LOSS


class TestPartialExits:
    """Tests for partial position exits."""

    def test_partial_exit_qty(self):
        """Test partial exit with specific qty."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            qty=3.0,
            limit=105.0,
        )

        bar = BarData(open=104.0, high=106.0, low=103.0, close=105.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].qty == 3.0

    def test_partial_exit_percent(self):
        """Test partial exit with percentage."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=100.0,
            qty_percent=25.0,
            limit=105.0,
        )

        bar = BarData(open=104.0, high=106.0, low=103.0, close=105.0)
        results = strategy.process_bar(bar)

        assert len(results) == 1
        assert results[0].qty == 25.0

    def test_multiple_partial_exits(self):
        """Test multiple partial exit orders."""
        strategy = StrategyExit(tick_size=0.01)

        # Two partial exits at different levels
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            qty=5.0,
            limit=103.0,
        )
        strategy.exit(
            id="exit2",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            qty=5.0,
            limit=106.0,
        )

        # First bar hits first exit
        bar1 = BarData(open=102.0, high=104.0, low=101.0, close=103.5)
        results1 = strategy.process_bar(bar1)
        assert len(results1) == 1
        assert results1[0].qty == 5.0

        # Second bar hits second exit
        bar2 = BarData(open=105.0, high=107.0, low=104.0, close=106.0)
        results2 = strategy.process_bar(bar2)
        assert len(results2) == 1
        assert results2[0].qty == 5.0


class TestComments:
    """Tests for order comments."""

    def test_profit_comment(self):
        """Test comment_profit is used when profit exit."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=105.0,
            comment="default",
            comment_profit="hit profit target",
        )

        bar = BarData(open=104.0, high=106.0, low=103.0, close=105.0)
        results = strategy.process_bar(bar)

        assert results[0].comment == "hit profit target"

    def test_loss_comment(self):
        """Test comment_loss is used when stop exit."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            stop=95.0,
            comment="default",
            comment_loss="stopped out",
        )

        bar = BarData(open=96.0, high=97.0, low=94.0, close=94.5)
        results = strategy.process_bar(bar)

        assert results[0].comment == "stopped out"

    def test_trailing_comment(self):
        """Test comment_trailing is used when trailing exit."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            trail_points=0,
            trail_offset=100,
            comment="default",
            comment_trailing="trailing stop hit",
        )

        bar = BarData(open=102.0, high=105.0, low=103.5, close=104.0)
        results = strategy.process_bar(bar)

        assert results[0].comment == "trailing stop hit"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_price_to_ticks(self):
        """Test price to ticks conversion."""
        assert price_to_ticks(1.0, 0.01) == 100
        assert price_to_ticks(0.5, 0.25) == 2

    def test_ticks_to_price(self):
        """Test ticks to price conversion."""
        assert ticks_to_price(100, 0.01) == 1.0
        assert ticks_to_price(4, 0.25) == 1.0

    def test_percent_to_ticks(self):
        """Test percentage to ticks conversion."""
        # 2% of 100 = 2.0, with tick 0.01 = 200 ticks
        assert percent_to_ticks(2.0, 100.0, 0.01) == 200

        # 1% of 50 = 0.5, with tick 0.25 = 2 ticks
        assert percent_to_ticks(1.0, 50.0, 0.25) == 2


class TestTickProcessing:
    """Tests for tick-by-tick processing."""

    def test_process_tick_limit_fill(self):
        """Test tick processing fills limit order."""
        strategy = StrategyExit(tick_size=0.01)
        strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            limit=105.0,
        )

        results = strategy.process_tick(104.0)
        assert len(results) == 0

        results = strategy.process_tick(105.0)
        assert len(results) == 1
        assert results[0].exit_type == ExitType.PROFIT

    def test_process_tick_trailing_activation(self):
        """Test tick processing activates and updates trailing stop."""
        strategy = StrategyExit(tick_size=0.01)
        order = strategy.exit(
            id="exit1",
            position_side=PositionSide.LONG,
            entry_price=100.0,
            position_qty=10.0,
            trail_points=200,  # Activate at 102
            trail_offset=50,   # 0.5 trail
        )

        # Not activated yet
        strategy.process_tick(101.0)
        assert not order.trailing_activated

        # Activates
        strategy.process_tick(102.5)
        assert order.trailing_activated
        assert order.best_price == 102.5
        assert order.trailing_stop_price == 102.0

        # Trails higher
        strategy.process_tick(104.0)
        assert order.best_price == 104.0
        assert order.trailing_stop_price == 103.5

        # Hits trailing stop
        results = strategy.process_tick(103.5)
        assert len(results) == 1
        assert results[0].exit_type == ExitType.TRAILING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
