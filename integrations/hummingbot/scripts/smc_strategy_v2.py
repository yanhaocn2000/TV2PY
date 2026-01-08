"""
Smart Money Concepts Strategy for Hummingbot V2

使用 TV2PY Smart Money Concepts 指标的 Hummingbot V2 脚本

策略逻辑:
    1. 使用 SMC 分析市场结构 (BOS, CHoCH)
    2. 在折价区寻找看涨 Order Block / FVG
    3. 在溢价区寻找看跌 Order Block / FVG
    4. 使用 Squeeze Momentum 确认入场时机
    5. 基于 ATR 设置止损止盈

使用方法:
    1. 将此文件复制到 Hummingbot 的 scripts 目录
    2. 确保 TV2PY 已安装或在 Python 路径中
    3. 运行: start --script smc_strategy_v2.py

配置参数:
    - trading_pair: 交易对 (如 ETH-USDT)
    - exchange: 交易所 (如 binance)
    - order_amount: 订单金额
    - leverage: 杠杆倍数 (仅限合约)
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional

# Hummingbot 导入
try:
    from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
    from hummingbot.connector.connector_base import ConnectorBase
    from hummingbot.core.data_type.common import OrderType, TradeType
    from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
    HAS_HUMMINGBOT = True
except ImportError:
    HAS_HUMMINGBOT = False
    # Mock classes for development/testing
    class ScriptStrategyBase:
        def logger(self):
            import logging
            return logging.getLogger(__name__)

    class ConnectorBase:
        pass

    class OrderType:
        MARKET = "market"
        LIMIT = "limit"

    class TradeType:
        BUY = "buy"
        SELL = "sell"

import pandas as pd
import numpy as np

# TV2PY 导入
import sys
from pathlib import Path
TV2PY_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TV2PY_PATH))

from integrations.hummingbot import TV2PYSignalGenerator, SignalType
from integrations.hummingbot.signal_generator import StrategyType


class SMCStrategyV2(ScriptStrategyBase):
    """
    Smart Money Concepts Strategy

    使用 SMC + Squeeze + Volume Profile 的综合策略
    """

    # ==================== 配置参数 ====================

    # 交易对和交易所
    trading_pair: str = "ETH-USDT"
    exchange: str = "binance_perpetual"

    # 订单参数
    order_amount: Decimal = Decimal("0.1")  # ETH 数量
    leverage: int = 5

    # K线参数
    candle_interval: str = "4h"
    candles_length: int = 200  # 需要的历史K线数量

    # 风险管理
    max_position: Decimal = Decimal("1.0")  # 最大持仓
    risk_per_trade: float = 0.02  # 每笔交易风险 2%
    min_confidence: float = 0.6  # 最小信号置信度

    # 状态变量
    _signal_generator: Optional[TV2PYSignalGenerator] = None
    _last_signal_time: int = 0
    _current_position: int = 0  # 1=多, -1=空, 0=无

    # ==================== 初始化 ====================

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)

        # 初始化信号生成器
        self._signal_generator = TV2PYSignalGenerator(
            strategy_type=StrategyType.SMART_MONEY,
            risk_per_trade=self.risk_per_trade,
            min_confidence=self.min_confidence,
        )

        self.logger().info("SMC Strategy V2 initialized")
        self.logger().info(f"Trading pair: {self.trading_pair}")
        self.logger().info(f"Exchange: {self.exchange}")

    # ==================== K线数据 ====================

    @property
    def candles_config(self) -> List[Dict]:
        """Hummingbot Candles 配置"""
        return [
            {
                "connector": self.exchange,
                "trading_pair": self.trading_pair,
                "interval": self.candle_interval,
                "max_records": self.candles_length,
            }
        ]

    def get_candles_df(self) -> Optional[pd.DataFrame]:
        """获取K线 DataFrame"""
        try:
            candles = self.candles.get(
                f"{self.exchange}_{self.trading_pair}_{self.candle_interval}"
            )
            if candles is not None and len(candles) > 0:
                return candles.candles_df
        except Exception as e:
            self.logger().error(f"Error getting candles: {e}")
        return None

    # ==================== 策略逻辑 ====================

    def on_tick(self):
        """每个 tick 执行的主要逻辑"""
        # 检查是否有足够的K线数据
        candles_df = self.get_candles_df()
        if candles_df is None or len(candles_df) < 100:
            return

        # 获取当前价格
        current_price = self.connectors[self.exchange].get_mid_price(self.trading_pair)
        if current_price is None:
            return

        # 获取当前持仓
        position = self._get_current_position()

        # 生成交易信号
        signal = self._signal_generator.get_signal(
            candles=candles_df,
            current_position=position,
        )

        # 检查信号置信度
        if signal.confidence < self.min_confidence:
            return

        # 执行交易
        self._execute_signal(signal, current_price)

    def _get_current_position(self) -> int:
        """获取当前持仓方向"""
        try:
            positions = self.connectors[self.exchange].account_positions
            for pos in positions.values():
                if pos.trading_pair == self.trading_pair:
                    if pos.amount > 0:
                        return 1
                    elif pos.amount < 0:
                        return -1
            return 0
        except Exception:
            return self._current_position

    def _execute_signal(self, signal, current_price: Decimal):
        """执行交易信号"""
        connector = self.connectors[self.exchange]

        # 计算订单数量
        amount = self.order_amount

        if signal.signal_type == SignalType.LONG:
            self.logger().info(f"LONG Signal: {signal.reason}")
            self.logger().info(f"  Confidence: {signal.confidence:.2%}")
            self.logger().info(f"  Stop Loss: {signal.stop_loss}")
            self.logger().info(f"  Take Profit: {signal.take_profit}")

            # 下多单
            self.buy(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=amount,
                order_type=OrderType.MARKET,
            )
            self._current_position = 1

            # 设置止损单 (如果支持)
            if signal.stop_loss:
                self._place_stop_loss(TradeType.SELL, signal.stop_loss, amount)

        elif signal.signal_type == SignalType.SHORT:
            self.logger().info(f"SHORT Signal: {signal.reason}")
            self.logger().info(f"  Confidence: {signal.confidence:.2%}")
            self.logger().info(f"  Stop Loss: {signal.stop_loss}")
            self.logger().info(f"  Take Profit: {signal.take_profit}")

            # 下空单
            self.sell(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=amount,
                order_type=OrderType.MARKET,
            )
            self._current_position = -1

            if signal.stop_loss:
                self._place_stop_loss(TradeType.BUY, signal.stop_loss, amount)

        elif signal.signal_type == SignalType.CLOSE_LONG:
            self.logger().info(f"CLOSE LONG: {signal.reason}")
            self.sell(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=amount,
                order_type=OrderType.MARKET,
            )
            self._current_position = 0

        elif signal.signal_type == SignalType.CLOSE_SHORT:
            self.logger().info(f"CLOSE SHORT: {signal.reason}")
            self.buy(
                connector_name=self.exchange,
                trading_pair=self.trading_pair,
                amount=amount,
                order_type=OrderType.MARKET,
            )
            self._current_position = 0

    def _place_stop_loss(self, side: TradeType, price: float, amount: Decimal):
        """下止损单"""
        try:
            if side == TradeType.SELL:
                self.sell(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=Decimal(str(price)),
                )
            else:
                self.buy(
                    connector_name=self.exchange,
                    trading_pair=self.trading_pair,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=Decimal(str(price)),
                )
        except Exception as e:
            self.logger().warning(f"Could not place stop loss: {e}")

    # ==================== 状态显示 ====================

    def format_status(self) -> str:
        """显示策略状态"""
        lines = []
        lines.append("=" * 50)
        lines.append("SMC Strategy V2 Status")
        lines.append("=" * 50)

        # 持仓状态
        position_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}
        lines.append(f"Position: {position_str.get(self._current_position, 'Unknown')}")

        # K线状态
        candles_df = self.get_candles_df()
        if candles_df is not None:
            lines.append(f"Candles: {len(candles_df)} bars")
            lines.append(f"Last price: {candles_df['close'].iloc[-1]:.2f}")
        else:
            lines.append("Candles: Not available")

        # 最近信号
        if candles_df is not None and len(candles_df) >= 100:
            signal = self._signal_generator.get_signal(candles_df, self._current_position)
            lines.append(f"\nLatest Signal:")
            lines.append(f"  Type: {signal.signal_type.name}")
            lines.append(f"  Strength: {signal.strength:.2%}")
            lines.append(f"  Confidence: {signal.confidence:.2%}")
            lines.append(f"  Reason: {signal.reason}")

            # SMC 指标
            if "smc" in signal.indicators:
                smc = signal.indicators["smc"]
                lines.append(f"\nSMC Indicators:")
                lines.append(f"  Trend: {smc.get('trend', 'N/A')}")
                lines.append(f"  In Premium: {smc.get('in_premium', 'N/A')}")
                lines.append(f"  In Discount: {smc.get('in_discount', 'N/A')}")
                lines.append(f"  Active Bullish OB: {smc.get('active_bullish_ob', 0)}")
                lines.append(f"  Active Bearish OB: {smc.get('active_bearish_ob', 0)}")

        lines.append("=" * 50)
        return "\n".join(lines)


# ==================== 测试代码 ====================

def test_strategy():
    """测试策略 (无需 Hummingbot)"""
    print("Testing SMC Strategy V2...")
    print("=" * 50)

    # 生成模拟数据
    np.random.seed(42)
    n = 200

    dates = pd.date_range(start="2024-01-01", periods=n, freq="4h")

    # 模拟价格走势
    base = 3000
    trend = np.cumsum(np.random.randn(n) * 20)
    close = base + trend
    high = close + np.abs(np.random.randn(n)) * 30
    low = close - np.abs(np.random.randn(n)) * 30
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = 1000000 + np.abs(np.random.randn(n)) * 500000

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    # 创建信号生成器
    signal_gen = TV2PYSignalGenerator(
        strategy_type=StrategyType.SMART_MONEY,
        risk_per_trade=0.02,
        min_confidence=0.5,
    )

    # 获取信号
    signal = signal_gen.get_signal(df, current_position=0)

    print(f"\nSignal Analysis:")
    print(f"  Type: {signal.signal_type.name}")
    print(f"  Price: ${signal.price:.2f}")
    print(f"  Strength: {signal.strength:.2%}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "  Stop Loss: N/A")
    print(f"  Take Profit: ${signal.take_profit:.2f}" if signal.take_profit else "  Take Profit: N/A")
    print(f"  Position Size: {signal.position_size:.2%}")
    print(f"  Reason: {signal.reason}")

    print("\nIndicator Values:")
    for name, values in signal.indicators.items():
        print(f"  {name}:")
        for k, v in values.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_strategy()
