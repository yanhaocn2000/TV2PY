"""
Keltner Channel Strategy - TradingView to Python Conversion

原始 Pine Script 策略: BB Keltner Squeeze Strategy
来源: https://github.com/hasnocool/tradingview-pine-scripts

策略逻辑:
1. Squeeze 检测: 当 Bollinger Bands 收缩到 Keltner Channel 内部时
2. 做多: Squeeze 释放且价格在中轨上方
3. 做空: Squeeze 释放且价格在中轨下方
4. 平仓: 方向改变时
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum
import numpy as np
import pandas as pd


class SqueezeType(Enum):
    """Squeeze 检测类型"""
    WITHIN = "within"  # BB 在 KC 内部
    WIDTH = "width"    # 宽度比率


class Direction(Enum):
    """持仓方向"""
    NONE = 0
    LONG = 1
    SHORT = 2


@dataclass
class KeltnerChannelParams:
    """Keltner Channel 策略参数"""
    # 基础参数
    length: int = 20

    # Bollinger Bands 参数
    bb_mult: float = 2.0

    # Keltner Channel 参数
    kc_mult: float = 1.5
    use_true_range: bool = True

    # Squeeze 参数
    squeeze_type: SqueezeType = SqueezeType.WITHIN
    width_ratio: float = 1.0

    # 止盈止损参数
    partial1_pct: float = 0.0      # 第一次部分止盈 %
    partial1_qty: float = 50.0     # 第一次止盈数量 %
    partial2_pct: float = 0.0      # 第二次部分止盈 %
    partial2_qty: float = 50.0     # 第二次止盈数量 %
    take_profit_pct: float = 0.0   # 完全止盈 %
    stop_loss_pct: float = 0.0     # 止损 %
    trailing_stop_pct: float = 0.0 # 移动止损 %


@dataclass
class KeltnerSignal:
    """Keltner Channel 信号"""
    timestamp: pd.Timestamp
    direction: Direction
    entry_price: float
    bb_upper: float
    bb_lower: float
    bb_basis: float
    kc_upper: float
    kc_lower: float
    is_squeeze: bool
    squeeze_released: bool = False


class KeltnerChannelIndicator:
    """
    Keltner Channel 指标计算

    与 TradingView 完全对齐的实现
    """

    def __init__(self, params: KeltnerChannelParams):
        self.params = params

    def calculate_bb(self, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 Bollinger Bands

        Pine Script:
            B2basis = sma(src, length)
            B2dev = B2mult * stdev(src, length)
            B2upper = B2basis + B2dev
            B2lower = B2basis - B2dev
        """
        length = self.params.length
        mult = self.params.bb_mult

        # SMA 中轨
        basis = self._sma(close, length)

        # 标准差
        dev = mult * self._stdev(close, length)

        upper = basis + dev
        lower = basis - dev

        return upper, basis, lower

    def calculate_kc(self, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 Keltner Channel

        Pine Script:
            Kma = ema(src, length)
            Krange = useTrueRange ? tr : high - low
            Krangema = ema(Krange, length)
            Kupper = Kma + Krangema * Kmult
            Klower = Kma - Krangema * Kmult
        """
        length = self.params.length
        mult = self.params.kc_mult

        # EMA 中轨
        basis = self._ema(close, length)

        # True Range 或 High-Low
        if self.params.use_true_range:
            tr = self._true_range(high, low, close)
        else:
            tr = high - low

        # ATR (EMA of TR)
        atr = self._ema(tr, length)

        upper = basis + atr * mult
        lower = basis - atr * mult

        return upper, basis, lower

    def detect_squeeze(self, bb_upper: np.ndarray, bb_lower: np.ndarray,
                       kc_upper: np.ndarray, kc_lower: np.ndarray,
                       bb_basis: np.ndarray, kc_basis: np.ndarray) -> np.ndarray:
        """
        检测 Squeeze 状态

        Pine Script:
            squeeze = squeeze_choice=="Width" ?
                      (B2width / Kwidth <= widthRatio) :
                      (B2upper <= Kupper or B2lower >= Klower)
        """
        if self.params.squeeze_type == SqueezeType.WIDTH:
            # 宽度比率法
            bb_width = (bb_upper - bb_lower) / bb_basis
            kc_width = (kc_upper - kc_lower) / kc_basis
            squeeze = bb_width / (kc_width + 1e-10) <= self.params.width_ratio
        else:
            # 包含法: BB 在 KC 内部
            squeeze = (bb_upper <= kc_upper) | (bb_lower >= kc_lower)

        return squeeze

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average - TradingView 对齐"""
        result = np.full_like(data, np.nan)
        alpha = 2.0 / (period + 1)

        # 第一个有效值使用 SMA
        result[period - 1] = np.mean(data[:period])

        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def _stdev(self, data: np.ndarray, period: int) -> np.ndarray:
        """Standard Deviation - TradingView 使用 population stdev"""
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            result[i] = np.std(window, ddof=0)  # population std
        return result

    def _true_range(self, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray) -> np.ndarray:
        """True Range"""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        return np.maximum(np.maximum(tr1, tr2), tr3)


class KeltnerChannelStrategy:
    """
    Keltner Channel 策略

    转换自 TradingView BB Keltner Squeeze Strategy
    """

    def __init__(self, params: Optional[KeltnerChannelParams] = None):
        self.params = params or KeltnerChannelParams()
        self.indicator = KeltnerChannelIndicator(self.params)
        self.direction = Direction.NONE
        self.entry_price = 0.0
        self.signals: List[KeltnerSignal] = []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        运行策略

        Args:
            df: OHLCV DataFrame

        Returns:
            带有信号的 DataFrame
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 计算 Bollinger Bands
        bb_upper, bb_basis, bb_lower = self.indicator.calculate_bb(close)

        # 计算 Keltner Channel
        kc_upper, kc_basis, kc_lower = self.indicator.calculate_kc(high, low, close)

        # 检测 Squeeze
        squeeze = self.indicator.detect_squeeze(
            bb_upper, bb_lower, kc_upper, kc_lower, bb_basis, kc_basis
        )

        # 生成信号
        signals = self._generate_signals(close, bb_basis, squeeze)

        # 添加到 DataFrame
        result = df.copy()
        result['bb_upper'] = bb_upper
        result['bb_basis'] = bb_basis
        result['bb_lower'] = bb_lower
        result['kc_upper'] = kc_upper
        result['kc_basis'] = kc_basis
        result['kc_lower'] = kc_lower
        result['squeeze'] = squeeze
        result['signal'] = signals
        result['position'] = self._calculate_positions(signals)

        return result

    def _generate_signals(self, close: np.ndarray, bb_basis: np.ndarray,
                          squeeze: np.ndarray) -> np.ndarray:
        """
        生成交易信号

        Pine Script 逻辑:
            midc = squeeze ? 0 : close > B2basis ? 1 : 2

            if direction == 0:
                if midc[1] == 0 and midc == 1:
                    strategy.entry("LONG", strategy.long)
                    direction := 1
                else if midc[1] == 0 and midc == 2:
                    strategy.entry("SHORT", strategy.short)
                    direction := 2
            else if direction != midc:
                strategy.close_all()
                direction := 0
        """
        n = len(close)
        signals = np.zeros(n)  # 0=无信号, 1=买入, -1=卖出, 2=平仓

        # 计算 midc (市场状态)
        # 0 = squeeze 中, 1 = 价格在中轨上方, 2 = 价格在中轨下方
        midc = np.zeros(n)
        for i in range(n):
            if np.isnan(bb_basis[i]):
                midc[i] = 0
            elif squeeze[i]:
                midc[i] = 0  # Squeeze 中
            elif close[i] > bb_basis[i]:
                midc[i] = 1  # 价格在中轨上方
            else:
                midc[i] = 2  # 价格在中轨下方

        direction = 0  # 0=无持仓, 1=多头, 2=空头

        for i in range(1, n):
            if np.isnan(bb_basis[i]):
                continue

            if direction == 0:
                # 无持仓, 等待 squeeze 释放
                if midc[i - 1] == 0 and midc[i] == 1:
                    # Squeeze 释放, 价格向上突破 -> 做多
                    signals[i] = 1
                    direction = 1
                elif midc[i - 1] == 0 and midc[i] == 2:
                    # Squeeze 释放, 价格向下突破 -> 做空
                    signals[i] = -1
                    direction = 2
            else:
                # 有持仓, 检查是否需要平仓
                if direction != midc[i] and midc[i] != 0:
                    # 方向改变, 平仓
                    signals[i] = 2  # 平仓信号
                    direction = 0

        return signals

    def _calculate_positions(self, signals: np.ndarray) -> np.ndarray:
        """计算持仓状态"""
        positions = np.zeros_like(signals)
        position = 0

        for i in range(len(signals)):
            if signals[i] == 1:
                position = 1
            elif signals[i] == -1:
                position = -1
            elif signals[i] == 2:
                position = 0
            positions[i] = position

        return positions

    def get_stop_loss_levels(self, entry_price: float,
                             direction: Direction) -> dict:
        """
        计算止盈止损价位

        Pine Script:
            strategy.exit("Take Profit", limit = close * (1 + take_profit))
            strategy.exit("Stop Loss", stop = close * (1 - stop_loss))
        """
        if direction == Direction.LONG:
            return {
                'partial1': entry_price * (1 + self.params.partial1_pct / 100) if self.params.partial1_pct > 0 else None,
                'partial2': entry_price * (1 + self.params.partial2_pct / 100) if self.params.partial2_pct > 0 else None,
                'take_profit': entry_price * (1 + self.params.take_profit_pct / 100) if self.params.take_profit_pct > 0 else None,
                'stop_loss': entry_price * (1 - self.params.stop_loss_pct / 100) if self.params.stop_loss_pct > 0 else None,
            }
        elif direction == Direction.SHORT:
            return {
                'partial1': entry_price * (1 - self.params.partial1_pct / 100) if self.params.partial1_pct > 0 else None,
                'partial2': entry_price * (1 - self.params.partial2_pct / 100) if self.params.partial2_pct > 0 else None,
                'take_profit': entry_price * (1 - self.params.take_profit_pct / 100) if self.params.take_profit_pct > 0 else None,
                'stop_loss': entry_price * (1 + self.params.stop_loss_pct / 100) if self.params.stop_loss_pct > 0 else None,
            }
        return {}


class KeltnerChannelPyneCore:
    """
    使用 PyneCore 实现的 Keltner Channel

    PyneCore 提供 TradingView 100% 兼容的指标计算
    """

    def __init__(self, params: Optional[KeltnerChannelParams] = None):
        self.params = params or KeltnerChannelParams()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用 PyneCore 运行策略"""
        try:
            from pynecore import ta

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # Bollinger Bands
            bb_basis = ta.sma(close, self.params.length)
            bb_dev = self.params.bb_mult * ta.stdev(close, self.params.length)
            bb_upper = bb_basis + bb_dev
            bb_lower = bb_basis - bb_dev

            # Keltner Channel
            kc_basis = ta.ema(close, self.params.length)
            atr = ta.atr(high, low, close, self.params.length)
            kc_upper = kc_basis + atr * self.params.kc_mult
            kc_lower = kc_basis - atr * self.params.kc_mult

            # Squeeze 检测
            if self.params.squeeze_type == SqueezeType.WIDTH:
                bb_width = (bb_upper - bb_lower) / bb_basis
                kc_width = (kc_upper - kc_lower) / kc_basis
                squeeze = bb_width / kc_width <= self.params.width_ratio
            else:
                squeeze = (bb_upper <= kc_upper) | (bb_lower >= kc_lower)

            result = df.copy()
            result['bb_upper'] = bb_upper
            result['bb_basis'] = bb_basis
            result['bb_lower'] = bb_lower
            result['kc_upper'] = kc_upper
            result['kc_basis'] = kc_basis
            result['kc_lower'] = kc_lower
            result['squeeze'] = squeeze

            return result

        except ImportError:
            print("PyneCore 未安装, 使用内置实现")
            strategy = KeltnerChannelStrategy(self.params)
            return strategy.run(df)


# 便捷函数
def keltner_channel(df: pd.DataFrame,
                    length: int = 20,
                    bb_mult: float = 2.0,
                    kc_mult: float = 1.5,
                    use_pynecore: bool = True) -> pd.DataFrame:
    """
    计算 Keltner Channel 和 Bollinger Bands Squeeze

    Args:
        df: OHLCV DataFrame
        length: 周期
        bb_mult: BB 标准差倍数
        kc_mult: KC ATR 倍数
        use_pynecore: 是否使用 PyneCore

    Returns:
        带有指标的 DataFrame
    """
    params = KeltnerChannelParams(
        length=length,
        bb_mult=bb_mult,
        kc_mult=kc_mult
    )

    if use_pynecore:
        indicator = KeltnerChannelPyneCore(params)
    else:
        indicator = KeltnerChannelStrategy(params)

    return indicator.run(df)


if __name__ == "__main__":
    # 示例用法
    import pandas as pd

    # 生成示例数据
    np.random.seed(42)
    n = 200

    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    # 运行策略
    params = KeltnerChannelParams(
        length=20,
        bb_mult=2.0,
        kc_mult=1.5,
        stop_loss_pct=2.0,
        take_profit_pct=4.0
    )

    strategy = KeltnerChannelStrategy(params)
    result = strategy.run(df)

    # 统计信号
    buy_signals = (result['signal'] == 1).sum()
    sell_signals = (result['signal'] == -1).sum()
    close_signals = (result['signal'] == 2).sum()
    squeeze_count = result['squeeze'].sum()

    print(f"策略: BB Keltner Squeeze")
    print(f"数据点: {n}")
    print(f"Squeeze 次数: {squeeze_count}")
    print(f"买入信号: {buy_signals}")
    print(f"卖出信号: {sell_signals}")
    print(f"平仓信号: {close_signals}")
