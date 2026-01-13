"""
Pine Script Compatible Indicators.

These indicators match TradingView's Pine Script ta.* functions.
Can be used standalone or integrated with PyneCore.
"""

import numpy as np
from typing import Union

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, list]


class PyneIndicators:
    """
    Collection of Pine Script compatible technical indicators.

    All indicators are designed to produce results identical to
    TradingView's ta.* functions.
    """

    @staticmethod
    def sma(source: ArrayLike, length: int) -> np.ndarray:
        """
        Simple Moving Average (ta.sma).

        Args:
            source: Input price series
            length: Number of periods

        Returns:
            SMA values
        """
        source = np.asarray(source, dtype=float)
        if len(source) < length:
            return np.full(len(source), np.nan)

        result = np.full(len(source), np.nan)
        for i in range(length - 1, len(source)):
            result[i] = np.mean(source[i - length + 1:i + 1])
        return result

    @staticmethod
    def ema(source: ArrayLike, length: int) -> np.ndarray:
        """
        Exponential Moving Average (ta.ema).

        Args:
            source: Input price series
            length: Number of periods

        Returns:
            EMA values
        """
        source = np.asarray(source, dtype=float)
        if len(source) == 0:
            return np.array([])

        alpha = 2.0 / (length + 1)
        result = np.full(len(source), np.nan)

        # Initialize with SMA
        if len(source) >= length:
            result[length - 1] = np.mean(source[:length])

            # Calculate EMA
            for i in range(length, len(source)):
                result[i] = alpha * source[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    def rma(source: ArrayLike, length: int) -> np.ndarray:
        """
        Relative Moving Average / Wilder's Smoothing (ta.rma).

        Args:
            source: Input price series
            length: Number of periods

        Returns:
            RMA values
        """
        source = np.asarray(source, dtype=float)
        if len(source) == 0:
            return np.array([])

        alpha = 1.0 / length
        result = np.full(len(source), np.nan)

        if len(source) >= length:
            result[length - 1] = np.mean(source[:length])

            for i in range(length, len(source)):
                result[i] = alpha * source[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    def rsi(source: ArrayLike, length: int = 14) -> np.ndarray:
        """
        Relative Strength Index (ta.rsi).

        Args:
            source: Input price series
            length: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        source = np.asarray(source, dtype=float)
        if len(source) < 2:
            return np.full(len(source), np.nan)

        # Calculate price changes
        delta = np.diff(source)
        delta = np.insert(delta, 0, np.nan)

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Calculate RMA of gains and losses
        avg_gain = PyneIndicators.rma(gains, length)
        avg_loss = PyneIndicators.rma(losses, length)

        # Calculate RSI
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))

        # Handle edge cases
        rsi = np.where(avg_loss == 0, 100, rsi)
        rsi = np.where(avg_gain == 0, 0, rsi)

        return rsi

    @staticmethod
    def macd(source: ArrayLike, fast_length: int = 12, slow_length: int = 26,
             signal_length: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving Average Convergence Divergence (ta.macd).

        Args:
            source: Input price series
            fast_length: Fast EMA period
            slow_length: Slow EMA period
            signal_length: Signal line period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = PyneIndicators.ema(source, fast_length)
        slow_ema = PyneIndicators.ema(source, slow_length)

        macd_line = fast_ema - slow_ema
        signal_line = PyneIndicators.ema(macd_line, signal_length)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(source: ArrayLike, length: int = 20,
                        mult: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands (ta.bb).

        Args:
            source: Input price series
            length: SMA period
            mult: Standard deviation multiplier

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        source = np.asarray(source, dtype=float)

        middle = PyneIndicators.sma(source, length)

        # Calculate rolling standard deviation
        std = np.full(len(source), np.nan)
        for i in range(length - 1, len(source)):
            std[i] = np.std(source[i - length + 1:i + 1], ddof=0)

        upper = middle + mult * std
        lower = middle - mult * std

        return upper, middle, lower

    @staticmethod
    def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike,
            length: int = 14) -> np.ndarray:
        """
        Average True Range (ta.atr).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            length: ATR period

        Returns:
            ATR values
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # True Range
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        return PyneIndicators.rma(tr, length)

    @staticmethod
    def stoch(high: ArrayLike, low: ArrayLike, close: ArrayLike,
              k_length: int = 14, k_smooth: int = 1,
              d_smooth: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Stochastic Oscillator (ta.stoch).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_length: %K period
            k_smooth: %K smoothing
            d_smooth: %D smoothing

        Returns:
            Tuple of (%K, %D)
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        # Calculate raw stochastic
        stoch_k = np.full(len(close), np.nan)
        for i in range(k_length - 1, len(close)):
            highest = np.max(high[i - k_length + 1:i + 1])
            lowest = np.min(low[i - k_length + 1:i + 1])
            if highest != lowest:
                stoch_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
            else:
                stoch_k[i] = 50

        # Smooth %K
        k = PyneIndicators.sma(stoch_k, k_smooth)

        # Calculate %D
        d = PyneIndicators.sma(k, d_smooth)

        return k, d

    @staticmethod
    def crossover(series1: ArrayLike, series2: ArrayLike) -> np.ndarray:
        """
        Crossover detection (ta.crossover).

        Returns True when series1 crosses above series2.

        Args:
            series1: First series
            series2: Second series (or constant)

        Returns:
            Boolean array
        """
        series1 = np.asarray(series1, dtype=float)
        series2 = np.asarray(series2, dtype=float) if hasattr(series2, '__len__') else np.full(len(series1), series2)

        if len(series1) < 2:
            return np.array([False] * len(series1))

        # Current: series1 > series2, Previous: series1 <= series2
        current_above = series1 > series2
        prev_below_or_equal = np.roll(series1, 1) <= np.roll(series2, 1)

        result = current_above & prev_below_or_equal
        result[0] = False

        return result

    @staticmethod
    def crossunder(series1: ArrayLike, series2: ArrayLike) -> np.ndarray:
        """
        Crossunder detection (ta.crossunder).

        Returns True when series1 crosses below series2.

        Args:
            series1: First series
            series2: Second series (or constant)

        Returns:
            Boolean array
        """
        series1 = np.asarray(series1, dtype=float)
        series2 = np.asarray(series2, dtype=float) if hasattr(series2, '__len__') else np.full(len(series1), series2)

        if len(series1) < 2:
            return np.array([False] * len(series1))

        # Current: series1 < series2, Previous: series1 >= series2
        current_below = series1 < series2
        prev_above_or_equal = np.roll(series1, 1) >= np.roll(series2, 1)

        result = current_below & prev_above_or_equal
        result[0] = False

        return result

    @staticmethod
    def highest(source: ArrayLike, length: int) -> np.ndarray:
        """
        Highest value over period (ta.highest).

        Args:
            source: Input series
            length: Lookback period

        Returns:
            Highest values
        """
        source = np.asarray(source, dtype=float)
        result = np.full(len(source), np.nan)

        for i in range(length - 1, len(source)):
            result[i] = np.max(source[i - length + 1:i + 1])

        return result

    @staticmethod
    def lowest(source: ArrayLike, length: int) -> np.ndarray:
        """
        Lowest value over period (ta.lowest).

        Args:
            source: Input series
            length: Lookback period

        Returns:
            Lowest values
        """
        source = np.asarray(source, dtype=float)
        result = np.full(len(source), np.nan)

        for i in range(length - 1, len(source)):
            result[i] = np.min(source[i - length + 1:i + 1])

        return result

    @staticmethod
    def change(source: ArrayLike, length: int = 1) -> np.ndarray:
        """
        Change in value (ta.change).

        Args:
            source: Input series
            length: Lookback period

        Returns:
            Change values
        """
        source = np.asarray(source, dtype=float)
        result = np.full(len(source), np.nan)

        for i in range(length, len(source)):
            result[i] = source[i] - source[i - length]

        return result

    @staticmethod
    def mom(source: ArrayLike, length: int = 10) -> np.ndarray:
        """
        Momentum (ta.mom).

        Args:
            source: Input series
            length: Momentum period

        Returns:
            Momentum values
        """
        return PyneIndicators.change(source, length)

    @staticmethod
    def roc(source: ArrayLike, length: int = 10) -> np.ndarray:
        """
        Rate of Change (ta.roc).

        Args:
            source: Input series
            length: ROC period

        Returns:
            ROC values (percentage)
        """
        source = np.asarray(source, dtype=float)
        result = np.full(len(source), np.nan)

        for i in range(length, len(source)):
            if source[i - length] != 0:
                result[i] = 100 * (source[i] - source[i - length]) / source[i - length]

        return result

    @staticmethod
    def wma(source: ArrayLike, length: int) -> np.ndarray:
        """
        Weighted Moving Average (ta.wma).

        Args:
            source: Input series
            length: WMA period

        Returns:
            WMA values
        """
        source = np.asarray(source, dtype=float)
        result = np.full(len(source), np.nan)

        weights = np.arange(1, length + 1)
        weight_sum = weights.sum()

        for i in range(length - 1, len(source)):
            window = source[i - length + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum

        return result

    @staticmethod
    def vwma(source: ArrayLike, volume: ArrayLike, length: int) -> np.ndarray:
        """
        Volume Weighted Moving Average (ta.vwma).

        Args:
            source: Input series (typically close)
            volume: Volume series
            length: VWMA period

        Returns:
            VWMA values
        """
        source = np.asarray(source, dtype=float)
        volume = np.asarray(volume, dtype=float)

        pv = source * volume

        pv_sum = np.full(len(source), np.nan)
        v_sum = np.full(len(source), np.nan)

        for i in range(length - 1, len(source)):
            pv_sum[i] = np.sum(pv[i - length + 1:i + 1])
            v_sum[i] = np.sum(volume[i - length + 1:i + 1])

        return np.divide(pv_sum, v_sum, out=np.full(len(source), np.nan), where=v_sum != 0)


# Convenience alias
ta = PyneIndicators
