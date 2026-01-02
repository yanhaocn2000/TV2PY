"""
多时间框架 (MTF) 实现

替代 Pine Script 的 request.security 功能

原理:
- Pine Script: request.security("BTCUSDT", "1D", close) 实时获取日线收盘价
- Python: 预先将 4H 数据重采样为 1D，然后合并到 4H 数据中
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    source_timeframe: str = None
) -> pd.DataFrame:
    """
    将 OHLCV 数据重采样到目标时间框架

    Args:
        df: 原始数据，需要有 timestamp, open, high, low, close, volume
        target_timeframe: 目标时间框架 ('1H', '4H', '1D', '1W')
        source_timeframe: 原始时间框架 (可选，用于验证)

    Returns:
        重采样后的 DataFrame

    Example:
        # 4H -> 1D
        daily = resample_ohlcv(df_4h, '1D')
    """
    df = df.copy()

    # 确保 timestamp 是 datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.set_index('timestamp')

    # 时间框架映射
    tf_map = {
        '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1H': '1H', '4H': '4H', '1D': '1D', '1W': '1W',
        '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W',
    }

    resample_rule = tf_map.get(target_timeframe, target_timeframe)

    # OHLCV 重采样规则
    resampled = df.resample(resample_rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    resampled = resampled.reset_index()
    return resampled


def merge_timeframes(
    base_df: pd.DataFrame,
    higher_tf_df: pd.DataFrame,
    columns: List[str] = None,
    prefix: str = 'htf_'
) -> pd.DataFrame:
    """
    将高时间框架数据合并到基础时间框架

    关键: 使用 merge_asof 确保不发生未来函数 (lookahead bias)
    - 每根 4H K线只能看到已完成的日线数据
    - 不能看到当前日线的收盘价 (因为还没收盘)

    Args:
        base_df: 基础时间框架数据 (如 4H)
        higher_tf_df: 高时间框架数据 (如 1D)
        columns: 要合并的列 (默认 close)
        prefix: 列名前缀

    Returns:
        合并后的 DataFrame
    """
    if columns is None:
        columns = ['close']

    base_df = base_df.copy()
    higher_tf_df = higher_tf_df.copy()

    # 确保时间戳格式
    if not pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        base_df['timestamp'] = pd.to_datetime(base_df['timestamp'])
    if not pd.api.types.is_datetime64_any_dtype(higher_tf_df['timestamp']):
        higher_tf_df['timestamp'] = pd.to_datetime(higher_tf_df['timestamp'])

    # 排序
    base_df = base_df.sort_values('timestamp')
    higher_tf_df = higher_tf_df.sort_values('timestamp')

    # 准备高时间框架数据
    htf_cols = ['timestamp'] + columns
    htf_data = higher_tf_df[htf_cols].copy()

    # 重命名列
    rename_map = {col: f'{prefix}{col}' for col in columns}
    htf_data = htf_data.rename(columns=rename_map)

    # 使用 merge_asof 进行时间对齐
    # direction='backward' 确保只使用过去的数据，避免未来函数
    merged = pd.merge_asof(
        base_df,
        htf_data,
        on='timestamp',
        direction='backward'  # 关键: 只看过去已完成的K线
    )

    return merged


def calculate_htf_indicator(
    df: pd.DataFrame,
    indicator_func,
    higher_tf: str,
    *args,
    **kwargs
) -> np.ndarray:
    """
    在高时间框架上计算指标，然后映射回基础时间框架

    Args:
        df: 基础时间框架数据
        indicator_func: 指标计算函数
        higher_tf: 高时间框架 ('1D', '1W')
        *args, **kwargs: 传给指标函数的参数

    Returns:
        映射到基础时间框架的指标值

    Example:
        # 在日线上计算 EMA，然后映射到 4H
        daily_ema = calculate_htf_indicator(df_4h, tv_ema, '1D', 20)
    """
    # 重采样到高时间框架
    htf_df = resample_ohlcv(df, higher_tf)

    # 计算指标
    htf_values = indicator_func(htf_df['close'].values, *args, **kwargs)
    htf_df['indicator'] = htf_values

    # 合并回基础时间框架
    merged = merge_timeframes(df, htf_df, columns=['indicator'], prefix='')

    return merged['indicator'].values


# =============================================================================
# 示例: 模拟 request.security
# =============================================================================

class RequestSecurity:
    """
    模拟 Pine Script 的 request.security 功能

    用法:
        security = RequestSecurity(df_4h)

        # 获取日线收盘价 (类似 request.security("", "1D", close))
        daily_close = security.get('1D', 'close')

        # 获取日线 EMA (类似 request.security("", "1D", ta.ema(close, 20)))
        daily_ema = security.get_indicator('1D', tv_ema, 20)
    """

    def __init__(self, base_df: pd.DataFrame):
        """
        Args:
            base_df: 基础时间框架数据
        """
        self.base_df = base_df.copy()
        self._cache: Dict[str, pd.DataFrame] = {}

    def _get_resampled(self, timeframe: str) -> pd.DataFrame:
        """获取或创建重采样数据 (带缓存)"""
        if timeframe not in self._cache:
            self._cache[timeframe] = resample_ohlcv(self.base_df, timeframe)
        return self._cache[timeframe]

    def get(self, timeframe: str, column: str = 'close') -> np.ndarray:
        """
        获取高时间框架的 OHLCV 数据

        类似: request.security(syminfo.tickerid, "1D", close)
        """
        htf_df = self._get_resampled(timeframe)
        merged = merge_timeframes(self.base_df, htf_df, columns=[column])
        return merged[f'htf_{column}'].values

    def get_indicator(
        self,
        timeframe: str,
        indicator_func,
        *args,
        column: str = 'close',
        **kwargs
    ) -> np.ndarray:
        """
        在高时间框架上计算指标

        类似: request.security(syminfo.tickerid, "1D", ta.ema(close, 20))
        """
        htf_df = self._get_resampled(timeframe)
        values = indicator_func(htf_df[column].values, *args, **kwargs)
        htf_df['_indicator'] = values
        merged = merge_timeframes(self.base_df, htf_df, columns=['_indicator'])
        return merged['htf__indicator'].values


# =============================================================================
# 完整示例
# =============================================================================

def example_mtf_strategy():
    """
    示例: 多时间框架策略

    策略逻辑:
    - 日线 EMA200 判断大趋势
    - 4H 信号入场
    """
    print("=" * 60)
    print("多时间框架策略示例")
    print("=" * 60)

    # 创建模拟数据 (4H)
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='4H')

    # 模拟价格
    returns = np.random.randn(n_bars) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.01)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    df_4h = pd.DataFrame({
        'timestamp': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_bars)
    })

    print(f"4H 数据: {len(df_4h)} 根K线")
    print(f"时间范围: {df_4h['timestamp'].min()} - {df_4h['timestamp'].max()}")

    # 重采样到日线
    df_daily = resample_ohlcv(df_4h, '1D')
    print(f"日线数据: {len(df_daily)} 根K线")

    # 方法1: 直接合并
    merged = merge_timeframes(df_4h, df_daily, columns=['close', 'high', 'low'])
    print(f"\n合并后数据预览:")
    print(merged[['timestamp', 'close', 'htf_close']].tail(10))

    # 方法2: 使用 RequestSecurity 类
    security = RequestSecurity(df_4h)

    # 获取日线收盘价
    daily_close = security.get('1D', 'close')
    print(f"\n日线收盘价 (最后5个): {daily_close[-5:]}")

    # 计算日线 EMA (简单实现)
    def simple_ema(src, length):
        alpha = 2 / (length + 1)
        result = np.zeros_like(src)
        result[0] = src[0]
        for i in range(1, len(src)):
            if np.isnan(src[i]):
                result[i] = result[i-1]
            else:
                result[i] = alpha * src[i] + (1 - alpha) * result[i-1]
        return result

    daily_ema20 = security.get_indicator('1D', simple_ema, 20)
    print(f"日线 EMA20 (最后5个): {daily_ema20[-5:]}")

    # 策略信号
    # 4H 收盘价 > 日线 EMA20 = 多头趋势
    trend_up = df_4h['close'].values > daily_ema20
    print(f"\n多头趋势 K线数: {np.sum(trend_up)} / {len(trend_up)}")

    return merged


if __name__ == "__main__":
    example_mtf_strategy()
