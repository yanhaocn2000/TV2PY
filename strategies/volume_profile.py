"""
Volume Profile - Python Conversion

核心算法:
    Volume Profile 显示在不同价格水平的成交量分布。

    由于标准 OHLCV 数据不提供逐笔成交，我们使用近似方法:
    - 假设每根K线的成交量在 High-Low 区间内均匀分布
    - 或使用三角分布 (更多成交量集中在收盘价附近)

关键概念:
    - POC (Point of Control): 成交量最大的价格水平
    - Value Area: 包含 70% 成交量的价格区间
    - VAH (Value Area High): 价值区间上边界
    - VAL (Value Area Low): 价值区间下边界
    - HVN (High Volume Node): 高成交量区域
    - LVN (Low Volume Node): 低成交量区域

TradingView 对应:
    - Volume Profile Visible Range
    - Volume Profile Fixed Range
    - Volume Profile Session Volume
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd


class DistributionMethod(Enum):
    """成交量分布方法"""
    UNIFORM = "uniform"           # 均匀分布
    TRIANGULAR = "triangular"     # 三角分布 (集中在收盘价)
    OHLC_WEIGHTED = "ohlc"        # OHLC 加权分布


@dataclass
class VolumeProfileResult:
    """Volume Profile 计算结果"""
    price_levels: np.ndarray      # 价格水平
    volume_at_price: np.ndarray   # 每个价格水平的成交量
    poc_price: float              # Point of Control 价格
    poc_volume: float             # POC 成交量
    vah: float                    # Value Area High
    val: float                    # Value Area Low
    value_area_volume: float      # Value Area 总成交量
    total_volume: float           # 总成交量
    hvn_prices: np.ndarray        # High Volume Node 价格
    lvn_prices: np.ndarray        # Low Volume Node 价格


class VolumeProfile:
    """
    Volume Profile Indicator

    分析价格-成交量分布，识别重要支撑阻力位。

    Parameters:
        row_size: 价格分箱数量 - 默认 24
        value_area_percent: Value Area 百分比 - 默认 70%
        distribution: 成交量分布方法 - 默认 triangular
    """

    def __init__(
        self,
        row_size: int = 24,
        value_area_percent: float = 70.0,
        distribution: DistributionMethod = DistributionMethod.TRIANGULAR,
    ):
        self.row_size = row_size
        self.value_area_percent = value_area_percent / 100.0
        self.distribution = distribution

    def _distribute_volume_uniform(
        self,
        high: float,
        low: float,
        volume: float,
        price_levels: np.ndarray,
        bin_size: float,
    ) -> np.ndarray:
        """均匀分布成交量"""
        contributions = np.zeros(len(price_levels))

        for i, level in enumerate(price_levels):
            level_low = level - bin_size / 2
            level_high = level + bin_size / 2

            # 计算重叠区域
            overlap_low = max(low, level_low)
            overlap_high = min(high, level_high)

            if overlap_high > overlap_low:
                bar_range = high - low
                if bar_range > 0:
                    overlap_ratio = (overlap_high - overlap_low) / bar_range
                    contributions[i] = volume * overlap_ratio

        return contributions

    def _distribute_volume_triangular(
        self,
        high: float,
        low: float,
        close: float,
        volume: float,
        price_levels: np.ndarray,
        bin_size: float,
    ) -> np.ndarray:
        """三角分布 - 更多成交量集中在收盘价附近"""
        contributions = np.zeros(len(price_levels))
        bar_range = high - low

        if bar_range <= 0:
            # 如果 high == low，所有成交量在一个价格
            idx = np.argmin(np.abs(price_levels - close))
            contributions[idx] = volume
            return contributions

        for i, level in enumerate(price_levels):
            level_low = level - bin_size / 2
            level_high = level + bin_size / 2

            overlap_low = max(low, level_low)
            overlap_high = min(high, level_high)

            if overlap_high > overlap_low:
                # 计算到收盘价的距离权重
                level_center = (overlap_low + overlap_high) / 2
                distance = abs(level_center - close)
                max_distance = bar_range

                # 三角权重: 离收盘价越近权重越高
                weight = 1.0 - (distance / max_distance) * 0.6

                overlap_ratio = (overlap_high - overlap_low) / bar_range
                contributions[i] = volume * overlap_ratio * weight

        # 归一化确保总和等于原始成交量
        total = contributions.sum()
        if total > 0:
            contributions = contributions * (volume / total)

        return contributions

    def _distribute_volume_ohlc(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        price_levels: np.ndarray,
        bin_size: float,
    ) -> np.ndarray:
        """OHLC 加权分布 - 成交量集中在 OHLC 四个价格"""
        contributions = np.zeros(len(price_levels))

        # OHLC 各占 25% 成交量
        ohlc_prices = [open_, high, low, close]
        ohlc_volumes = [volume * 0.25] * 4

        for price, vol in zip(ohlc_prices, ohlc_volumes):
            idx = np.argmin(np.abs(price_levels - price))
            contributions[idx] += vol

        return contributions

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        open_: Optional[np.ndarray] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> VolumeProfileResult:
        """
        计算 Volume Profile

        Parameters:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组
            volume: 成交量数组
            open_: 开盘价数组 (OHLC 分布方法需要)
            start_idx: 起始索引 (默认 0)
            end_idx: 结束索引 (默认 len-1)
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)

        n = len(close)

        # 确定范围
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = n - 1

        # 截取数据范围
        h = high[start_idx:end_idx + 1]
        l = low[start_idx:end_idx + 1]
        c = close[start_idx:end_idx + 1]
        v = volume[start_idx:end_idx + 1]

        if open_ is not None:
            o = np.asarray(open_, dtype=float)[start_idx:end_idx + 1]
        else:
            o = c  # 如果没有开盘价，使用收盘价

        # 确定价格范围
        price_high = np.nanmax(h)
        price_low = np.nanmin(l)
        price_range = price_high - price_low

        if price_range <= 0:
            price_range = price_high * 0.01  # 防止除零

        # 创建价格水平
        bin_size = price_range / self.row_size
        price_levels = np.linspace(
            price_low + bin_size / 2,
            price_high - bin_size / 2,
            self.row_size
        )

        # 累积每个价格水平的成交量
        volume_at_price = np.zeros(self.row_size)

        for i in range(len(c)):
            if self.distribution == DistributionMethod.UNIFORM:
                contrib = self._distribute_volume_uniform(
                    h[i], l[i], v[i], price_levels, bin_size
                )
            elif self.distribution == DistributionMethod.TRIANGULAR:
                contrib = self._distribute_volume_triangular(
                    h[i], l[i], c[i], v[i], price_levels, bin_size
                )
            else:  # OHLC
                contrib = self._distribute_volume_ohlc(
                    o[i], h[i], l[i], c[i], v[i], price_levels, bin_size
                )

            volume_at_price += contrib

        # 计算 POC (Point of Control)
        poc_idx = np.argmax(volume_at_price)
        poc_price = price_levels[poc_idx]
        poc_volume = volume_at_price[poc_idx]

        # 计算 Value Area (包含 70% 成交量的区域)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_percent

        # 从 POC 向两侧扩展直到达到目标成交量
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        va_volume = poc_volume

        while va_volume < target_volume and (va_low_idx > 0 or va_high_idx < self.row_size - 1):
            # 比较上下两侧的成交量，选择较大的一侧扩展
            vol_below = volume_at_price[va_low_idx - 1] if va_low_idx > 0 else 0
            vol_above = volume_at_price[va_high_idx + 1] if va_high_idx < self.row_size - 1 else 0

            if vol_below >= vol_above and va_low_idx > 0:
                va_low_idx -= 1
                va_volume += vol_below
            elif va_high_idx < self.row_size - 1:
                va_high_idx += 1
                va_volume += vol_above
            elif va_low_idx > 0:
                va_low_idx -= 1
                va_volume += vol_below
            else:
                break

        val = price_levels[va_low_idx] - bin_size / 2
        vah = price_levels[va_high_idx] + bin_size / 2

        # 识别 HVN 和 LVN
        avg_volume = total_volume / self.row_size
        std_volume = np.std(volume_at_price)

        hvn_threshold = avg_volume + std_volume
        lvn_threshold = avg_volume - std_volume * 0.5

        hvn_mask = volume_at_price > hvn_threshold
        lvn_mask = (volume_at_price < lvn_threshold) & (volume_at_price > 0)

        hvn_prices = price_levels[hvn_mask]
        lvn_prices = price_levels[lvn_mask]

        return VolumeProfileResult(
            price_levels=price_levels,
            volume_at_price=volume_at_price,
            poc_price=poc_price,
            poc_volume=poc_volume,
            vah=vah,
            val=val,
            value_area_volume=va_volume,
            total_volume=total_volume,
            hvn_prices=hvn_prices,
            lvn_prices=lvn_prices,
        )

    def get_support_resistance(
        self,
        result: VolumeProfileResult,
        current_price: float,
    ) -> Tuple[List[float], List[float]]:
        """
        基于 Volume Profile 获取支撑和阻力位

        Returns:
            (支撑位列表, 阻力位列表)
        """
        supports = []
        resistances = []

        # POC, VAH, VAL 都是重要水平
        key_levels = [result.poc_price, result.vah, result.val]
        key_levels.extend(result.hvn_prices.tolist())

        for level in key_levels:
            if level < current_price:
                supports.append(level)
            elif level > current_price:
                resistances.append(level)

        supports.sort(reverse=True)  # 从高到低
        resistances.sort()           # 从低到高

        return supports, resistances


class SessionVolumeProfile:
    """
    Session Volume Profile

    按交易日/时段计算 Volume Profile
    """

    def __init__(
        self,
        row_size: int = 24,
        value_area_percent: float = 70.0,
    ):
        self.vp = VolumeProfile(row_size, value_area_percent)

    def calculate_by_date(
        self,
        dates: pd.DatetimeIndex,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> dict:
        """按日期计算每个交易日的 Volume Profile"""
        results = {}

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }, index=dates)

        # 按日期分组
        for date, group in df.groupby(df.index.date):
            if len(group) > 0:
                result = self.vp.calculate(
                    group['high'].values,
                    group['low'].values,
                    group['close'].values,
                    group['volume'].values,
                )
                results[date] = result

        return results


class VolumeProfilePyneCore:
    """PyneCore 兼容的 Volume Profile 实现"""

    def __init__(
        self,
        row_size: int = 24,
        value_area_percent: float = 70.0,
        distribution: str = "triangular",
    ):
        dist_map = {
            "uniform": DistributionMethod.UNIFORM,
            "triangular": DistributionMethod.TRIANGULAR,
            "ohlc": DistributionMethod.OHLC_WEIGHTED,
        }
        self.vp = VolumeProfile(
            row_size,
            value_area_percent,
            dist_map.get(distribution, DistributionMethod.TRIANGULAR)
        )

    def __call__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        result = self.vp.calculate(
            high.values, low.values, close.values, volume.values
        )

        # 返回关键水平
        return pd.DataFrame({
            "poc": [result.poc_price] * len(close),
            "vah": [result.vah] * len(close),
            "val": [result.val] * len(close),
        }, index=close.index)


# TradingView Pine Script 参考
PINE_SCRIPT_VOLUME_PROFILE = '''
//@version=5
indicator("Volume Profile [Visible Range]", overlay=true)

// TradingView 内置 Volume Profile 使用 tick 级别数据
// 以下是简化的近似实现

rowSize = input.int(24, "Row Size")
valueAreaPercent = input.float(70, "Value Area %")

// 注意: Pine Script 中需要使用 array 来累积成交量
// 这里展示概念性代码

var float[] volumeAtPrice = array.new_float(rowSize, 0.0)
var float[] priceLevels = array.new_float(rowSize, 0.0)

// 计算价格范围
priceHigh = ta.highest(high, 100)
priceLow = ta.lowest(low, 100)
binSize = (priceHigh - priceLow) / rowSize

// 分配当前K线的成交量到价格水平
for i = 0 to rowSize - 1
    level = priceLow + (i + 0.5) * binSize
    if high >= level and low <= level
        // 简化: 均匀分布
        currentVol = array.get(volumeAtPrice, i)
        array.set(volumeAtPrice, i, currentVol + volume / rowSize)

// 找到 POC
maxVol = 0.0
pocIdx = 0
for i = 0 to rowSize - 1
    if array.get(volumeAtPrice, i) > maxVol
        maxVol := array.get(volumeAtPrice, i)
        pocIdx := i

pocPrice = priceLow + (pocIdx + 0.5) * binSize

// 绘制 POC 线
plot(pocPrice, "POC", color=color.yellow, linewidth=2)
'''


def main():
    print("=" * 60)
    print("Volume Profile - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 200

    # 生成模拟数据
    base_price = 100.0
    trend = np.sin(np.linspace(0, 4 * np.pi, n)) * 10
    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.8
    low = close - np.abs(np.random.randn(n)) * 0.8

    # 成交量 - 在价格极端时增加
    base_volume = 1000000
    volume = base_volume + np.abs(np.random.randn(n)) * 500000
    # 在价格转折点增加成交量
    volume += np.abs(np.cos(np.linspace(0, 4 * np.pi, n))) * 300000

    # 计算 Volume Profile
    vp = VolumeProfile(row_size=20, value_area_percent=70)
    result = vp.calculate(high, low, close, volume)

    print(f"\n分析范围: {n} 根K线")
    print(f"价格范围: {low.min():.2f} - {high.max():.2f}")
    print(f"\n关键水平:")
    print(f"  POC (Point of Control): {result.poc_price:.2f}")
    print(f"  POC Volume: {result.poc_volume:,.0f}")
    print(f"  Value Area High (VAH): {result.vah:.2f}")
    print(f"  Value Area Low (VAL): {result.val:.2f}")
    print(f"  Value Area Volume: {result.value_area_volume:,.0f} ({result.value_area_volume/result.total_volume*100:.1f}%)")
    print(f"  Total Volume: {result.total_volume:,.0f}")

    print(f"\nHigh Volume Nodes (HVN):")
    for price in result.hvn_prices[:5]:
        print(f"  {price:.2f}")

    print(f"\nLow Volume Nodes (LVN):")
    for price in result.lvn_prices[:5]:
        print(f"  {price:.2f}")

    # 获取支撑阻力
    current_price = close[-1]
    supports, resistances = vp.get_support_resistance(result, current_price)

    print(f"\n当前价格: {current_price:.2f}")
    print(f"支撑位: {[f'{s:.2f}' for s in supports[:3]]}")
    print(f"阻力位: {[f'{r:.2f}' for r in resistances[:3]]}")

    # Volume Profile 分布
    print(f"\n成交量分布 (价格 -> 成交量):")
    print("-" * 40)
    max_vol = result.volume_at_price.max()
    for i in range(len(result.price_levels) - 1, -1, -1):
        price = result.price_levels[i]
        vol = result.volume_at_price[i]
        bar_len = int(vol / max_vol * 30)
        marker = " <-- POC" if abs(price - result.poc_price) < 0.01 else ""
        print(f"  {price:7.2f} | {'█' * bar_len}{marker}")


if __name__ == "__main__":
    main()
