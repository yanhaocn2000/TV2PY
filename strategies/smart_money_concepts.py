"""
Smart Money Concepts (SMC) - Python Conversion

核心概念:
    Smart Money Concepts 是一套基于机构交易行为的价格分析方法。

主要组件:
    1. Order Blocks (订单块) - 机构大单进场区域
    2. Fair Value Gaps / FVG (公允价值缺口) - 价格快速移动留下的缺口
    3. Break of Structure / BOS (结构突破) - 趋势延续信号
    4. Change of Character / CHoCH (特性转变) - 趋势反转信号
    5. Liquidity Zones (流动性区域) - 止损聚集区
    6. Premium/Discount Zones (溢价/折价区域) - 相对价值区域
    7. Imbalance (失衡) - 买卖力量不平衡区域

TradingView 参考:
    - Smart Money Concepts [LuxAlgo]
    - ICT Concepts
    - Order Blocks & Breaker Blocks
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd


class OrderBlockType(Enum):
    """订单块类型"""
    BULLISH = "bullish"    # 看涨订单块 (需求区)
    BEARISH = "bearish"    # 看跌订单块 (供给区)


class FVGType(Enum):
    """FVG 类型"""
    BULLISH = "bullish"    # 看涨 FVG (向上缺口)
    BEARISH = "bearish"    # 看跌 FVG (向下缺口)


class StructureType(Enum):
    """结构类型"""
    BOS = "bos"            # Break of Structure (趋势延续)
    CHOCH = "choch"        # Change of Character (趋势反转)


@dataclass
class SwingPoint:
    """摆动点"""
    index: int
    price: float
    type: str  # 'high' or 'low'


@dataclass
class OrderBlock:
    """订单块"""
    type: OrderBlockType
    start_index: int
    top: float
    bottom: float
    volume: float
    valid: bool = True     # 是否仍然有效
    mitigated: bool = False  # 是否已被消化
    mitigation_index: Optional[int] = None


@dataclass
class FairValueGap:
    """公允价值缺口"""
    type: FVGType
    index: int
    top: float
    bottom: float
    filled: bool = False
    fill_percent: float = 0.0


@dataclass
class StructureBreak:
    """结构突破"""
    type: StructureType
    index: int
    price: float
    direction: int  # 1 = bullish, -1 = bearish


@dataclass
class LiquidityZone:
    """流动性区域"""
    type: str  # 'buy_side' or 'sell_side'
    price: float
    strength: int  # 触及次数
    swept: bool = False


@dataclass
class SMCResult:
    """Smart Money Concepts 计算结果"""
    # Swing Points
    swing_highs: np.ndarray
    swing_lows: np.ndarray

    # Order Blocks
    order_blocks: List[OrderBlock]
    active_bullish_ob: List[OrderBlock]
    active_bearish_ob: List[OrderBlock]

    # Fair Value Gaps
    fvg_list: List[FairValueGap]
    active_bullish_fvg: List[FairValueGap]
    active_bearish_fvg: List[FairValueGap]

    # Structure
    structure_breaks: List[StructureBreak]
    trend: np.ndarray  # 1=bullish, -1=bearish, 0=neutral

    # Liquidity
    liquidity_zones: List[LiquidityZone]

    # Premium/Discount
    equilibrium: np.ndarray
    premium_zone: np.ndarray  # 溢价区上边界
    discount_zone: np.ndarray  # 折价区下边界


class SmartMoneyConcepts:
    """
    Smart Money Concepts Analysis

    分析机构交易行为，识别高概率交易区域。

    Parameters:
        swing_length: Swing 检测周期 - 默认 10
        ob_lookback: Order Block 回溯周期 - 默认 50
        fvg_filter: FVG 最小缺口百分比 - 默认 0.0
        show_mitigated: 是否显示已消化的 OB - 默认 False
    """

    def __init__(
        self,
        swing_length: int = 10,
        ob_lookback: int = 50,
        fvg_filter: float = 0.0,
        show_mitigated: bool = False,
    ):
        self.swing_length = swing_length
        self.ob_lookback = ob_lookback
        self.fvg_filter = fvg_filter / 100.0
        self.show_mitigated = show_mitigated

    def _detect_swing_points(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[SwingPoint]]:
        """检测 Swing High/Low"""
        n = len(high)
        swing_highs = np.full(n, np.nan)
        swing_lows = np.full(n, np.nan)
        swing_points = []

        left = self.swing_length
        right = self.swing_length

        for i in range(left, n - right):
            # Swing High
            is_swing_high = True
            for j in range(1, left + 1):
                if high[i] <= high[i - j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                for j in range(1, right + 1):
                    if high[i] <= high[i + j]:
                        is_swing_high = False
                        break
            if is_swing_high:
                swing_highs[i] = high[i]
                swing_points.append(SwingPoint(i, high[i], 'high'))

            # Swing Low
            is_swing_low = True
            for j in range(1, left + 1):
                if low[i] >= low[i - j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                for j in range(1, right + 1):
                    if low[i] >= low[i + j]:
                        is_swing_low = False
                        break
            if is_swing_low:
                swing_lows[i] = low[i]
                swing_points.append(SwingPoint(i, low[i], 'low'))

        return swing_highs, swing_lows, sorted(swing_points, key=lambda x: x.index)

    def _detect_order_blocks(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        swing_points: List[SwingPoint],
    ) -> List[OrderBlock]:
        """
        检测 Order Blocks

        Bullish OB: 下跌趋势中最后一根阴线，之后价格上涨突破
        Bearish OB: 上涨趋势中最后一根阳线，之后价格下跌突破
        """
        order_blocks = []
        n = len(close)

        for point in swing_points:
            if point.type == 'low':  # 潜在 Bullish OB
                # 寻找前面的阴线
                for j in range(1, min(self.ob_lookback, point.index)):
                    idx = point.index - j
                    if close[idx] < open_[idx]:  # 阴线
                        # 检查之后是否有强势上涨
                        if point.index + 1 < n:
                            future_high = np.max(high[point.index:min(point.index + 10, n)])
                            if future_high > high[idx]:
                                ob = OrderBlock(
                                    type=OrderBlockType.BULLISH,
                                    start_index=idx,
                                    top=high[idx],
                                    bottom=low[idx],
                                    volume=volume[idx],
                                )
                                order_blocks.append(ob)
                                break

            elif point.type == 'high':  # 潜在 Bearish OB
                # 寻找前面的阳线
                for j in range(1, min(self.ob_lookback, point.index)):
                    idx = point.index - j
                    if close[idx] > open_[idx]:  # 阳线
                        # 检查之后是否有强势下跌
                        if point.index + 1 < n:
                            future_low = np.min(low[point.index:min(point.index + 10, n)])
                            if future_low < low[idx]:
                                ob = OrderBlock(
                                    type=OrderBlockType.BEARISH,
                                    start_index=idx,
                                    top=high[idx],
                                    bottom=low[idx],
                                    volume=volume[idx],
                                )
                                order_blocks.append(ob)
                                break

        return order_blocks

    def _check_ob_mitigation(
        self,
        order_blocks: List[OrderBlock],
        high: np.ndarray,
        low: np.ndarray,
    ) -> None:
        """检查 Order Block 是否被消化"""
        n = len(high)

        for ob in order_blocks:
            if ob.mitigated:
                continue

            for i in range(ob.start_index + 1, n):
                if ob.type == OrderBlockType.BULLISH:
                    # Bullish OB 被消化: 价格跌破 OB 底部
                    if low[i] < ob.bottom:
                        ob.mitigated = True
                        ob.mitigation_index = i
                        break
                else:  # BEARISH
                    # Bearish OB 被消化: 价格涨破 OB 顶部
                    if high[i] > ob.top:
                        ob.mitigated = True
                        ob.mitigation_index = i
                        break

    def _detect_fvg(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> List[FairValueGap]:
        """
        检测 Fair Value Gaps (公允价值缺口)

        Bullish FVG: 第一根K线的高点 < 第三根K线的低点
        Bearish FVG: 第一根K线的低点 > 第三根K线的高点
        """
        fvg_list = []
        n = len(close)

        for i in range(2, n):
            # Bullish FVG
            if low[i] > high[i - 2]:
                gap_size = low[i] - high[i - 2]
                avg_price = (high[i - 2] + low[i]) / 2
                gap_percent = gap_size / avg_price

                if gap_percent >= self.fvg_filter:
                    fvg = FairValueGap(
                        type=FVGType.BULLISH,
                        index=i - 1,  # 中间K线
                        top=low[i],
                        bottom=high[i - 2],
                    )
                    fvg_list.append(fvg)

            # Bearish FVG
            if high[i] < low[i - 2]:
                gap_size = low[i - 2] - high[i]
                avg_price = (low[i - 2] + high[i]) / 2
                gap_percent = gap_size / avg_price

                if gap_percent >= self.fvg_filter:
                    fvg = FairValueGap(
                        type=FVGType.BEARISH,
                        index=i - 1,
                        top=low[i - 2],
                        bottom=high[i],
                    )
                    fvg_list.append(fvg)

        return fvg_list

    def _check_fvg_fill(
        self,
        fvg_list: List[FairValueGap],
        high: np.ndarray,
        low: np.ndarray,
    ) -> None:
        """检查 FVG 是否被填补"""
        n = len(high)

        for fvg in fvg_list:
            if fvg.filled:
                continue

            gap_size = fvg.top - fvg.bottom

            for i in range(fvg.index + 1, n):
                if fvg.type == FVGType.BULLISH:
                    # 价格回撤到 FVG 区域
                    if low[i] <= fvg.top:
                        fill_depth = fvg.top - max(low[i], fvg.bottom)
                        fvg.fill_percent = min(fill_depth / gap_size, 1.0)
                        if low[i] <= fvg.bottom:
                            fvg.filled = True
                            break
                else:  # BEARISH
                    if high[i] >= fvg.bottom:
                        fill_depth = min(high[i], fvg.top) - fvg.bottom
                        fvg.fill_percent = min(fill_depth / gap_size, 1.0)
                        if high[i] >= fvg.top:
                            fvg.filled = True
                            break

    def _detect_structure_breaks(
        self,
        close: np.ndarray,
        swing_points: List[SwingPoint],
    ) -> Tuple[List[StructureBreak], np.ndarray]:
        """
        检测 BOS 和 CHoCH

        BOS (Break of Structure): 趋势延续
            - 上涨趋势中突破前高 = Bullish BOS
            - 下跌趋势中跌破前低 = Bearish BOS

        CHoCH (Change of Character): 趋势反转
            - 上涨趋势中跌破前低 = Bearish CHoCH
            - 下跌趋势中突破前高 = Bullish CHoCH
        """
        n = len(close)
        structure_breaks = []
        trend = np.zeros(n, dtype=int)

        if len(swing_points) < 2:
            return structure_breaks, trend

        current_trend = 0  # 0=neutral, 1=bullish, -1=bearish
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None

        # 初始化前两个 swing points
        for point in swing_points:
            if point.type == 'high' and last_high is None:
                last_high = point
            elif point.type == 'low' and last_low is None:
                last_low = point
            if last_high and last_low:
                break

        swing_idx = 0
        for point in swing_points:
            swing_idx += 1

            if point.type == 'high':
                if last_high is not None:
                    # 检测结构突破
                    for i in range(last_high.index + 1, min(point.index + 1, n)):
                        if close[i] > last_high.price:
                            if current_trend == -1:
                                # 下跌趋势中突破前高 = CHoCH
                                sb = StructureBreak(
                                    type=StructureType.CHOCH,
                                    index=i,
                                    price=last_high.price,
                                    direction=1,
                                )
                                structure_breaks.append(sb)
                                current_trend = 1
                            elif current_trend == 1:
                                # 上涨趋势延续 = BOS
                                sb = StructureBreak(
                                    type=StructureType.BOS,
                                    index=i,
                                    price=last_high.price,
                                    direction=1,
                                )
                                structure_breaks.append(sb)
                            else:
                                current_trend = 1
                            break
                last_high = point

            else:  # low
                if last_low is not None:
                    for i in range(last_low.index + 1, min(point.index + 1, n)):
                        if close[i] < last_low.price:
                            if current_trend == 1:
                                # 上涨趋势中跌破前低 = CHoCH
                                sb = StructureBreak(
                                    type=StructureType.CHOCH,
                                    index=i,
                                    price=last_low.price,
                                    direction=-1,
                                )
                                structure_breaks.append(sb)
                                current_trend = -1
                            elif current_trend == -1:
                                # 下跌趋势延续 = BOS
                                sb = StructureBreak(
                                    type=StructureType.BOS,
                                    index=i,
                                    price=last_low.price,
                                    direction=-1,
                                )
                                structure_breaks.append(sb)
                            else:
                                current_trend = -1
                            break
                last_low = point

        # 填充趋势数组
        current = 0
        sb_idx = 0
        for i in range(n):
            while sb_idx < len(structure_breaks) and structure_breaks[sb_idx].index <= i:
                current = structure_breaks[sb_idx].direction
                sb_idx += 1
            trend[i] = current

        return structure_breaks, trend

    def _detect_liquidity_zones(
        self,
        high: np.ndarray,
        low: np.ndarray,
        swing_points: List[SwingPoint],
    ) -> List[LiquidityZone]:
        """
        检测流动性区域

        Equal Highs/Lows = 流动性聚集区
        """
        liquidity_zones = []
        tolerance = 0.001  # 0.1% 容差

        # 检测 Equal Highs (Buy Side Liquidity)
        high_points = [p for p in swing_points if p.type == 'high']
        for i, p1 in enumerate(high_points):
            strength = 1
            for j, p2 in enumerate(high_points):
                if i != j:
                    if abs(p1.price - p2.price) / p1.price < tolerance:
                        strength += 1

            if strength >= 2:
                zone = LiquidityZone(
                    type='buy_side',
                    price=p1.price,
                    strength=strength,
                )
                liquidity_zones.append(zone)

        # 检测 Equal Lows (Sell Side Liquidity)
        low_points = [p for p in swing_points if p.type == 'low']
        for i, p1 in enumerate(low_points):
            strength = 1
            for j, p2 in enumerate(low_points):
                if i != j:
                    if abs(p1.price - p2.price) / p1.price < tolerance:
                        strength += 1

            if strength >= 2:
                zone = LiquidityZone(
                    type='sell_side',
                    price=p1.price,
                    strength=strength,
                )
                liquidity_zones.append(zone)

        return liquidity_zones

    def _calculate_premium_discount(
        self,
        high: np.ndarray,
        low: np.ndarray,
        lookback: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 Premium/Discount 区域

        Equilibrium = (最高价 + 最低价) / 2
        Premium = 价格高于 Equilibrium (卖出区域)
        Discount = 价格低于 Equilibrium (买入区域)
        """
        n = len(high)
        equilibrium = np.full(n, np.nan)
        premium = np.full(n, np.nan)
        discount = np.full(n, np.nan)

        for i in range(lookback - 1, n):
            period_high = np.max(high[i - lookback + 1:i + 1])
            period_low = np.min(low[i - lookback + 1:i + 1])

            eq = (period_high + period_low) / 2
            equilibrium[i] = eq

            # Premium zone: 61.8% - 100% of range
            premium[i] = period_low + (period_high - period_low) * 0.618

            # Discount zone: 0% - 38.2% of range
            discount[i] = period_low + (period_high - period_low) * 0.382

        return equilibrium, premium, discount

    def calculate(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> SMCResult:
        """计算所有 Smart Money Concepts"""
        open_ = np.asarray(open_, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)
        volume = np.asarray(volume, dtype=float)

        # 1. Swing Points
        swing_highs, swing_lows, swing_points = self._detect_swing_points(high, low)

        # 2. Order Blocks
        order_blocks = self._detect_order_blocks(
            open_, high, low, close, volume, swing_points
        )
        self._check_ob_mitigation(order_blocks, high, low)

        active_bullish_ob = [
            ob for ob in order_blocks
            if ob.type == OrderBlockType.BULLISH and (not ob.mitigated or self.show_mitigated)
        ]
        active_bearish_ob = [
            ob for ob in order_blocks
            if ob.type == OrderBlockType.BEARISH and (not ob.mitigated or self.show_mitigated)
        ]

        # 3. Fair Value Gaps
        fvg_list = self._detect_fvg(high, low, close)
        self._check_fvg_fill(fvg_list, high, low)

        active_bullish_fvg = [
            fvg for fvg in fvg_list
            if fvg.type == FVGType.BULLISH and not fvg.filled
        ]
        active_bearish_fvg = [
            fvg for fvg in fvg_list
            if fvg.type == FVGType.BEARISH and not fvg.filled
        ]

        # 4. Structure Breaks (BOS & CHoCH)
        structure_breaks, trend = self._detect_structure_breaks(close, swing_points)

        # 5. Liquidity Zones
        liquidity_zones = self._detect_liquidity_zones(high, low, swing_points)

        # 6. Premium/Discount
        equilibrium, premium, discount = self._calculate_premium_discount(high, low)

        return SMCResult(
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            order_blocks=order_blocks,
            active_bullish_ob=active_bullish_ob,
            active_bearish_ob=active_bearish_ob,
            fvg_list=fvg_list,
            active_bullish_fvg=active_bullish_fvg,
            active_bearish_fvg=active_bearish_fvg,
            structure_breaks=structure_breaks,
            trend=trend,
            liquidity_zones=liquidity_zones,
            equilibrium=equilibrium,
            premium_zone=premium,
            discount_zone=discount,
        )

    def get_signals(
        self,
        result: SMCResult,
        close: np.ndarray,
        index: int,
    ) -> dict:
        """
        获取当前K线的交易信号

        Returns:
            dict with keys:
                - trend: 当前趋势
                - in_premium: 是否在溢价区
                - in_discount: 是否在折价区
                - near_bullish_ob: 附近的看涨订单块
                - near_bearish_ob: 附近的看跌订单块
                - near_bullish_fvg: 附近的看涨 FVG
                - near_bearish_fvg: 附近的看跌 FVG
                - recent_bos: 最近的 BOS
                - recent_choch: 最近的 CHoCH
        """
        current_price = close[index]
        n = len(close)

        # 趋势
        trend = result.trend[index]

        # Premium/Discount
        in_premium = current_price > result.premium_zone[index] if not np.isnan(result.premium_zone[index]) else False
        in_discount = current_price < result.discount_zone[index] if not np.isnan(result.discount_zone[index]) else False

        # 附近的 Order Blocks
        near_bullish_ob = []
        for ob in result.active_bullish_ob:
            if ob.bottom <= current_price <= ob.top * 1.01:  # 接近或在 OB 内
                near_bullish_ob.append(ob)

        near_bearish_ob = []
        for ob in result.active_bearish_ob:
            if ob.bottom * 0.99 <= current_price <= ob.top:
                near_bearish_ob.append(ob)

        # 附近的 FVG
        near_bullish_fvg = []
        for fvg in result.active_bullish_fvg:
            if fvg.bottom <= current_price <= fvg.top * 1.01:
                near_bullish_fvg.append(fvg)

        near_bearish_fvg = []
        for fvg in result.active_bearish_fvg:
            if fvg.bottom * 0.99 <= current_price <= fvg.top:
                near_bearish_fvg.append(fvg)

        # 最近的结构突破
        recent_bos = None
        recent_choch = None
        for sb in reversed(result.structure_breaks):
            if sb.index <= index:
                if sb.type == StructureType.BOS and recent_bos is None:
                    recent_bos = sb
                elif sb.type == StructureType.CHOCH and recent_choch is None:
                    recent_choch = sb
            if recent_bos and recent_choch:
                break

        return {
            'trend': trend,
            'in_premium': in_premium,
            'in_discount': in_discount,
            'near_bullish_ob': near_bullish_ob,
            'near_bearish_ob': near_bearish_ob,
            'near_bullish_fvg': near_bullish_fvg,
            'near_bearish_fvg': near_bearish_fvg,
            'recent_bos': recent_bos,
            'recent_choch': recent_choch,
        }


class SMCPyneCore:
    """PyneCore 兼容的 Smart Money Concepts 实现"""

    def __init__(
        self,
        swing_length: int = 10,
        ob_lookback: int = 50,
    ):
        self.smc = SmartMoneyConcepts(swing_length, ob_lookback)

    def __call__(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        result = self.smc.calculate(
            open_.values, high.values, low.values, close.values, volume.values
        )

        return pd.DataFrame({
            "swing_high": result.swing_highs,
            "swing_low": result.swing_lows,
            "trend": result.trend,
            "equilibrium": result.equilibrium,
            "premium": result.premium_zone,
            "discount": result.discount_zone,
        }, index=close.index)


# TradingView Pine Script 参考
PINE_SCRIPT_SMC = '''
//@version=5
indicator("Smart Money Concepts", overlay=true, max_boxes_count=500, max_lines_count=500)

// 输入参数
swingLength = input.int(10, "Swing Length")
showOB = input.bool(true, "Show Order Blocks")
showFVG = input.bool(true, "Show Fair Value Gaps")
showStructure = input.bool(true, "Show Structure")

// Swing Detection
swingHigh = ta.pivothigh(high, swingLength, swingLength)
swingLow = ta.pivotlow(low, swingLength, swingLength)

// Fair Value Gap Detection
bullishFVG = low > high[2]
bearishFVG = high < low[2]

// Structure Break
var float lastHigh = na
var float lastLow = na
var int trend = 0

if not na(swingHigh)
    if close > lastHigh and trend == -1
        // CHoCH - Bullish
        label.new(bar_index, low, "CHoCH", color=color.green)
        trend := 1
    else if close > lastHigh and trend == 1
        // BOS - Bullish continuation
        label.new(bar_index, low, "BOS", color=color.blue)
    lastHigh := swingHigh

if not na(swingLow)
    if close < lastLow and trend == 1
        // CHoCH - Bearish
        label.new(bar_index, high, "CHoCH", color=color.red)
        trend := -1
    else if close < lastLow and trend == -1
        // BOS - Bearish continuation
        label.new(bar_index, high, "BOS", color=color.orange)
    lastLow := swingLow

// Order Block (简化)
bullishOB = close[1] < open[1] and close > open and close > high[1]
bearishOB = close[1] > open[1] and close < open and close < low[1]

// 绘制 FVG
if showFVG and bullishFVG
    box.new(bar_index - 1, low, bar_index, high[2], bgcolor=color.new(color.green, 80))

if showFVG and bearishFVG
    box.new(bar_index - 1, low[2], bar_index, high, bgcolor=color.new(color.red, 80))

// Premium/Discount
lookback = 50
periodHigh = ta.highest(high, lookback)
periodLow = ta.lowest(low, lookback)
equilibrium = (periodHigh + periodLow) / 2

plot(equilibrium, "Equilibrium", color=color.gray)
'''


def main():
    print("=" * 60)
    print("Smart Money Concepts - Python Implementation")
    print("=" * 60)

    np.random.seed(42)
    n = 200

    # 生成带趋势变化的数据
    base_price = 100.0
    trend = np.zeros(n)
    trend[:70] = np.linspace(0, 15, 70)      # 上涨
    trend[70:130] = np.linspace(15, 5, 60)   # 下跌
    trend[130:] = np.linspace(5, 20, 70)     # 上涨

    noise = np.cumsum(np.random.randn(n) * 0.3)
    close = base_price + trend + noise

    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.5
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.5
    volume = 1000000 + np.abs(np.random.randn(n)) * 500000

    # 计算 SMC
    smc = SmartMoneyConcepts(swing_length=5, ob_lookback=30)
    result = smc.calculate(open_, high, low, close, volume)

    print(f"\n分析范围: {n} 根K线")
    print(f"价格范围: {low.min():.2f} - {high.max():.2f}")

    # Swing Points
    swing_high_count = np.sum(~np.isnan(result.swing_highs))
    swing_low_count = np.sum(~np.isnan(result.swing_lows))
    print(f"\nSwing Points:")
    print(f"  Swing Highs: {swing_high_count}")
    print(f"  Swing Lows: {swing_low_count}")

    # Order Blocks
    print(f"\nOrder Blocks:")
    print(f"  Total: {len(result.order_blocks)}")
    print(f"  Active Bullish OB: {len(result.active_bullish_ob)}")
    print(f"  Active Bearish OB: {len(result.active_bearish_ob)}")

    for ob in result.active_bullish_ob[-3:]:
        print(f"    Bullish OB @ Bar {ob.start_index}: {ob.bottom:.2f} - {ob.top:.2f}")

    # FVG
    print(f"\nFair Value Gaps:")
    print(f"  Total: {len(result.fvg_list)}")
    print(f"  Active Bullish FVG: {len(result.active_bullish_fvg)}")
    print(f"  Active Bearish FVG: {len(result.active_bearish_fvg)}")

    for fvg in result.active_bullish_fvg[-3:]:
        print(f"    Bullish FVG @ Bar {fvg.index}: {fvg.bottom:.2f} - {fvg.top:.2f}")

    # Structure Breaks
    print(f"\nStructure Breaks:")
    bos_count = sum(1 for sb in result.structure_breaks if sb.type == StructureType.BOS)
    choch_count = sum(1 for sb in result.structure_breaks if sb.type == StructureType.CHOCH)
    print(f"  BOS: {bos_count}")
    print(f"  CHoCH: {choch_count}")

    for sb in result.structure_breaks[-5:]:
        direction = "Bullish" if sb.direction == 1 else "Bearish"
        print(f"    {sb.type.value.upper()} @ Bar {sb.index}: {direction} @ {sb.price:.2f}")

    # Liquidity Zones
    print(f"\nLiquidity Zones:")
    buy_side = [z for z in result.liquidity_zones if z.type == 'buy_side']
    sell_side = [z for z in result.liquidity_zones if z.type == 'sell_side']
    print(f"  Buy Side (Equal Highs): {len(buy_side)}")
    print(f"  Sell Side (Equal Lows): {len(sell_side)}")

    # Current State
    print(f"\n当前状态 (Bar {n-1}):")
    signals = smc.get_signals(result, close, n - 1)
    trend_str = "Bullish" if signals['trend'] == 1 else "Bearish" if signals['trend'] == -1 else "Neutral"
    print(f"  趋势: {trend_str}")
    print(f"  当前价格: {close[-1]:.2f}")
    print(f"  Equilibrium: {result.equilibrium[-1]:.2f}")
    print(f"  Premium Zone: > {result.premium_zone[-1]:.2f}")
    print(f"  Discount Zone: < {result.discount_zone[-1]:.2f}")
    print(f"  在溢价区: {signals['in_premium']}")
    print(f"  在折价区: {signals['in_discount']}")


if __name__ == "__main__":
    main()
