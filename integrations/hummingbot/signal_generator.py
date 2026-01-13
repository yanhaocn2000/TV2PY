"""
TV2PY Signal Generator for Hummingbot

提供预配置的信号生成器，用于 Hummingbot 策略

信号类型:
    - TREND_FOLLOWING: 趋势跟踪
    - MEAN_REVERSION: 均值回归
    - SMART_MONEY: 智能资金
    - MOMENTUM: 动量交易
    - BREAKOUT: 突破交易
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from enum import Enum

import numpy as np
import pandas as pd

from .indicator_adapter import IndicatorAdapter, CandleData, IndicatorResult


class SignalType(Enum):
    """信号类型"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
    CLOSE_LONG = 2
    CLOSE_SHORT = -2


class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    SMART_MONEY = "smart_money"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    CUSTOM = "custom"


@dataclass
class TradingSignal:
    """
    交易信号

    Attributes:
        signal_type: 信号类型 (LONG, SHORT, NEUTRAL, CLOSE_LONG, CLOSE_SHORT)
        price: 当前价格
        strength: 信号强度 (0-1)
        confidence: 信号置信度 (0-1)
        stop_loss: 建议止损价
        take_profit: 建议止盈价
        position_size: 建议仓位比例 (0-1)
        reason: 信号原因
        indicators: 指标详情
    """
    signal_type: SignalType
    price: float
    strength: float = 0.0
    confidence: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0
    reason: str = ""
    indicators: Dict[str, Any] = None

    def __post_init__(self):
        if self.indicators is None:
            self.indicators = {}

    @property
    def is_entry(self) -> bool:
        return self.signal_type in [SignalType.LONG, SignalType.SHORT]

    @property
    def is_exit(self) -> bool:
        return self.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]

    @property
    def is_long(self) -> bool:
        return self.signal_type == SignalType.LONG

    @property
    def is_short(self) -> bool:
        return self.signal_type == SignalType.SHORT


class TV2PYSignalGenerator:
    """
    TV2PY 信号生成器

    预配置的策略组合，用于 Hummingbot 策略

    使用示例:
        # 创建信号生成器
        signal_gen = TV2PYSignalGenerator(strategy_type=StrategyType.SMART_MONEY)

        # 从 Hummingbot candles 获取信号
        signal = signal_gen.get_signal(candles_df)

        if signal.is_long:
            # 执行买入
            pass
        elif signal.is_short:
            # 执行卖出
            pass
    """

    # 策略预设配置
    STRATEGY_CONFIGS = {
        StrategyType.TREND_FOLLOWING: {
            "indicators": [
                ("supertrend", {"atr_period": 10, "multiplier": 3.0}),
                ("squeeze", {"bb_length": 20}),
                ("choppiness", {"length": 14}),
                ("atr", {"length": 14}),
            ],
            "description": "趋势跟踪策略：SuperTrend + Squeeze + Choppiness",
        },
        StrategyType.MEAN_REVERSION: {
            "indicators": [
                ("bollinger", {"length": 20, "mult": 2.0}),
                ("choppiness", {"length": 14}),
                ("volume_profile", {"row_size": 24}),
                ("atr", {"length": 14}),
            ],
            "description": "均值回归策略：Bollinger Bands + Volume Profile",
        },
        StrategyType.SMART_MONEY: {
            "indicators": [
                ("smc", {"swing_length": 10, "ob_lookback": 50}),
                ("volume_profile", {"row_size": 24}),
                ("squeeze", {"bb_length": 20}),
                ("atr", {"length": 14}),
            ],
            "description": "智能资金策略：SMC + Volume Profile + Squeeze",
        },
        StrategyType.MOMENTUM: {
            "indicators": [
                ("squeeze", {"bb_length": 20}),
                ("supertrend", {"atr_period": 10}),
                ("atr", {"length": 14}),
            ],
            "description": "动量策略：Squeeze Momentum + SuperTrend",
        },
        StrategyType.BREAKOUT: {
            "indicators": [
                ("squeeze", {"bb_length": 20}),
                ("bollinger", {"length": 20}),
                ("choppiness", {"length": 14}),
                ("atr", {"length": 14}),
            ],
            "description": "突破策略：Squeeze + Bollinger Bands",
        },
    }

    def __init__(
        self,
        strategy_type: StrategyType = StrategyType.SMART_MONEY,
        risk_per_trade: float = 0.02,
        min_confidence: float = 0.5,
        custom_indicators: Optional[List[tuple]] = None,
    ):
        """
        初始化信号生成器

        Args:
            strategy_type: 策略类型
            risk_per_trade: 每笔交易风险比例 (默认 2%)
            min_confidence: 最小信号置信度
            custom_indicators: 自定义指标列表 [(name, kwargs), ...]
        """
        self.strategy_type = strategy_type
        self.risk_per_trade = risk_per_trade
        self.min_confidence = min_confidence

        # 初始化指标适配器
        self.adapter = IndicatorAdapter()

        # 添加指标
        if custom_indicators:
            for name, kwargs in custom_indicators:
                self.adapter.add_indicator(name, **kwargs)
        elif strategy_type in self.STRATEGY_CONFIGS:
            config = self.STRATEGY_CONFIGS[strategy_type]
            for name, kwargs in config["indicators"]:
                self.adapter.add_indicator(name, **kwargs)

    def get_signal(
        self,
        candles: Union[CandleData, pd.DataFrame, List],
        current_position: int = 0,
    ) -> TradingSignal:
        """
        获取交易信号

        Args:
            candles: K线数据
            current_position: 当前持仓 (1=多头, -1=空头, 0=无持仓)

        Returns:
            TradingSignal: 交易信号
        """
        # 计算所有指标
        results = self.adapter.calculate(candles)

        # 获取当前价格
        if isinstance(candles, pd.DataFrame):
            data = CandleData.from_dataframe(candles)
        elif isinstance(candles, list):
            data = CandleData.from_hummingbot_candles(candles)
        else:
            data = candles

        current_price = float(data.close[-1])

        # 根据策略类型生成信号
        if self.strategy_type == StrategyType.TREND_FOLLOWING:
            return self._trend_following_signal(results, current_price, current_position)
        elif self.strategy_type == StrategyType.MEAN_REVERSION:
            return self._mean_reversion_signal(results, current_price, current_position)
        elif self.strategy_type == StrategyType.SMART_MONEY:
            return self._smart_money_signal(results, current_price, current_position)
        elif self.strategy_type == StrategyType.MOMENTUM:
            return self._momentum_signal(results, current_price, current_position)
        elif self.strategy_type == StrategyType.BREAKOUT:
            return self._breakout_signal(results, current_price, current_position)
        else:
            # 默认使用综合信号
            combined = self.adapter.get_combined_signal(results)
            return self._create_signal_from_combined(combined, current_price, results)

    def _get_indicator_result(self, results: List[IndicatorResult], name: str) -> Optional[IndicatorResult]:
        """获取指定指标的结果"""
        for r in results:
            if r.name == name:
                return r
        return None

    def _trend_following_signal(
        self,
        results: List[IndicatorResult],
        price: float,
        position: int,
    ) -> TradingSignal:
        """趋势跟踪策略信号"""
        supertrend = self._get_indicator_result(results, "supertrend")
        squeeze = self._get_indicator_result(results, "squeeze")
        choppiness = self._get_indicator_result(results, "choppiness")
        atr = self._get_indicator_result(results, "atr")

        signal_type = SignalType.NEUTRAL
        strength = 0.0
        confidence = 0.0
        reason = ""
        stop_loss = None
        take_profit = None

        # 检查是否为趋势市场
        is_trending = choppiness and choppiness.values.get("is_trending", False)

        if supertrend and squeeze:
            st_direction = supertrend.values.get("direction", 0)
            squeeze_on = squeeze.values.get("squeeze_on", True)
            momentum_rising = squeeze.values.get("momentum_rising", False)
            momentum = squeeze.values.get("momentum", 0)

            # 趋势市场 + SuperTrend 方向 + Squeeze 释放
            if is_trending and not squeeze_on:
                if st_direction == 1 and momentum > 0 and momentum_rising:
                    signal_type = SignalType.LONG
                    strength = 0.8
                    confidence = 0.7
                    reason = "趋势市场 + SuperTrend 多头 + Squeeze 释放上涨"
                    if supertrend:
                        stop_loss = supertrend.values.get("supertrend")
                elif st_direction == -1 and momentum < 0 and not momentum_rising:
                    signal_type = SignalType.SHORT
                    strength = 0.8
                    confidence = 0.7
                    reason = "趋势市场 + SuperTrend 空头 + Squeeze 释放下跌"
                    if supertrend:
                        stop_loss = supertrend.values.get("supertrend")

            # 出场信号
            if position == 1 and st_direction == -1:
                signal_type = SignalType.CLOSE_LONG
                reason = "SuperTrend 转空，平多"
            elif position == -1 and st_direction == 1:
                signal_type = SignalType.CLOSE_SHORT
                reason = "SuperTrend 转多，平空"

        # 计算止盈
        if atr and stop_loss:
            atr_value = atr.values.get("atr", 0)
            if signal_type == SignalType.LONG:
                take_profit = price + atr_value * 3
            elif signal_type == SignalType.SHORT:
                take_profit = price - atr_value * 3

        return TradingSignal(
            signal_type=signal_type,
            price=price,
            strength=strength,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss) if stop_loss else 0,
            reason=reason,
            indicators={r.name: r.values for r in results},
        )

    def _mean_reversion_signal(
        self,
        results: List[IndicatorResult],
        price: float,
        position: int,
    ) -> TradingSignal:
        """均值回归策略信号"""
        bollinger = self._get_indicator_result(results, "bollinger")
        choppiness = self._get_indicator_result(results, "choppiness")
        volume_profile = self._get_indicator_result(results, "volume_profile")
        atr = self._get_indicator_result(results, "atr")

        signal_type = SignalType.NEUTRAL
        strength = 0.0
        confidence = 0.0
        reason = ""
        stop_loss = None
        take_profit = None

        is_ranging = choppiness and choppiness.values.get("is_ranging", False)

        if bollinger and is_ranging:
            upper = bollinger.values.get("upper", 0)
            lower = bollinger.values.get("lower", 0)
            middle = bollinger.values.get("middle", 0)

            # 价格触及下轨
            if price <= lower:
                signal_type = SignalType.LONG
                strength = 0.7
                confidence = 0.6
                reason = "震荡市场 + 价格触及布林下轨"
                stop_loss = lower - (upper - lower) * 0.1
                take_profit = middle

            # 价格触及上轨
            elif price >= upper:
                signal_type = SignalType.SHORT
                strength = 0.7
                confidence = 0.6
                reason = "震荡市场 + 价格触及布林上轨"
                stop_loss = upper + (upper - lower) * 0.1
                take_profit = middle

            # 结合 Volume Profile
            if volume_profile:
                poc = volume_profile.values.get("poc", 0)
                # 价格在 POC 附近增加置信度
                if abs(price - poc) / price < 0.02:
                    confidence += 0.1

        # 出场信号
        if position == 1 and bollinger:
            if price >= bollinger.values.get("middle", 0):
                signal_type = SignalType.CLOSE_LONG
                reason = "价格回归均线，平多"
        elif position == -1 and bollinger:
            if price <= bollinger.values.get("middle", 0):
                signal_type = SignalType.CLOSE_SHORT
                reason = "价格回归均线，平空"

        return TradingSignal(
            signal_type=signal_type,
            price=price,
            strength=strength,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss) if stop_loss else 0,
            reason=reason,
            indicators={r.name: r.values for r in results},
        )

    def _smart_money_signal(
        self,
        results: List[IndicatorResult],
        price: float,
        position: int,
    ) -> TradingSignal:
        """智能资金策略信号"""
        smc = self._get_indicator_result(results, "smc")
        volume_profile = self._get_indicator_result(results, "volume_profile")
        squeeze = self._get_indicator_result(results, "squeeze")
        atr = self._get_indicator_result(results, "atr")

        signal_type = SignalType.NEUTRAL
        strength = 0.0
        confidence = 0.0
        reason = ""
        stop_loss = None
        take_profit = None

        if smc:
            trend = smc.values.get("trend", 0)
            in_discount = smc.values.get("in_discount", False)
            in_premium = smc.values.get("in_premium", False)
            bullish_ob = smc.values.get("active_bullish_ob", 0)
            bearish_ob = smc.values.get("active_bearish_ob", 0)
            bullish_fvg = smc.values.get("active_bullish_fvg", 0)
            bearish_fvg = smc.values.get("active_bearish_fvg", 0)
            equilibrium = smc.values.get("equilibrium", price)

            # 多头信号: 折价区 + 看涨结构
            if in_discount and trend >= 0:
                if bullish_ob > 0 or bullish_fvg > 0:
                    signal_type = SignalType.LONG
                    strength = 0.8
                    confidence = 0.7
                    reason = f"SMC: 折价区 + {'订单块' if bullish_ob else 'FVG'}"

                    # Squeeze 确认
                    if squeeze and not squeeze.values.get("squeeze_on", True):
                        if squeeze.values.get("momentum", 0) > 0:
                            confidence += 0.1
                            reason += " + Squeeze 释放"

            # 空头信号: 溢价区 + 看跌结构
            elif in_premium and trend <= 0:
                if bearish_ob > 0 or bearish_fvg > 0:
                    signal_type = SignalType.SHORT
                    strength = 0.8
                    confidence = 0.7
                    reason = f"SMC: 溢价区 + {'订单块' if bearish_ob else 'FVG'}"

                    if squeeze and not squeeze.values.get("squeeze_on", True):
                        if squeeze.values.get("momentum", 0) < 0:
                            confidence += 0.1
                            reason += " + Squeeze 释放"

            # 设置止损止盈
            if atr:
                atr_value = atr.values.get("atr", 0)
                if signal_type == SignalType.LONG:
                    stop_loss = price - atr_value * 2
                    take_profit = equilibrium if equilibrium > price else price + atr_value * 3
                elif signal_type == SignalType.SHORT:
                    stop_loss = price + atr_value * 2
                    take_profit = equilibrium if equilibrium < price else price - atr_value * 3

            # 出场信号
            if position == 1:
                if in_premium or trend == -1:
                    signal_type = SignalType.CLOSE_LONG
                    reason = "SMC: 到达溢价区/趋势转空"
            elif position == -1:
                if in_discount or trend == 1:
                    signal_type = SignalType.CLOSE_SHORT
                    reason = "SMC: 到达折价区/趋势转多"

        return TradingSignal(
            signal_type=signal_type,
            price=price,
            strength=strength,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss) if stop_loss else 0,
            reason=reason,
            indicators={r.name: r.values for r in results},
        )

    def _momentum_signal(
        self,
        results: List[IndicatorResult],
        price: float,
        position: int,
    ) -> TradingSignal:
        """动量策略信号"""
        squeeze = self._get_indicator_result(results, "squeeze")
        supertrend = self._get_indicator_result(results, "supertrend")
        atr = self._get_indicator_result(results, "atr")

        signal_type = SignalType.NEUTRAL
        strength = 0.0
        confidence = 0.0
        reason = ""
        stop_loss = None
        take_profit = None

        if squeeze and supertrend:
            squeeze_on = squeeze.values.get("squeeze_on", True)
            momentum = squeeze.values.get("momentum", 0)
            momentum_rising = squeeze.values.get("momentum_rising", False)
            st_direction = supertrend.values.get("direction", 0)

            # Squeeze 释放 + 方向确认
            if not squeeze_on:
                if momentum > 0 and momentum_rising and st_direction == 1:
                    signal_type = SignalType.LONG
                    strength = min(abs(momentum) / 5, 1.0)
                    confidence = 0.75
                    reason = "Squeeze 释放 + 动量上涨 + SuperTrend 多头"
                    stop_loss = supertrend.values.get("supertrend")

                elif momentum < 0 and not momentum_rising and st_direction == -1:
                    signal_type = SignalType.SHORT
                    strength = min(abs(momentum) / 5, 1.0)
                    confidence = 0.75
                    reason = "Squeeze 释放 + 动量下跌 + SuperTrend 空头"
                    stop_loss = supertrend.values.get("supertrend")

            # 动量衰减出场
            if position == 1 and momentum < 0:
                signal_type = SignalType.CLOSE_LONG
                reason = "动量转负，平多"
            elif position == -1 and momentum > 0:
                signal_type = SignalType.CLOSE_SHORT
                reason = "动量转正，平空"

        # 止盈
        if atr and stop_loss:
            atr_value = atr.values.get("atr", 0)
            if signal_type == SignalType.LONG:
                take_profit = price + atr_value * 2.5
            elif signal_type == SignalType.SHORT:
                take_profit = price - atr_value * 2.5

        return TradingSignal(
            signal_type=signal_type,
            price=price,
            strength=strength,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss) if stop_loss else 0,
            reason=reason,
            indicators={r.name: r.values for r in results},
        )

    def _breakout_signal(
        self,
        results: List[IndicatorResult],
        price: float,
        position: int,
    ) -> TradingSignal:
        """突破策略信号"""
        squeeze = self._get_indicator_result(results, "squeeze")
        bollinger = self._get_indicator_result(results, "bollinger")
        choppiness = self._get_indicator_result(results, "choppiness")
        atr = self._get_indicator_result(results, "atr")

        signal_type = SignalType.NEUTRAL
        strength = 0.0
        confidence = 0.0
        reason = ""
        stop_loss = None
        take_profit = None

        if squeeze and bollinger:
            squeeze_on = squeeze.values.get("squeeze_on", True)
            momentum = squeeze.values.get("momentum", 0)
            upper = bollinger.values.get("upper", 0)
            lower = bollinger.values.get("lower", 0)
            bandwidth = bollinger.values.get("bandwidth", 100)

            # Squeeze 释放 = 突破信号
            # 布林带收窄 (bandwidth < 10%) 后释放更有效
            if not squeeze_on and bandwidth < 15:
                if momentum > 0 and price > upper * 0.99:
                    signal_type = SignalType.LONG
                    strength = 0.85
                    confidence = 0.7
                    reason = "Squeeze 释放 + 布林带收窄后向上突破"
                    stop_loss = lower

                elif momentum < 0 and price < lower * 1.01:
                    signal_type = SignalType.SHORT
                    strength = 0.85
                    confidence = 0.7
                    reason = "Squeeze 释放 + 布林带收窄后向下突破"
                    stop_loss = upper

        # 止盈
        if atr and stop_loss:
            atr_value = atr.values.get("atr", 0)
            if signal_type == SignalType.LONG:
                take_profit = price + atr_value * 3
            elif signal_type == SignalType.SHORT:
                take_profit = price - atr_value * 3

        return TradingSignal(
            signal_type=signal_type,
            price=price,
            strength=strength,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(price, stop_loss) if stop_loss else 0,
            reason=reason,
            indicators={r.name: r.values for r in results},
        )

    def _create_signal_from_combined(
        self,
        combined: Dict[str, Any],
        price: float,
        results: List[IndicatorResult],
    ) -> TradingSignal:
        """从综合信号创建交易信号"""
        signal = combined.get("signal", 0)
        strength = combined.get("strength", 0)
        confidence = combined.get("confidence", 0)

        if signal == 1:
            signal_type = SignalType.LONG
        elif signal == -1:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.NEUTRAL

        return TradingSignal(
            signal_type=signal_type,
            price=price,
            strength=strength,
            confidence=confidence,
            reason="综合指标信号",
            indicators=combined.get("details", {}),
        )

    def _calculate_position_size(self, price: float, stop_loss: Optional[float]) -> float:
        """
        计算仓位大小 (基于风险比例)

        使用固定风险模型: position_size = risk / (|price - stop_loss| / price)
        """
        if stop_loss is None or stop_loss == price:
            return 0.0

        risk_percent = abs(price - stop_loss) / price
        if risk_percent == 0:
            return 0.0

        # 仓位 = 风险比例 / 止损距离比例
        position_size = self.risk_per_trade / risk_percent

        # 限制最大仓位
        return min(position_size, 1.0)
