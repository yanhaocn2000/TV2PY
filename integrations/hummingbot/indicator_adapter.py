"""
TV2PY Indicator Adapter for Hummingbot

将 TV2PY 指标适配为 Hummingbot 可用的格式

Hummingbot 使用 Candles 数据格式，本适配器提供:
1. 数据格式转换
2. 指标计算封装
3. 实时更新支持
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum

import numpy as np
import pandas as pd

# 添加 TV2PY 路径
TV2PY_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TV2PY_PATH))

# 导入 TV2PY 指标
try:
    from strategies.smart_money_concepts import SmartMoneyConcepts, SMCResult
    from strategies.squeeze_momentum import SqueezeMomentumIndicator, SqueezeMomentumParams
    from strategies.supertrend import SuperTrendIndicator
    from strategies.volume_profile import VolumeProfile
    from strategies.divergence import RSIDivergence
    from strategies.swing_detection import MarketStructure
    from strategies.choppiness_index import ChoppinessIndex
    from strategies.bollinger_bands import BollingerBands
    from strategies.atr import ATRIndicator
    HAS_TV2PY = True
except ImportError as e:
    print(f"Warning: Could not import TV2PY indicators: {e}")
    print("Make sure TV2PY is in your PYTHONPATH")
    HAS_TV2PY = False


@dataclass
class CandleData:
    """
    K线数据结构 (兼容 Hummingbot Candles)

    Hummingbot 的 Candles 数据格式:
    - timestamp: Unix 时间戳
    - open, high, low, close: OHLC 价格
    - volume: 成交量
    """
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CandleData":
        """从 DataFrame 创建"""
        return cls(
            timestamp=df.index.astype(np.int64) // 10**9 if hasattr(df.index, 'astype') else np.arange(len(df)),
            open=df['open'].values if 'open' in df.columns else df['Open'].values,
            high=df['high'].values if 'high' in df.columns else df['High'].values,
            low=df['low'].values if 'low' in df.columns else df['Low'].values,
            close=df['close'].values if 'close' in df.columns else df['Close'].values,
            volume=df['volume'].values if 'volume' in df.columns else df['Volume'].values,
        )

    @classmethod
    def from_hummingbot_candles(cls, candles: List[List]) -> "CandleData":
        """
        从 Hummingbot Candles 格式创建

        Hummingbot candles 格式: [[timestamp, open, high, low, close, volume], ...]
        """
        arr = np.array(candles)
        return cls(
            timestamp=arr[:, 0],
            open=arr[:, 1],
            high=arr[:, 2],
            low=arr[:, 3],
            close=arr[:, 4],
            volume=arr[:, 5],
        )

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }, index=pd.to_datetime(self.timestamp, unit='s'))

    def __len__(self) -> int:
        return len(self.close)


@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str
    values: Dict[str, Any]
    signal: Optional[int] = None  # 1=买入, -1=卖出, 0=中性
    strength: float = 0.0  # 信号强度 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndicatorAdapter:
    """
    TV2PY 指标适配器

    将 TV2PY 指标封装为 Hummingbot 可用的格式

    使用示例:
        adapter = IndicatorAdapter()
        adapter.add_indicator("smc", swing_length=10)
        adapter.add_indicator("squeeze")
        adapter.add_indicator("supertrend", atr_period=10, multiplier=3)

        results = adapter.calculate(candle_data)
        for result in results:
            print(f"{result.name}: signal={result.signal}")
    """

    def __init__(self):
        self.indicators: Dict[str, Any] = {}
        self.indicator_configs: Dict[str, Dict] = {}

    def add_indicator(self, name: str, **kwargs) -> "IndicatorAdapter":
        """
        添加指标

        支持的指标:
            - smc: Smart Money Concepts
            - squeeze: Squeeze Momentum
            - supertrend: SuperTrend
            - volume_profile: Volume Profile
            - divergence: RSI Divergence
            - market_structure: Market Structure
            - choppiness: Choppiness Index
            - bollinger: Bollinger Bands
            - atr: Average True Range
        """
        if not HAS_TV2PY:
            raise ImportError("TV2PY indicators not available")

        indicator_map = {
            "smc": lambda **kw: SmartMoneyConcepts(
                swing_length=kw.get("swing_length", 10),
                ob_lookback=kw.get("ob_lookback", 50),
            ),
            "squeeze": lambda **kw: SqueezeMomentumIndicator(
                SqueezeMomentumParams(
                    bb_length=kw.get("bb_length", 20),
                    bb_mult=kw.get("bb_mult", 2.0),
                    kc_length=kw.get("kc_length", 20),
                    kc_mult=kw.get("kc_mult", 1.5),
                )
            ),
            "supertrend": lambda **kw: SuperTrendIndicator(
                period=kw.get("period", kw.get("atr_period", 10)),
                multiplier=kw.get("multiplier", 3.0),
            ),
            "volume_profile": lambda **kw: VolumeProfile(
                row_size=kw.get("row_size", 24),
                value_area_percent=kw.get("value_area_percent", 70.0),
            ),
            "divergence": lambda **kw: RSIDivergence(
                pivot_lookback=kw.get("pivot_lookback", 5),
            ),
            "market_structure": lambda **kw: MarketStructure(
                left_bars=kw.get("left_bars", 5),
                right_bars=kw.get("right_bars", 5),
            ),
            "choppiness": lambda **kw: ChoppinessIndex(
                period=kw.get("period", kw.get("length", 14)),
            ),
            "bollinger": lambda **kw: BollingerBands(
                period=kw.get("period", kw.get("length", 20)),
                mult=kw.get("mult", 2.0),
            ),
            "atr": lambda **kw: ATRIndicator(
                period=kw.get("period", kw.get("length", 14)),
            ),
        }

        if name not in indicator_map:
            raise ValueError(f"Unknown indicator: {name}. Available: {list(indicator_map.keys())}")

        self.indicators[name] = indicator_map[name](**kwargs)
        self.indicator_configs[name] = kwargs
        return self

    def calculate(self, candles: Union[CandleData, pd.DataFrame, List]) -> List[IndicatorResult]:
        """
        计算所有指标

        Args:
            candles: K线数据 (CandleData, DataFrame, 或 Hummingbot 格式)

        Returns:
            List[IndicatorResult]: 指标计算结果列表
        """
        # 转换数据格式
        if isinstance(candles, pd.DataFrame):
            data = CandleData.from_dataframe(candles)
        elif isinstance(candles, list):
            data = CandleData.from_hummingbot_candles(candles)
        else:
            data = candles

        results = []

        for name, indicator in self.indicators.items():
            try:
                result = self._calculate_indicator(name, indicator, data)
                results.append(result)
            except Exception as e:
                results.append(IndicatorResult(
                    name=name,
                    values={"error": str(e)},
                    signal=0,
                    strength=0.0,
                ))

        return results

    def _calculate_indicator(
        self,
        name: str,
        indicator: Any,
        data: CandleData
    ) -> IndicatorResult:
        """计算单个指标"""

        if name == "smc":
            result = indicator.calculate(
                data.open, data.high, data.low, data.close, data.volume
            )
            # 获取当前信号
            signals = indicator.get_signals(result, data.close, len(data) - 1)
            signal = 0
            strength = 0.0

            if signals['in_discount'] and signals['near_bullish_ob']:
                signal = 1
                strength = 0.8
            elif signals['in_premium'] and signals['near_bearish_ob']:
                signal = -1
                strength = 0.8
            elif signals['trend'] == 1:
                signal = 1
                strength = 0.5
            elif signals['trend'] == -1:
                signal = -1
                strength = 0.5

            return IndicatorResult(
                name="smc",
                values={
                    "trend": signals['trend'],
                    "in_premium": signals['in_premium'],
                    "in_discount": signals['in_discount'],
                    "equilibrium": result.equilibrium[-1] if len(result.equilibrium) > 0 else None,
                    "active_bullish_ob": len(result.active_bullish_ob),
                    "active_bearish_ob": len(result.active_bearish_ob),
                    "active_bullish_fvg": len(result.active_bullish_fvg),
                    "active_bearish_fvg": len(result.active_bearish_fvg),
                },
                signal=signal,
                strength=strength,
                metadata={"recent_choch": signals.get('recent_choch'), "recent_bos": signals.get('recent_bos')},
            )

        elif name == "squeeze":
            result = indicator.calculate(data.high, data.low, data.close)
            signal = 0
            strength = 0.0

            if result.squeeze_on[-1]:
                # Squeeze 激活中，等待释放
                signal = 0
                strength = 0.3
            elif result.momentum[-1] > 0 and result.momentum[-1] > result.momentum[-2]:
                signal = 1
                strength = min(abs(result.momentum[-1]) / 10, 1.0)
            elif result.momentum[-1] < 0 and result.momentum[-1] < result.momentum[-2]:
                signal = -1
                strength = min(abs(result.momentum[-1]) / 10, 1.0)

            return IndicatorResult(
                name="squeeze",
                values={
                    "squeeze_on": bool(result.squeeze_on[-1]),
                    "momentum": float(result.momentum[-1]),
                    "momentum_rising": bool(result.momentum[-1] > result.momentum[-2]) if len(result.momentum) > 1 else False,
                },
                signal=signal,
                strength=strength,
            )

        elif name == "supertrend":
            result = indicator.calculate(data.high, data.low, data.close)
            signal = int(result.direction[-1])
            strength = 0.7

            return IndicatorResult(
                name="supertrend",
                values={
                    "supertrend": float(result.supertrend[-1]),
                    "direction": int(result.direction[-1]),
                    "upper_band": float(result.upper_band[-1]),
                    "lower_band": float(result.lower_band[-1]),
                },
                signal=signal,
                strength=strength,
            )

        elif name == "volume_profile":
            result = indicator.calculate(data.high, data.low, data.close, data.volume)
            current_price = data.close[-1]
            signal = 0
            strength = 0.0

            # 价格在 POC 附近
            if abs(current_price - result.poc_price) / current_price < 0.01:
                strength = 0.3
            # 价格在折价区
            elif current_price < result.val:
                signal = 1
                strength = 0.6
            # 价格在溢价区
            elif current_price > result.vah:
                signal = -1
                strength = 0.6

            return IndicatorResult(
                name="volume_profile",
                values={
                    "poc": float(result.poc_price),
                    "vah": float(result.vah),
                    "val": float(result.val),
                    "hvn_count": len(result.hvn_prices),
                },
                signal=signal,
                strength=strength,
            )

        elif name == "choppiness":
            result = indicator.calculate(data.high, data.low, data.close)
            ci_value = result.choppiness[-1]
            signal = 0
            strength = 0.0

            # CI > 61.8 表示震荡市场
            if ci_value > 61.8:
                signal = 0  # 不适合趋势交易
                strength = 0.0
            # CI < 38.2 表示趋势市场
            elif ci_value < 38.2:
                signal = 0  # 信号需要配合其他指标
                strength = 0.8

            return IndicatorResult(
                name="choppiness",
                values={
                    "choppiness": float(ci_value),
                    "is_trending": ci_value < 38.2,
                    "is_ranging": ci_value > 61.8,
                },
                signal=signal,
                strength=strength,
            )

        elif name == "bollinger":
            result = indicator.calculate(data.close)
            current_price = data.close[-1]
            signal = 0
            strength = 0.0

            if current_price <= result.lower[-1]:
                signal = 1
                strength = 0.7
            elif current_price >= result.upper[-1]:
                signal = -1
                strength = 0.7

            return IndicatorResult(
                name="bollinger",
                values={
                    "upper": float(result.upper[-1]),
                    "middle": float(result.basis[-1]),
                    "lower": float(result.lower[-1]),
                    "bandwidth": float((result.upper[-1] - result.lower[-1]) / result.basis[-1] * 100),
                },
                signal=signal,
                strength=strength,
            )

        elif name == "atr":
            result = indicator.calculate(data.high, data.low, data.close)
            atr_value = result.atr[-1]

            return IndicatorResult(
                name="atr",
                values={
                    "atr": float(atr_value),
                    "atr_percent": float(atr_value / data.close[-1] * 100),
                },
                signal=0,  # ATR 不提供方向信号
                strength=0.0,
            )

        else:
            return IndicatorResult(
                name=name,
                values={"error": f"Calculation not implemented for {name}"},
                signal=0,
                strength=0.0,
            )

    def get_combined_signal(self, results: List[IndicatorResult]) -> Dict[str, Any]:
        """
        综合多个指标的信号

        Args:
            results: 指标计算结果列表

        Returns:
            Dict with:
                - signal: 综合信号 (1=买入, -1=卖出, 0=中性)
                - strength: 信号强度 (0-1)
                - confidence: 指标一致性 (0-1)
                - details: 各指标详情
        """
        if not results:
            return {"signal": 0, "strength": 0.0, "confidence": 0.0, "details": {}}

        signals = []
        weights = []

        for result in results:
            if result.signal != 0:
                signals.append(result.signal)
                weights.append(result.strength)

        if not signals:
            return {"signal": 0, "strength": 0.0, "confidence": 0.0, "details": {r.name: r.values for r in results}}

        # 加权平均信号
        weighted_signal = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
        final_signal = 1 if weighted_signal > 0.3 else (-1 if weighted_signal < -0.3 else 0)

        # 计算一致性
        same_direction = sum(1 for s in signals if s == final_signal)
        confidence = same_direction / len(signals) if signals else 0

        return {
            "signal": final_signal,
            "strength": sum(weights) / len(weights) if weights else 0,
            "confidence": confidence,
            "details": {r.name: r.values for r in results},
        }
