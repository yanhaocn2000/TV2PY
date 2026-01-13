"""
ICT Signal Generator

ICT 方法论信号生成器

基于 ICT 概念生成交易信号:
    - Order Block 回测入场
    - FVG 入场
    - BOS/CHoCH 确认
    - Kill Zone 过滤
    - Liquidity 目标
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from .smc import smc, ICTAnalyzer, ICTAnalysisResult
from .sessions import KillZones, TradingSessions


class SignalType(Enum):
    """信号类型"""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


class EntryType(Enum):
    """入场类型"""
    ORDER_BLOCK = "order_block"
    FVG = "fvg"
    BOS = "bos"
    CHOCH = "choch"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    OPTIMAL_TRADE_ENTRY = "ote"  # 0.618-0.786 回撤


@dataclass
class ICTSignal:
    """ICT 交易信号"""
    signal_type: SignalType
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: Any
    kill_zone: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        """计算风险收益比"""
        if self.signal_type == SignalType.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit

        return reward / risk if risk > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal_type.value,
            "entry_type": self.entry_type.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": self.risk_reward_ratio,
            "confidence": self.confidence,
            "kill_zone": self.kill_zone,
            "reason": self.reason,
        }


class ICTSignalGenerator:
    """
    ICT 信号生成器

    使用示例:
        generator = ICTSignalGenerator(
            min_rr=2.0,
            use_kill_zones=True,
        )

        signals = generator.generate_signals(df)

        for signal in signals:
            print(f"{signal.signal_type.value}: Entry={signal.entry_price}, "
                  f"SL={signal.stop_loss}, TP={signal.take_profit}, "
                  f"R:R={signal.risk_reward_ratio:.2f}")
    """

    def __init__(
        self,
        swing_length: int = 50,
        min_rr: float = 2.0,
        use_kill_zones: bool = True,
        ob_retest: bool = True,
        fvg_entry: bool = True,
        min_confidence: float = 0.6,
    ):
        """
        参数:
            swing_length: 摆动高低点的查找范围
            min_rr: 最小风险收益比
            use_kill_zones: 是否只在杀戮区内生成信号
            ob_retest: 是否使用订单块回测入场
            fvg_entry: 是否使用 FVG 入场
            min_confidence: 最小置信度
        """
        self.analyzer = ICTAnalyzer(swing_length=swing_length)
        self.kill_zones = KillZones()
        self.min_rr = min_rr
        self.use_kill_zones = use_kill_zones
        self.ob_retest = ob_retest
        self.fvg_entry = fvg_entry
        self.min_confidence = min_confidence

    def generate_signals(
        self,
        ohlc: pd.DataFrame,
        lookback: int = 50,
    ) -> List[ICTSignal]:
        """
        生成 ICT 交易信号

        参数:
            ohlc: OHLCV DataFrame
            lookback: 回看K线数量

        返回:
            List[ICTSignal]: 信号列表
        """
        signals = []

        # 执行 ICT 分析
        analysis = self.analyzer.analyze(ohlc)

        # 获取市场偏向
        bias = self.analyzer.get_bias(ohlc)

        # 遍历最近的K线寻找入场机会
        recent_range = range(max(0, len(ohlc) - lookback), len(ohlc))

        for i in recent_range:
            current_bar = ohlc.iloc[i]
            current_time = ohlc.index[i]

            # 检查是否在杀戮区
            kill_zone = None
            if self.use_kill_zones:
                if hasattr(current_time, 'time'):
                    is_kz, kz_name = self.kill_zones.is_in_kill_zone(current_time.time())
                    if not is_kz:
                        continue
                    kill_zone = kz_name

            # 订单块入场信号
            if self.ob_retest:
                ob_signal = self._check_order_block_entry(
                    ohlc, analysis, i, bias, kill_zone
                )
                if ob_signal:
                    signals.append(ob_signal)

            # FVG 入场信号
            if self.fvg_entry:
                fvg_signal = self._check_fvg_entry(
                    ohlc, analysis, i, bias, kill_zone
                )
                if fvg_signal:
                    signals.append(fvg_signal)

            # BOS/CHoCH 信号
            structure_signal = self._check_structure_signal(
                ohlc, analysis, i, bias, kill_zone
            )
            if structure_signal:
                signals.append(structure_signal)

        return signals

    def _check_order_block_entry(
        self,
        ohlc: pd.DataFrame,
        analysis: ICTAnalysisResult,
        bar_index: int,
        bias: Dict,
        kill_zone: Optional[str],
    ) -> Optional[ICTSignal]:
        """检查订单块回测入场"""
        ob = analysis.order_blocks
        current_bar = ohlc.iloc[bar_index]
        current_low = current_bar["low"]
        current_high = current_bar["high"]
        current_close = current_bar["close"]

        # 查找活跃的订单块
        for i in range(bar_index):
            if pd.isna(ob["OB"].iloc[i]):
                continue

            ob_type = ob["OB"].iloc[i]
            ob_top = ob["Top"].iloc[i]
            ob_bottom = ob["Bottom"].iloc[i]
            mitigated = ob["MitigatedIndex"].iloc[i]

            # 跳过已消除的订单块
            if mitigated > 0 and mitigated <= bar_index:
                continue

            # 看涨订单块 - 价格回测到订单块区域
            if ob_type == 1 and bias["bias"] in ["bullish", "neutral"]:
                if current_low <= ob_top and current_close > ob_bottom:
                    # 入场在订单块顶部
                    entry = ob_top
                    stop_loss = ob_bottom - (ob_top - ob_bottom) * 0.2
                    # 寻找目标 - 使用流动性或前高
                    take_profit = self._find_target(ohlc, analysis, bar_index, "long")

                    if self._check_rr(entry, stop_loss, take_profit, "long"):
                        return ICTSignal(
                            signal_type=SignalType.LONG,
                            entry_type=EntryType.ORDER_BLOCK,
                            entry_price=entry,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=0.7 + (0.2 if kill_zone else 0),
                            timestamp=ohlc.index[bar_index],
                            kill_zone=kill_zone,
                            reason=f"Bullish OB retest at {ob_top:.2f}",
                            metadata={
                                "ob_index": i,
                                "bias": bias["bias"],
                            }
                        )

            # 看跌订单块
            elif ob_type == -1 and bias["bias"] in ["bearish", "neutral"]:
                if current_high >= ob_bottom and current_close < ob_top:
                    entry = ob_bottom
                    stop_loss = ob_top + (ob_top - ob_bottom) * 0.2
                    take_profit = self._find_target(ohlc, analysis, bar_index, "short")

                    if self._check_rr(entry, stop_loss, take_profit, "short"):
                        return ICTSignal(
                            signal_type=SignalType.SHORT,
                            entry_type=EntryType.ORDER_BLOCK,
                            entry_price=entry,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=0.7 + (0.2 if kill_zone else 0),
                            timestamp=ohlc.index[bar_index],
                            kill_zone=kill_zone,
                            reason=f"Bearish OB retest at {ob_bottom:.2f}",
                            metadata={
                                "ob_index": i,
                                "bias": bias["bias"],
                            }
                        )

        return None

    def _check_fvg_entry(
        self,
        ohlc: pd.DataFrame,
        analysis: ICTAnalysisResult,
        bar_index: int,
        bias: Dict,
        kill_zone: Optional[str],
    ) -> Optional[ICTSignal]:
        """检查 FVG 入场"""
        fvg = analysis.fvg
        current_bar = ohlc.iloc[bar_index]
        current_low = current_bar["low"]
        current_high = current_bar["high"]

        # 查找未填补的 FVG
        for i in range(max(0, bar_index - 20), bar_index):
            if pd.isna(fvg["FVG"].iloc[i]):
                continue

            fvg_type = fvg["FVG"].iloc[i]
            fvg_top = fvg["Top"].iloc[i]
            fvg_bottom = fvg["Bottom"].iloc[i]
            mitigated = fvg["MitigatedIndex"].iloc[i]

            if mitigated > 0 and mitigated <= bar_index:
                continue

            # 看涨 FVG
            if fvg_type == 1 and bias["bias"] in ["bullish", "neutral"]:
                if current_low <= fvg_top and current_low >= fvg_bottom:
                    entry = (fvg_top + fvg_bottom) / 2
                    stop_loss = fvg_bottom - (fvg_top - fvg_bottom)
                    take_profit = self._find_target(ohlc, analysis, bar_index, "long")

                    if self._check_rr(entry, stop_loss, take_profit, "long"):
                        return ICTSignal(
                            signal_type=SignalType.LONG,
                            entry_type=EntryType.FVG,
                            entry_price=entry,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=0.65 + (0.2 if kill_zone else 0),
                            timestamp=ohlc.index[bar_index],
                            kill_zone=kill_zone,
                            reason=f"Bullish FVG entry at {entry:.2f}",
                        )

            # 看跌 FVG
            elif fvg_type == -1 and bias["bias"] in ["bearish", "neutral"]:
                if current_high >= fvg_bottom and current_high <= fvg_top:
                    entry = (fvg_top + fvg_bottom) / 2
                    stop_loss = fvg_top + (fvg_top - fvg_bottom)
                    take_profit = self._find_target(ohlc, analysis, bar_index, "short")

                    if self._check_rr(entry, stop_loss, take_profit, "short"):
                        return ICTSignal(
                            signal_type=SignalType.SHORT,
                            entry_type=EntryType.FVG,
                            entry_price=entry,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            confidence=0.65 + (0.2 if kill_zone else 0),
                            timestamp=ohlc.index[bar_index],
                            kill_zone=kill_zone,
                            reason=f"Bearish FVG entry at {entry:.2f}",
                        )

        return None

    def _check_structure_signal(
        self,
        ohlc: pd.DataFrame,
        analysis: ICTAnalysisResult,
        bar_index: int,
        bias: Dict,
        kill_zone: Optional[str],
    ) -> Optional[ICTSignal]:
        """检查 BOS/CHoCH 结构信号"""
        bos_choch = analysis.bos_choch

        # 检查当前K线是否有 CHoCH (更强的信号)
        if not pd.isna(bos_choch["CHOCH"].iloc[bar_index]):
            choch_type = bos_choch["CHOCH"].iloc[bar_index]
            level = bos_choch["Level"].iloc[bar_index]
            current_close = ohlc["close"].iloc[bar_index]

            if choch_type == 1:  # 看涨 CHoCH
                entry = current_close
                stop_loss = ohlc["low"].iloc[max(0, bar_index - 5):bar_index + 1].min()
                take_profit = self._find_target(ohlc, analysis, bar_index, "long")

                if self._check_rr(entry, stop_loss, take_profit, "long"):
                    return ICTSignal(
                        signal_type=SignalType.LONG,
                        entry_type=EntryType.CHOCH,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence=0.8 + (0.15 if kill_zone else 0),
                        timestamp=ohlc.index[bar_index],
                        kill_zone=kill_zone,
                        reason=f"Bullish CHoCH at {level:.2f}",
                    )

            elif choch_type == -1:  # 看跌 CHoCH
                entry = current_close
                stop_loss = ohlc["high"].iloc[max(0, bar_index - 5):bar_index + 1].max()
                take_profit = self._find_target(ohlc, analysis, bar_index, "short")

                if self._check_rr(entry, stop_loss, take_profit, "short"):
                    return ICTSignal(
                        signal_type=SignalType.SHORT,
                        entry_type=EntryType.CHOCH,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence=0.8 + (0.15 if kill_zone else 0),
                        timestamp=ohlc.index[bar_index],
                        kill_zone=kill_zone,
                        reason=f"Bearish CHoCH at {level:.2f}",
                    )

        return None

    def _find_target(
        self,
        ohlc: pd.DataFrame,
        analysis: ICTAnalysisResult,
        bar_index: int,
        direction: str,
    ) -> float:
        """
        寻找目标价位

        优先级:
        1. 流动性区域
        2. 前高/前低
        3. 固定倍数止盈
        """
        liquidity = analysis.liquidity
        swing_hl = analysis.swing_highs_lows
        current_price = ohlc["close"].iloc[bar_index]

        if direction == "long":
            # 寻找上方流动性
            for i in range(bar_index, len(liquidity)):
                if liquidity["Liquidity"].iloc[i] == 1:
                    return liquidity["Level"].iloc[i]

            # 寻找前高
            highs = swing_hl[swing_hl["HighLow"] == 1]["Level"]
            future_highs = highs[highs.index > bar_index]
            if len(future_highs) > 0:
                return future_highs.iloc[0]

            # 默认: 当前价格 + 2倍风险
            return current_price * 1.03

        else:  # short
            # 寻找下方流动性
            for i in range(bar_index, len(liquidity)):
                if liquidity["Liquidity"].iloc[i] == -1:
                    return liquidity["Level"].iloc[i]

            # 寻找前低
            lows = swing_hl[swing_hl["HighLow"] == -1]["Level"]
            future_lows = lows[lows.index > bar_index]
            if len(future_lows) > 0:
                return future_lows.iloc[0]

            return current_price * 0.97

    def _check_rr(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float,
        direction: str,
    ) -> bool:
        """检查风险收益比是否满足要求"""
        if direction == "long":
            risk = entry - stop_loss
            reward = take_profit - entry
        else:
            risk = stop_loss - entry
            reward = entry - take_profit

        if risk <= 0:
            return False

        rr = reward / risk
        return rr >= self.min_rr

    def get_current_signal(self, ohlc: pd.DataFrame) -> Optional[ICTSignal]:
        """获取当前K线的信号（如果有）"""
        signals = self.generate_signals(ohlc, lookback=1)
        return signals[-1] if signals else None

    def backtest_signals(
        self,
        ohlc: pd.DataFrame,
        initial_capital: float = 10000,
        risk_per_trade: float = 0.02,
    ) -> Dict[str, Any]:
        """
        回测信号表现

        参数:
            ohlc: OHLCV 数据
            initial_capital: 初始资金
            risk_per_trade: 每笔风险比例

        返回:
            回测统计数据
        """
        signals = self.generate_signals(ohlc, lookback=len(ohlc))

        trades = []
        capital = initial_capital

        for signal in signals:
            # 计算仓位大小
            risk_amount = capital * risk_per_trade

            if signal.signal_type == SignalType.LONG:
                risk_per_unit = signal.entry_price - signal.stop_loss
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                # 模拟交易结果 (简化: 假设达到目标)
                pnl = (signal.take_profit - signal.entry_price) * position_size * 0.6  # 60% 胜率
                pnl -= (signal.entry_price - signal.stop_loss) * position_size * 0.4

            else:
                risk_per_unit = signal.stop_loss - signal.entry_price
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0

                pnl = (signal.entry_price - signal.take_profit) * position_size * 0.6
                pnl -= (signal.stop_loss - signal.entry_price) * position_size * 0.4

            capital += pnl
            trades.append({
                "signal": signal.signal_type.value,
                "entry_type": signal.entry_type.value,
                "pnl": pnl,
                "capital": capital,
                "rr": signal.risk_reward_ratio,
            })

        # 计算统计数据
        total_trades = len(trades)
        if total_trades == 0:
            return {"error": "No trades generated"}

        wins = sum(1 for t in trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in trades)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": wins / total_trades,
            "total_pnl": total_pnl,
            "total_return": (capital - initial_capital) / initial_capital,
            "final_capital": capital,
            "avg_rr": np.mean([t["rr"] for t in trades]),
            "trades": trades,
        }
