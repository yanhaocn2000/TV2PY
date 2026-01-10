"""
ICT Trading Sessions and Kill Zones

交易时段和杀戮区模块

Kill Zones (杀戮区) 是 ICT 方法论中的重要概念:
    - 市场在特定时间段内波动性最大
    - 机构订单在这些时段内最活跃
    - 是进场的最佳时机

主要时段:
    - Asian Session: 亚洲盘 (00:00 - 09:00 UTC)
    - London Session: 伦敦盘 (07:00 - 16:00 UTC)
    - New York Session: 纽约盘 (13:00 - 22:00 UTC)

杀戮区:
    - Asian Kill Zone: 00:00 - 04:00 UTC
    - London Open Kill Zone: 06:00 - 09:00 UTC
    - New York Kill Zone: 11:00 - 14:00 UTC
    - London Close Kill Zone: 14:00 - 16:00 UTC
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time
import pandas as pd
import numpy as np


@dataclass
class SessionConfig:
    """交易时段配置"""
    name: str
    start_time: time
    end_time: time
    timezone: str = "UTC"
    description: str = ""

    def is_active(self, current_time: time) -> bool:
        """检查当前时间是否在时段内"""
        if self.start_time < self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:
            # 跨午夜的时段
            return current_time >= self.start_time or current_time <= self.end_time


class TradingSessions:
    """
    交易时段管理器

    使用示例:
        sessions = TradingSessions()

        # 检查当前是否在伦敦时段
        is_london = sessions.is_in_session("London", current_time)

        # 获取时段内的K线
        london_candles = sessions.filter_session(df, "London")

        # 获取时段高低点
        session_hl = sessions.get_session_high_low(df, "London")
    """

    # 预定义时段
    DEFAULT_SESSIONS: Dict[str, SessionConfig] = {
        "Sydney": SessionConfig(
            name="Sydney",
            start_time=time(21, 0),
            end_time=time(6, 0),
            description="悉尼时段 (21:00 - 06:00 UTC)"
        ),
        "Tokyo": SessionConfig(
            name="Tokyo",
            start_time=time(0, 0),
            end_time=time(9, 0),
            description="东京时段 (00:00 - 09:00 UTC)"
        ),
        "London": SessionConfig(
            name="London",
            start_time=time(7, 0),
            end_time=time(16, 0),
            description="伦敦时段 (07:00 - 16:00 UTC)"
        ),
        "New York": SessionConfig(
            name="New York",
            start_time=time(13, 0),
            end_time=time(22, 0),
            description="纽约时段 (13:00 - 22:00 UTC)"
        ),
    }

    def __init__(self, custom_sessions: Optional[Dict[str, SessionConfig]] = None):
        """
        参数:
            custom_sessions: 自定义时段配置
        """
        self.sessions = self.DEFAULT_SESSIONS.copy()
        if custom_sessions:
            self.sessions.update(custom_sessions)

    def add_session(self, config: SessionConfig):
        """添加自定义时段"""
        self.sessions[config.name] = config

    def is_in_session(self, session_name: str, current_time: time) -> bool:
        """
        检查是否在指定时段内

        参数:
            session_name: 时段名称
            current_time: 当前时间

        返回:
            bool: 是否在时段内
        """
        if session_name not in self.sessions:
            raise ValueError(f"Unknown session: {session_name}")
        return self.sessions[session_name].is_active(current_time)

    def get_active_sessions(self, current_time: time) -> List[str]:
        """获取当前活跃的所有时段"""
        active = []
        for name, config in self.sessions.items():
            if config.is_active(current_time):
                active.append(name)
        return active

    def filter_session(
        self,
        ohlc: pd.DataFrame,
        session_name: str
    ) -> pd.DataFrame:
        """
        过滤出指定时段内的K线

        参数:
            ohlc: OHLCV DataFrame (索引必须是 datetime)
            session_name: 时段名称

        返回:
            DataFrame: 时段内的K线
        """
        if session_name not in self.sessions:
            raise ValueError(f"Unknown session: {session_name}")

        config = self.sessions[session_name]
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)

        mask = ohlc.index.map(
            lambda x: config.is_active(x.time())
        )

        return ohlc[mask]

    def get_session_high_low(
        self,
        ohlc: pd.DataFrame,
        session_name: str
    ) -> pd.DataFrame:
        """
        获取每个时段的高低点

        返回:
            DataFrame:
                - SessionHigh: 时段最高价
                - SessionLow: 时段最低价
                - SessionOpen: 时段开盘价
                - SessionClose: 时段收盘价
        """
        session_df = self.filter_session(ohlc, session_name)

        if len(session_df) == 0:
            return pd.DataFrame()

        # 按日期分组
        session_df = session_df.copy()
        session_df["date"] = session_df.index.date

        result = session_df.groupby("date").agg({
            "high": "max",
            "low": "min",
            "open": "first",
            "close": "last",
        }).rename(columns={
            "high": "SessionHigh",
            "low": "SessionLow",
            "open": "SessionOpen",
            "close": "SessionClose",
        })

        return result

    def mark_sessions(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        标记每根K线所属的时段

        返回:
            DataFrame: 原数据加上时段标记列
        """
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)

        for session_name in self.sessions:
            config = self.sessions[session_name]
            ohlc[f"In{session_name}"] = ohlc.index.map(
                lambda x: 1 if config.is_active(x.time()) else 0
            )

        return ohlc


class KillZones:
    """
    ICT 杀戮区 (Kill Zones)

    杀戮区是 ICT 方法论中高概率交易时段。

    使用示例:
        kz = KillZones()

        # 检查是否在杀戮区
        is_kz = kz.is_in_kill_zone(current_time)

        # 获取杀戮区内的K线
        kz_candles = kz.filter_kill_zone(df, "London Open")

        # 分析杀戮区
        analysis = kz.analyze_kill_zones(df)
    """

    # 杀戮区定义
    KILL_ZONES: Dict[str, SessionConfig] = {
        "Asian": SessionConfig(
            name="Asian Kill Zone",
            start_time=time(0, 0),
            end_time=time(4, 0),
            description="亚洲杀戮区 (00:00 - 04:00 UTC)"
        ),
        "London Open": SessionConfig(
            name="London Open Kill Zone",
            start_time=time(6, 0),
            end_time=time(9, 0),
            description="伦敦开盘杀戮区 (06:00 - 09:00 UTC)"
        ),
        "New York": SessionConfig(
            name="New York Kill Zone",
            start_time=time(11, 0),
            end_time=time(14, 0),
            description="纽约杀戮区 (11:00 - 14:00 UTC)"
        ),
        "London Close": SessionConfig(
            name="London Close Kill Zone",
            start_time=time(14, 0),
            end_time=time(16, 0),
            description="伦敦收盘杀戮区 (14:00 - 16:00 UTC)"
        ),
    }

    def __init__(self):
        self.kill_zones = self.KILL_ZONES.copy()

    def is_in_kill_zone(self, current_time: time) -> Tuple[bool, Optional[str]]:
        """
        检查是否在任何杀戮区内

        返回:
            Tuple[bool, Optional[str]]: (是否在杀戮区, 杀戮区名称)
        """
        for name, config in self.kill_zones.items():
            if config.is_active(current_time):
                return True, name
        return False, None

    def get_active_kill_zone(self, current_time: time) -> Optional[str]:
        """获取当前活跃的杀戮区"""
        is_active, name = self.is_in_kill_zone(current_time)
        return name if is_active else None

    def filter_kill_zone(
        self,
        ohlc: pd.DataFrame,
        kill_zone_name: str
    ) -> pd.DataFrame:
        """过滤出指定杀戮区的K线"""
        if kill_zone_name not in self.kill_zones:
            raise ValueError(f"Unknown kill zone: {kill_zone_name}")

        config = self.kill_zones[kill_zone_name]
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)

        mask = ohlc.index.map(lambda x: config.is_active(x.time()))
        return ohlc[mask]

    def mark_kill_zones(self, ohlc: pd.DataFrame) -> pd.DataFrame:
        """标记每根K线是否在杀戮区内"""
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)

        # 添加通用杀戮区标记
        def get_kz(dt):
            _, name = self.is_in_kill_zone(dt.time())
            return name if name else ""

        ohlc["KillZone"] = ohlc.index.map(get_kz)
        ohlc["InKillZone"] = ohlc["KillZone"].apply(lambda x: 1 if x else 0)

        return ohlc

    def analyze_kill_zones(self, ohlc: pd.DataFrame) -> Dict[str, Dict]:
        """
        分析各杀戮区的表现

        返回每个杀戮区的统计数据:
            - avg_range: 平均波动范围
            - avg_volume: 平均成交量
            - candle_count: K线数量
            - bullish_ratio: 阳线比例
        """
        ohlc = ohlc.copy()
        ohlc.columns = [c.lower() for c in ohlc.columns]
        ohlc.index = pd.to_datetime(ohlc.index)

        analysis = {}

        for name, config in self.kill_zones.items():
            kz_df = self.filter_kill_zone(ohlc, name)

            if len(kz_df) == 0:
                analysis[name] = {
                    "avg_range": 0,
                    "avg_volume": 0,
                    "candle_count": 0,
                    "bullish_ratio": 0,
                }
                continue

            # 计算统计数据
            ranges = kz_df["high"] - kz_df["low"]
            bullish = (kz_df["close"] > kz_df["open"]).sum()

            analysis[name] = {
                "avg_range": float(ranges.mean()),
                "avg_volume": float(kz_df["volume"].mean()) if "volume" in kz_df.columns else 0,
                "candle_count": len(kz_df),
                "bullish_ratio": float(bullish / len(kz_df)),
                "max_range": float(ranges.max()),
                "description": config.description,
            }

        return analysis

    def get_optimal_entry_times(self) -> List[Dict]:
        """
        获取最佳入场时间建议

        基于 ICT 方法论的最佳入场时间。
        """
        return [
            {
                "time": "02:00 - 04:00 UTC",
                "kill_zone": "Asian",
                "description": "亚洲盘后段，适合捕捉亚洲区间突破",
                "best_for": ["突破交易", "区间交易"],
            },
            {
                "time": "07:00 - 09:00 UTC",
                "kill_zone": "London Open",
                "description": "伦敦开盘，波动性最大，流动性充足",
                "best_for": ["趋势交易", "动量交易"],
            },
            {
                "time": "12:00 - 14:00 UTC",
                "kill_zone": "New York",
                "description": "纽约开盘，与伦敦重叠，高波动",
                "best_for": ["趋势交易", "反转交易"],
            },
            {
                "time": "14:00 - 16:00 UTC",
                "kill_zone": "London Close",
                "description": "伦敦收盘，机构平仓，可能出现反转",
                "best_for": ["反转交易", "获利了结"],
            },
        ]


@dataclass
class DailyBias:
    """每日偏向分析"""
    date: datetime
    asian_high: float
    asian_low: float
    london_direction: str  # "bullish" / "bearish" / "neutral"
    ny_direction: str
    overall_bias: str
    confidence: float


class ICTDailyAnalysis:
    """
    ICT 每日分析

    结合时段分析和市场结构判断每日偏向。
    """

    def __init__(self):
        self.sessions = TradingSessions()
        self.kill_zones = KillZones()

    def analyze_daily_bias(self, ohlc: pd.DataFrame) -> List[DailyBias]:
        """
        分析每日市场偏向

        基于:
        1. 亚洲盘区间
        2. 伦敦盘方向
        3. 纽约盘确认
        """
        ohlc = ohlc.copy()
        ohlc.columns = [c.lower() for c in ohlc.columns]
        ohlc.index = pd.to_datetime(ohlc.index)
        ohlc["date"] = ohlc.index.date

        results = []

        for date in ohlc["date"].unique():
            day_data = ohlc[ohlc["date"] == date]

            # 获取亚洲盘高低点
            asian_df = self.kill_zones.filter_kill_zone(day_data, "Asian")
            if len(asian_df) > 0:
                asian_high = asian_df["high"].max()
                asian_low = asian_df["low"].min()
            else:
                asian_high = asian_low = 0

            # 分析伦敦盘方向
            london_df = self.kill_zones.filter_kill_zone(day_data, "London Open")
            london_direction = self._get_direction(london_df, asian_high, asian_low)

            # 分析纽约盘方向
            ny_df = self.kill_zones.filter_kill_zone(day_data, "New York")
            ny_direction = self._get_direction(ny_df, asian_high, asian_low)

            # 综合判断
            overall_bias, confidence = self._get_overall_bias(
                london_direction, ny_direction
            )

            results.append(DailyBias(
                date=date,
                asian_high=asian_high,
                asian_low=asian_low,
                london_direction=london_direction,
                ny_direction=ny_direction,
                overall_bias=overall_bias,
                confidence=confidence,
            ))

        return results

    def _get_direction(
        self,
        df: pd.DataFrame,
        asian_high: float,
        asian_low: float
    ) -> str:
        """判断时段方向"""
        if len(df) == 0:
            return "neutral"

        session_close = df["close"].iloc[-1]

        if session_close > asian_high:
            return "bullish"
        elif session_close < asian_low:
            return "bearish"
        else:
            return "neutral"

    def _get_overall_bias(
        self,
        london_dir: str,
        ny_dir: str
    ) -> Tuple[str, float]:
        """获取整体偏向"""
        if london_dir == ny_dir:
            if london_dir == "bullish":
                return "bullish", 0.9
            elif london_dir == "bearish":
                return "bearish", 0.9
            else:
                return "neutral", 0.5
        elif london_dir == "neutral":
            return ny_dir, 0.6
        elif ny_dir == "neutral":
            return london_dir, 0.6
        else:
            # 方向冲突
            return "neutral", 0.3
