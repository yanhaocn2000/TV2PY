"""
ICT Smart Money Concepts Implementation

Based on smart-money-concepts by Josh Attridge
https://github.com/joshyattridge/smart-money-concepts
License: MIT

增强功能:
    - ICTAnalyzer 类提供统一接口
    - 中文文档和注释
    - 与 TV2PY 其他模块的集成
"""

from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


def inputvalidator(input_="ohlc"):
    """输入验证装饰器"""
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and not attr.startswith('_'):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


@apply(inputvalidator(input_="ohlc"))
class smc:
    """
    Smart Money Concepts (ICT 方法论)

    提供完整的 ICT 技术分析工具集。

    使用示例:
        from integrations.ict import smc

        # 检测摆动高低点
        swing_hl = smc.swing_highs_lows(df, swing_length=50)

        # 检测公允价值缺口
        fvg = smc.fvg(df)

        # 检测订单块
        ob = smc.ob(df, swing_hl)

        # 检测结构突破
        bos_choch = smc.bos_choch(df, swing_hl)
    """

    __version__ = "0.0.26"

    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive: bool = False) -> DataFrame:
        """
        FVG - Fair Value Gap (公允价值缺口)

        当前一根K线的高点低于下一根K线的低点（看涨）
        或前一根K线的低点高于下一根K线的高点（看跌）

        参数:
            ohlc: DataFrame - OHLCV 数据
            join_consecutive: bool - 是否合并连续的 FVG

        返回:
            DataFrame:
                - FVG: 1=看涨缺口, -1=看跌缺口
                - Top: 缺口顶部
                - Bottom: 缺口底部
                - MitigatedIndex: 被填补的K线索引
        """
        fvg = np.where(
            (
                (ohlc["high"].shift(1) < ohlc["low"].shift(-1))
                & (ohlc["close"] > ohlc["open"])
            )
            | (
                (ohlc["low"].shift(1) > ohlc["high"].shift(-1))
                & (ohlc["close"] < ohlc["open"])
            ),
            np.where(ohlc["close"] > ohlc["open"], 1, -1),
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["low"].shift(-1),
                ohlc["low"].shift(1),
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["high"].shift(1),
                ohlc["high"].shift(-1),
            ),
            np.nan,
        )

        if join_consecutive:
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    fvg[i] = top[i] = bottom[i] = np.nan

        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2:] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2:] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        return pd.concat(
            [
                pd.Series(fvg, name="FVG"),
                pd.Series(top, name="Top"),
                pd.Series(bottom, name="Bottom"),
                pd.Series(mitigated_index, name="MitigatedIndex"),
            ],
            axis=1,
        )

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> DataFrame:
        """
        Swing Highs and Lows (摆动高低点)

        在指定范围内识别价格的峰值和谷值。

        参数:
            ohlc: DataFrame - OHLCV 数据
            swing_length: int - 前后查找的K线数量

        返回:
            DataFrame:
                - HighLow: 1=摆动高点, -1=摆动低点
                - Level: 价格水平
        """
        swing_length *= 2
        swing_highs_lows = np.where(
            ohlc["high"]
            == ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max(),
            1,
            np.where(
                ohlc["low"]
                == ohlc["low"].shift(-(swing_length // 2)).rolling(swing_length).min(),
                -1,
                np.nan,
            ),
        )

        while True:
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            if len(positions) < 2:
                break

            current = swing_highs_lows[positions[:-1]]
            next_val = swing_highs_lows[positions[1:]]

            highs = ohlc["high"].iloc[positions[:-1]].values
            lows = ohlc["low"].iloc[positions[:-1]].values

            next_highs = ohlc["high"].iloc[positions[1:]].values
            next_lows = ohlc["low"].iloc[positions[1:]].values

            index_to_remove = np.zeros(len(positions), dtype=bool)

            consecutive_highs = (current == 1) & (next_val == 1)
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            consecutive_lows = (current == -1) & (next_val == -1)
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

            if not index_to_remove.any():
                break

            swing_highs_lows[positions[index_to_remove]] = np.nan

        positions = np.where(~np.isnan(swing_highs_lows))[0]

        if len(positions) > 0:
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        return pd.concat(
            [
                pd.Series(swing_highs_lows, name="HighLow"),
                pd.Series(level, name="Level"),
            ],
            axis=1,
        )

    @classmethod
    def bos_choch(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True
    ) -> DataFrame:
        """
        BOS - Break of Structure (结构突破)
        CHoCH - Change of Character (性质改变)

        参数:
            ohlc: DataFrame - OHLCV 数据
            swing_highs_lows: DataFrame - swing_highs_lows() 的输出
            close_break: bool - 是否以收盘价判断突破

        返回:
            DataFrame:
                - BOS: 1=看涨突破, -1=看跌突破
                - CHOCH: 1=看涨性质改变, -1=看跌性质改变
                - Level: 突破的价格水平
                - BrokenIndex: 突破发生的K线索引
        """
        swing_highs_lows = swing_highs_lows.copy()

        level_order = []
        highs_lows_order = []

        bos = np.zeros(len(ohlc), dtype=np.int32)
        choch = np.zeros(len(ohlc), dtype=np.int32)
        level = np.zeros(len(ohlc), dtype=np.float32)

        last_positions = []

        for i in range(len(swing_highs_lows["HighLow"])):
            if not np.isnan(swing_highs_lows["HighLow"][i]):
                level_order.append(swing_highs_lows["Level"][i])
                highs_lows_order.append(swing_highs_lows["HighLow"][i])
                if len(level_order) >= 4:
                    # bullish bos
                    bos[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-4]
                                < level_order[-2]
                                < level_order[-3]
                                < level_order[-1]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bearish bos
                    bos[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-4]
                                > level_order[-2]
                                > level_order[-3]
                                > level_order[-1]
                            )
                        )
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # bullish choch
                    choch[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-1]
                                > level_order[-3]
                                > level_order[-4]
                                > level_order[-2]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                    # bearish choch
                    choch[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-1]
                                < level_order[-3]
                                < level_order[-4]
                                > level_order[-2]
                            )
                        )
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                last_positions.append(i)

        broken = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if bos[i] == 1 or choch[i] == 1:
                mask = ohlc["close" if close_break else "high"][i + 2:] > level[i]
            elif bos[i] == -1 or choch[i] == -1:
                mask = ohlc["close" if close_break else "low"][i + 2:] < level[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0

        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        return pd.concat(
            [
                pd.Series(bos, name="BOS"),
                pd.Series(choch, name="CHOCH"),
                pd.Series(level, name="Level"),
                pd.Series(broken, name="BrokenIndex"),
            ],
            axis=1,
        )

    @classmethod
    def ob(
        cls,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        close_mitigation: bool = False,
    ) -> DataFrame:
        """
        OB - Order Blocks (订单块)

        检测机构订单集中的价格区域。

        参数:
            ohlc: DataFrame - OHLCV 数据
            swing_highs_lows: DataFrame - swing_highs_lows() 的输出
            close_mitigation: bool - 是否以收盘价判断消除

        返回:
            DataFrame:
                - OB: 1=看涨订单块, -1=看跌订单块
                - Top: 订单块顶部
                - Bottom: 订单块底部
                - OBVolume: 成交量
                - MitigatedIndex: 被消除的K线索引
                - Percentage: 强度百分比
        """
        ohlc_len = len(ohlc)
        _open = ohlc["open"].values
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values
        _volume = ohlc["volume"].values
        swing_hl = swing_highs_lows["HighLow"].values

        crossed = np.full(ohlc_len, False, dtype=bool)
        ob = np.zeros(ohlc_len, dtype=np.int32)
        top_arr = np.zeros(ohlc_len, dtype=np.float32)
        bottom_arr = np.zeros(ohlc_len, dtype=np.float32)
        obVolume = np.zeros(ohlc_len, dtype=np.float32)
        lowVolume = np.zeros(ohlc_len, dtype=np.float32)
        highVolume = np.zeros(ohlc_len, dtype=np.float32)
        percentage = np.zeros(ohlc_len, dtype=np.float32)
        mitigated_index = np.zeros(ohlc_len, dtype=np.int32)
        breaker = np.full(ohlc_len, False, dtype=bool)

        swing_high_indices = np.flatnonzero(swing_hl == 1)
        swing_low_indices = np.flatnonzero(swing_hl == -1)

        # Bullish Order Blocks
        active_bullish = []
        for i in range(ohlc_len):
            close_index = i
            for idx in active_bullish.copy():
                if breaker[idx]:
                    if _high[close_index] > top_arr[idx]:
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bullish.remove(idx)
                else:
                    if ((not close_mitigation and _low[close_index] < bottom_arr[idx])
                        or (close_mitigation and min(_open[close_index], _close[close_index]) < bottom_arr[idx])):
                        breaker[idx] = True
                        mitigated_index[idx] = close_index - 1

            pos = np.searchsorted(swing_high_indices, close_index)
            last_top_index = swing_high_indices[pos - 1] if pos > 0 else None

            if last_top_index is not None:
                if _close[close_index] > _high[last_top_index] and not crossed[last_top_index]:
                    crossed[last_top_index] = True
                    default_index = close_index - 1
                    obBtm = _high[default_index]
                    obTop = _low[default_index]
                    obIndex = default_index
                    if close_index - last_top_index > 1:
                        start = last_top_index + 1
                        end = close_index
                        if end > start:
                            segment = _low[start:end]
                            min_val = segment.min()
                            candidates = np.nonzero(segment == min_val)[0]
                            if candidates.size:
                                candidate_index = start + candidates[-1]
                                obBtm = _low[candidate_index]
                                obTop = _high[candidate_index]
                                obIndex = candidate_index
                    ob[obIndex] = 1
                    top_arr[obIndex] = obTop
                    bottom_arr[obIndex] = obBtm
                    vol_cur = _volume[close_index]
                    vol_prev1 = _volume[close_index - 1] if close_index >= 1 else 0.0
                    vol_prev2 = _volume[close_index - 2] if close_index >= 2 else 0.0
                    obVolume[obIndex] = vol_cur + vol_prev1 + vol_prev2
                    lowVolume[obIndex] = vol_prev2
                    highVolume[obIndex] = vol_cur + vol_prev1
                    max_vol = max(highVolume[obIndex], lowVolume[obIndex])
                    percentage[obIndex] = (min(highVolume[obIndex], lowVolume[obIndex]) / max_vol * 100.0) if max_vol != 0 else 100.0
                    active_bullish.append(obIndex)

        # Bearish Order Blocks
        active_bearish = []
        for i in range(ohlc_len):
            close_index = i
            for idx in active_bearish.copy():
                if breaker[idx]:
                    if _low[close_index] < bottom_arr[idx]:
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bearish.remove(idx)
                else:
                    if ((not close_mitigation and _high[close_index] > top_arr[idx])
                        or (close_mitigation and max(_open[close_index], _close[close_index]) > top_arr[idx])):
                        breaker[idx] = True
                        mitigated_index[idx] = close_index

            pos = np.searchsorted(swing_low_indices, close_index)
            last_btm_index = swing_low_indices[pos - 1] if pos > 0 else None

            if last_btm_index is not None:
                if _close[close_index] < _low[last_btm_index] and not crossed[last_btm_index]:
                    crossed[last_btm_index] = True
                    default_index = close_index - 1
                    obTop = _high[default_index]
                    obBtm = _low[default_index]
                    obIndex = default_index
                    if close_index - last_btm_index > 1:
                        start = last_btm_index + 1
                        end = close_index
                        if end > start:
                            segment = _high[start:end]
                            max_val = segment.max()
                            candidates = np.nonzero(segment == max_val)[0]
                            if candidates.size:
                                candidate_index = start + candidates[-1]
                                obTop = _high[candidate_index]
                                obBtm = _low[candidate_index]
                                obIndex = candidate_index
                    ob[obIndex] = -1
                    top_arr[obIndex] = obTop
                    bottom_arr[obIndex] = obBtm
                    vol_cur = _volume[close_index]
                    vol_prev1 = _volume[close_index - 1] if close_index >= 1 else 0.0
                    vol_prev2 = _volume[close_index - 2] if close_index >= 2 else 0.0
                    obVolume[obIndex] = vol_cur + vol_prev1 + vol_prev2
                    lowVolume[obIndex] = vol_cur + vol_prev1
                    highVolume[obIndex] = vol_prev2
                    max_vol = max(highVolume[obIndex], lowVolume[obIndex])
                    percentage[obIndex] = (min(highVolume[obIndex], lowVolume[obIndex]) / max_vol * 100.0) if max_vol != 0 else 100.0
                    active_bearish.append(obIndex)

        ob = np.where(ob != 0, ob, np.nan)
        top_arr = np.where(~np.isnan(ob), top_arr, np.nan)
        bottom_arr = np.where(~np.isnan(ob), bottom_arr, np.nan)
        obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
        mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
        percentage = np.where(~np.isnan(ob), percentage, np.nan)

        return pd.concat(
            [
                pd.Series(ob, name="OB"),
                pd.Series(top_arr, name="Top"),
                pd.Series(bottom_arr, name="Bottom"),
                pd.Series(obVolume, name="OBVolume"),
                pd.Series(mitigated_index, name="MitigatedIndex"),
                pd.Series(percentage, name="Percentage"),
            ],
            axis=1,
        )

    @classmethod
    def liquidity(cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01) -> DataFrame:
        """
        Liquidity (流动性)

        检测多个高点或低点聚集形成的流动性区域。

        参数:
            ohlc: DataFrame - OHLCV 数据
            swing_highs_lows: DataFrame - swing_highs_lows() 的输出
            range_percent: float - 判定流动性的价格范围百分比

        返回:
            DataFrame:
                - Liquidity: 1=看涨流动性, -1=看跌流动性
                - Level: 流动性水平
                - End: 流动性区域结束索引
                - Swept: 流动性被扫荡的K线索引
        """
        shl = swing_highs_lows.copy()
        n = len(ohlc)

        pip_range = (ohlc["high"].max() - ohlc["low"].min()) * range_percent

        ohlc_high = ohlc["high"].values
        ohlc_low = ohlc["low"].values
        shl_HL = shl["HighLow"].values.copy()
        shl_Level = shl["Level"].values.copy()

        liquidity = np.full(n, np.nan, dtype=np.float32)
        liquidity_level = np.full(n, np.nan, dtype=np.float32)
        liquidity_end = np.full(n, np.nan, dtype=np.float32)
        liquidity_swept = np.full(n, np.nan, dtype=np.float32)

        # Bullish liquidity
        bull_indices = np.nonzero(shl_HL == 1)[0]
        for i in bull_indices:
            if shl_HL[i] != 1:
                continue
            high_level = shl_Level[i]
            range_low = high_level - pip_range
            range_high = high_level + pip_range
            group_levels = [high_level]
            group_end = i

            c_start = i + 1
            if c_start < n:
                cond = ohlc_high[c_start:] >= range_high
                if np.any(cond):
                    swept = c_start + int(np.argmax(cond))
                else:
                    swept = 0
            else:
                swept = 0

            for j in bull_indices:
                if j <= i:
                    continue
                if swept and j >= swept:
                    break
                if shl_HL[j] == 1 and (range_low <= shl_Level[j] <= range_high):
                    group_levels.append(shl_Level[j])
                    group_end = j
                    shl_HL[j] = 0
            if len(group_levels) > 1:
                avg_level = sum(group_levels) / len(group_levels)
                liquidity[i] = 1
                liquidity_level[i] = avg_level
                liquidity_end[i] = group_end
                liquidity_swept[i] = swept

        # Bearish liquidity
        bear_indices = np.nonzero(shl_HL == -1)[0]
        for i in bear_indices:
            if shl_HL[i] != -1:
                continue
            low_level = shl_Level[i]
            range_low = low_level - pip_range
            range_high = low_level + pip_range
            group_levels = [low_level]
            group_end = i

            c_start = i + 1
            if c_start < n:
                cond = ohlc_low[c_start:] <= range_low
                if np.any(cond):
                    swept = c_start + int(np.argmax(cond))
                else:
                    swept = 0
            else:
                swept = 0

            for j in bear_indices:
                if j <= i:
                    continue
                if swept and j >= swept:
                    break
                if shl_HL[j] == -1 and (range_low <= shl_Level[j] <= range_high):
                    group_levels.append(shl_Level[j])
                    group_end = j
                    shl_HL[j] = 0
            if len(group_levels) > 1:
                avg_level = sum(group_levels) / len(group_levels)
                liquidity[i] = -1
                liquidity_level[i] = avg_level
                liquidity_end[i] = group_end
                liquidity_swept[i] = swept

        return pd.concat(
            [
                pd.Series(liquidity, name="Liquidity"),
                pd.Series(liquidity_level, name="Level"),
                pd.Series(liquidity_end, name="End"),
                pd.Series(liquidity_swept, name="Swept"),
            ],
            axis=1,
        )

    @classmethod
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
        """
        Previous High Low (前高/前低)

        返回指定时间周期的前高和前低。

        参数:
            ohlc: DataFrame - OHLCV 数据（索引必须是时间）
            time_frame: str - 时间周期 (15m, 1H, 4H, 1D, 1W, 1M)

        返回:
            DataFrame:
                - PreviousHigh: 前高
                - PreviousLow: 前低
                - BrokenHigh: 是否突破前高
                - BrokenLow: 是否突破前低
        """
        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)

        previous_high = np.zeros(len(ohlc), dtype=np.float32)
        previous_low = np.zeros(len(ohlc), dtype=np.float32)
        broken_high = np.zeros(len(ohlc), dtype=np.int32)
        broken_low = np.zeros(len(ohlc), dtype=np.int32)

        resampled_ohlc = ohlc.resample(time_frame).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()

        currently_broken_high = False
        currently_broken_low = False
        last_broken_time = None
        for i in range(len(ohlc)):
            resampled_previous_index = np.where(
                resampled_ohlc.index < ohlc.index[i]
            )[0]
            if len(resampled_previous_index) <= 1:
                previous_high[i] = np.nan
                previous_low[i] = np.nan
                continue
            resampled_previous_index = resampled_previous_index[-2]

            if last_broken_time != resampled_previous_index:
                currently_broken_high = False
                currently_broken_low = False
                last_broken_time = resampled_previous_index

            previous_high[i] = resampled_ohlc["high"].iloc[resampled_previous_index]
            previous_low[i] = resampled_ohlc["low"].iloc[resampled_previous_index]
            currently_broken_high = ohlc["high"].iloc[i] > previous_high[i] or currently_broken_high
            currently_broken_low = ohlc["low"].iloc[i] < previous_low[i] or currently_broken_low
            broken_high[i] = 1 if currently_broken_high else 0
            broken_low[i] = 1 if currently_broken_low else 0

        return pd.concat(
            [
                pd.Series(previous_high, name="PreviousHigh"),
                pd.Series(previous_low, name="PreviousLow"),
                pd.Series(broken_high, name="BrokenHigh"),
                pd.Series(broken_low, name="BrokenLow"),
            ],
            axis=1,
        )

    @classmethod
    def sessions(
        cls,
        ohlc: DataFrame,
        session: str,
        start_time: str = "",
        end_time: str = "",
        time_zone: str = "UTC",
    ) -> DataFrame:
        """
        Sessions (交易时段)

        识别K线是否在指定交易时段内。

        参数:
            ohlc: DataFrame - OHLCV 数据
            session: str - 时段名称:
                - Sydney: 悉尼时段
                - Tokyo: 东京时段
                - London: 伦敦时段
                - New York: 纽约时段
                - Asian kill zone: 亚洲杀戮区
                - London open kill zone: 伦敦开盘杀戮区
                - New York kill zone: 纽约杀戮区
                - london close kill zone: 伦敦收盘杀戮区
                - Custom: 自定义
            start_time: str - 自定义时段开始时间 "HH:MM"
            end_time: str - 自定义时段结束时间 "HH:MM"
            time_zone: str - 时区

        返回:
            DataFrame:
                - Active: 1=在时段内, 0=不在
                - High: 时段内最高点
                - Low: 时段内最低点
        """
        if session == "Custom" and (start_time == "" or end_time == ""):
            raise ValueError("Custom session requires a start and end time")

        default_sessions = {
            "Sydney": {"start": "21:00", "end": "06:00"},
            "Tokyo": {"start": "00:00", "end": "09:00"},
            "London": {"start": "07:00", "end": "16:00"},
            "New York": {"start": "13:00", "end": "22:00"},
            "Asian kill zone": {"start": "00:00", "end": "04:00"},
            "London open kill zone": {"start": "6:00", "end": "9:00"},
            "New York kill zone": {"start": "11:00", "end": "14:00"},
            "london close kill zone": {"start": "14:00", "end": "16:00"},
            "Custom": {"start": start_time, "end": end_time},
        }

        ohlc = ohlc.copy()
        ohlc.index = pd.to_datetime(ohlc.index)
        if time_zone != "UTC":
            time_zone = time_zone.replace("GMT", "Etc/GMT")
            time_zone = time_zone.replace("UTC", "Etc/GMT")
            ohlc.index = ohlc.index.tz_localize(time_zone).tz_convert("UTC")

        start_time = datetime.strptime(default_sessions[session]["start"], "%H:%M")
        end_time = datetime.strptime(default_sessions[session]["end"], "%H:%M")

        active = np.zeros(len(ohlc), dtype=np.int32)
        high = np.zeros(len(ohlc), dtype=np.float32)
        low = np.zeros(len(ohlc), dtype=np.float32)

        for i in range(len(ohlc)):
            current_time = datetime.strptime(ohlc.index[i].strftime("%H:%M"), "%H:%M")
            if (start_time < end_time and start_time <= current_time <= end_time) or (
                start_time >= end_time
                and (start_time <= current_time or current_time <= end_time)
            ):
                active[i] = 1
                high[i] = max(ohlc["high"].iloc[i], high[i - 1] if i > 0 else 0)
                low[i] = min(
                    ohlc["low"].iloc[i],
                    low[i - 1] if i > 0 and low[i - 1] != 0 else float("inf"),
                )

        return pd.concat(
            [
                pd.Series(active, name="Active"),
                pd.Series(high, name="High"),
                pd.Series(low, name="Low"),
            ],
            axis=1,
        )

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> DataFrame:
        """
        Retracements (回撤)

        计算从摆动高点或低点的回撤百分比。

        参数:
            ohlc: DataFrame - OHLCV 数据
            swing_highs_lows: DataFrame - swing_highs_lows() 的输出

        返回:
            DataFrame:
                - Direction: 1=看涨回撤, -1=看跌回撤
                - CurrentRetracement%: 当前回撤百分比
                - DeepestRetracement%: 最深回撤百分比
        """
        swing_highs_lows = swing_highs_lows.copy()

        direction = np.zeros(len(ohlc), dtype=np.int32)
        current_retracement = np.zeros(len(ohlc), dtype=np.float64)
        deepest_retracement = np.zeros(len(ohlc), dtype=np.float64)

        top = 0
        bottom = 0
        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"].iloc[i] == 1:
                direction[i] = 1
                top = swing_highs_lows["Level"].iloc[i]
            elif swing_highs_lows["HighLow"].iloc[i] == -1:
                direction[i] = -1
                bottom = swing_highs_lows["Level"].iloc[i]
            else:
                direction[i] = direction[i - 1] if i > 0 else 0

            if i > 0 and direction[i - 1] == 1 and top != bottom:
                current_retracement[i] = round(
                    100 - (((ohlc["low"].iloc[i] - bottom) / (top - bottom)) * 100), 1
                )
                deepest_retracement[i] = max(
                    deepest_retracement[i - 1] if direction[i - 1] == 1 else 0,
                    current_retracement[i],
                )
            if direction[i] == -1 and top != bottom:
                current_retracement[i] = round(
                    100 - ((ohlc["high"].iloc[i] - top) / (bottom - top)) * 100, 1
                )
                deepest_retracement[i] = max(
                    deepest_retracement[i - 1] if i > 0 and direction[i - 1] == -1 else 0,
                    current_retracement[i],
                )

        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):
                break
            if direction[i] != direction[i + 1]:
                remove_first_count += 1
            direction[i] = 0
            current_retracement[i] = 0
            deepest_retracement[i] = 0
            if remove_first_count == 3:
                direction[i + 1] = 0
                current_retracement[i + 1] = 0
                deepest_retracement[i + 1] = 0
                break

        return pd.concat(
            [
                pd.Series(direction, name="Direction"),
                pd.Series(current_retracement, name="CurrentRetracement%"),
                pd.Series(deepest_retracement, name="DeepestRetracement%"),
            ],
            axis=1,
        )


@dataclass
class ICTAnalysisResult:
    """ICT 分析结果"""
    swing_highs_lows: DataFrame
    fvg: DataFrame
    order_blocks: DataFrame
    bos_choch: DataFrame
    liquidity: DataFrame
    retracements: DataFrame
    sessions: Optional[DataFrame] = None

    def get_active_bullish_ob(self) -> DataFrame:
        """获取活跃的看涨订单块"""
        ob = self.order_blocks
        return ob[(ob["OB"] == 1) & (ob["MitigatedIndex"] == 0)]

    def get_active_bearish_ob(self) -> DataFrame:
        """获取活跃的看跌订单块"""
        ob = self.order_blocks
        return ob[(ob["OB"] == -1) & (ob["MitigatedIndex"] == 0)]

    def get_unfilled_fvg(self) -> DataFrame:
        """获取未填补的 FVG"""
        fvg = self.fvg
        return fvg[fvg["MitigatedIndex"] == 0]


class ICTAnalyzer:
    """
    ICT 分析器

    提供一站式 ICT 分析接口。

    使用示例:
        analyzer = ICTAnalyzer(swing_length=50)
        result = analyzer.analyze(df)

        print(f"活跃订单块: {len(result.get_active_bullish_ob())}")
        print(f"未填补FVG: {len(result.get_unfilled_fvg())}")
    """

    def __init__(
        self,
        swing_length: int = 50,
        fvg_join_consecutive: bool = False,
        ob_close_mitigation: bool = False,
        liquidity_range_percent: float = 0.01,
    ):
        """
        参数:
            swing_length: 摆动高低点的查找范围
            fvg_join_consecutive: 是否合并连续FVG
            ob_close_mitigation: 是否以收盘价判断订单块消除
            liquidity_range_percent: 流动性范围百分比
        """
        self.swing_length = swing_length
        self.fvg_join_consecutive = fvg_join_consecutive
        self.ob_close_mitigation = ob_close_mitigation
        self.liquidity_range_percent = liquidity_range_percent

    def analyze(
        self,
        ohlc: DataFrame,
        session: Optional[str] = None,
    ) -> ICTAnalysisResult:
        """
        执行完整的 ICT 分析

        参数:
            ohlc: DataFrame - OHLCV 数据
            session: str - 可选的交易时段

        返回:
            ICTAnalysisResult: 分析结果
        """
        # 确保列名小写
        ohlc = ohlc.copy()
        ohlc.columns = [c.lower() for c in ohlc.columns]

        # 计算摆动高低点
        swing_hl = smc.swing_highs_lows(ohlc, self.swing_length)

        # 计算各项指标
        fvg = smc.fvg(ohlc, self.fvg_join_consecutive)
        ob = smc.ob(ohlc, swing_hl, self.ob_close_mitigation)
        bos_choch = smc.bos_choch(ohlc, swing_hl)
        liquidity = smc.liquidity(ohlc, swing_hl, self.liquidity_range_percent)
        retracements = smc.retracements(ohlc, swing_hl)

        # 可选的时段分析
        sessions_df = None
        if session:
            sessions_df = smc.sessions(ohlc, session)

        return ICTAnalysisResult(
            swing_highs_lows=swing_hl,
            fvg=fvg,
            order_blocks=ob,
            bos_choch=bos_choch,
            liquidity=liquidity,
            retracements=retracements,
            sessions=sessions_df,
        )

    def get_bias(self, ohlc: DataFrame) -> Dict[str, Any]:
        """
        获取市场偏向

        基于 BOS/CHoCH 判断当前市场方向。
        """
        result = self.analyze(ohlc)
        bos_choch = result.bos_choch

        # 获取最近的 BOS/CHoCH
        recent_bos = bos_choch["BOS"].dropna()
        recent_choch = bos_choch["CHOCH"].dropna()

        bias = "neutral"
        confidence = 0.0

        if len(recent_bos) > 0:
            last_bos = recent_bos.iloc[-1]
            if last_bos == 1:
                bias = "bullish"
                confidence = 0.7
            elif last_bos == -1:
                bias = "bearish"
                confidence = 0.7

        if len(recent_choch) > 0:
            last_choch = recent_choch.iloc[-1]
            if last_choch == 1:
                bias = "bullish"
                confidence = 0.9  # CHoCH 信号更强
            elif last_choch == -1:
                bias = "bearish"
                confidence = 0.9

        return {
            "bias": bias,
            "confidence": confidence,
            "last_bos": recent_bos.iloc[-1] if len(recent_bos) > 0 else None,
            "last_choch": recent_choch.iloc[-1] if len(recent_choch) > 0 else None,
        }
