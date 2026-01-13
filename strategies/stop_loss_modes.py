"""
止损止盈模式实现

支持多种止损止盈策略:
1. 固定止损 (Fixed Stop Loss)
2. ATR 动态止损 (ATR-based Stop Loss)
3. 追踪止损 (Trailing Stop Loss)
4. 追踪止盈 (Trailing Take Profit)
5. 百分比止损止盈 (Percentage-based)
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class StopLossMode(Enum):
    """止损模式"""
    FIXED = "fixed"           # 固定价格
    FIXED_PCT = "fixed_pct"   # 固定百分比
    ATR = "atr"               # ATR 倍数
    TRAILING = "trailing"     # 追踪止损
    TRAILING_ATR = "trailing_atr"  # ATR 追踪止损


class TakeProfitMode(Enum):
    """止盈模式"""
    FIXED = "fixed"           # 固定价格
    FIXED_PCT = "fixed_pct"   # 固定百分比
    ATR = "atr"               # ATR 倍数
    TRAILING = "trailing"     # 追踪止盈
    RR_RATIO = "rr_ratio"     # 风险回报比


@dataclass
class StopLossManager:
    """
    止损止盈管理器

    用法:
        manager = StopLossManager(
            sl_mode=StopLossMode.ATR,
            tp_mode=TakeProfitMode.TRAILING,
            atr_mult=2.0,
            trail_atr_mult=3.0
        )

        # 开仓时初始化
        manager.on_entry(side='long', entry_price=100, atr=2.5)

        # 每根K线更新
        sl, tp = manager.update(high=105, low=99, close=103)

        # 检查是否触发
        triggered, exit_price, reason = manager.check_exit(high=105, low=99)
    """

    # 止损设置
    sl_mode: StopLossMode = StopLossMode.ATR
    sl_fixed_price: float = 0.0
    sl_fixed_pct: float = 2.0  # 2%
    sl_atr_mult: float = 2.0
    sl_trail_pct: float = 1.0  # 追踪止损回撤百分比
    sl_trail_atr_mult: float = 2.0  # 追踪止损 ATR 倍数

    # 止盈设置
    tp_mode: TakeProfitMode = TakeProfitMode.TRAILING
    tp_fixed_price: float = 0.0
    tp_fixed_pct: float = 4.0  # 4%
    tp_atr_mult: float = 3.0
    tp_trail_pct: float = 1.5  # 追踪止盈回撤百分比
    tp_rr_ratio: float = 2.0  # 风险回报比

    # 内部状态
    _side: str = None  # 'long' or 'short'
    _entry_price: float = 0.0
    _entry_atr: float = 0.0
    _stop_loss: float = 0.0
    _take_profit: float = 0.0
    _highest_since_entry: float = 0.0  # 入场后最高价
    _lowest_since_entry: float = float('inf')  # 入场后最低价
    _in_position: bool = False

    def on_entry(
        self,
        side: str,
        entry_price: float,
        atr: float = None,
        fixed_sl: float = None,
        fixed_tp: float = None
    ):
        """
        开仓时调用

        Args:
            side: 'long' or 'short'
            entry_price: 入场价格
            atr: 当前 ATR 值 (如果使用 ATR 模式)
            fixed_sl: 固定止损价格 (如果使用固定模式)
            fixed_tp: 固定止盈价格 (如果使用固定模式)
        """
        self._side = side
        self._entry_price = entry_price
        self._entry_atr = atr or 0.0
        self._highest_since_entry = entry_price
        self._lowest_since_entry = entry_price
        self._in_position = True

        # 计算初始止损
        self._stop_loss = self._calc_initial_stop_loss(fixed_sl)

        # 计算初始止盈
        self._take_profit = self._calc_initial_take_profit(fixed_tp)

    def _calc_initial_stop_loss(self, fixed_sl: float = None) -> float:
        """计算初始止损价格"""
        if fixed_sl is not None:
            return fixed_sl

        is_long = self._side == 'long'

        if self.sl_mode == StopLossMode.FIXED:
            return self.sl_fixed_price

        elif self.sl_mode == StopLossMode.FIXED_PCT:
            if is_long:
                return self._entry_price * (1 - self.sl_fixed_pct / 100)
            else:
                return self._entry_price * (1 + self.sl_fixed_pct / 100)

        elif self.sl_mode in (StopLossMode.ATR, StopLossMode.TRAILING_ATR):
            if is_long:
                return self._entry_price - self._entry_atr * self.sl_atr_mult
            else:
                return self._entry_price + self._entry_atr * self.sl_atr_mult

        elif self.sl_mode == StopLossMode.TRAILING:
            # 追踪止损初始值 = 入场价
            if is_long:
                return self._entry_price * (1 - self.sl_trail_pct / 100)
            else:
                return self._entry_price * (1 + self.sl_trail_pct / 100)

        return 0.0

    def _calc_initial_take_profit(self, fixed_tp: float = None) -> float:
        """计算初始止盈价格"""
        if fixed_tp is not None:
            return fixed_tp

        is_long = self._side == 'long'

        if self.tp_mode == TakeProfitMode.FIXED:
            return self.tp_fixed_price

        elif self.tp_mode == TakeProfitMode.FIXED_PCT:
            if is_long:
                return self._entry_price * (1 + self.tp_fixed_pct / 100)
            else:
                return self._entry_price * (1 - self.tp_fixed_pct / 100)

        elif self.tp_mode == TakeProfitMode.ATR:
            if is_long:
                return self._entry_price + self._entry_atr * self.tp_atr_mult
            else:
                return self._entry_price - self._entry_atr * self.tp_atr_mult

        elif self.tp_mode == TakeProfitMode.RR_RATIO:
            # 根据止损距离计算止盈
            risk = abs(self._entry_price - self._stop_loss)
            if is_long:
                return self._entry_price + risk * self.tp_rr_ratio
            else:
                return self._entry_price - risk * self.tp_rr_ratio

        elif self.tp_mode == TakeProfitMode.TRAILING:
            # 追踪止盈初始值 = 无限远
            return float('inf') if is_long else 0.0

        return 0.0

    def update(
        self,
        high: float,
        low: float,
        close: float,
        atr: float = None
    ) -> Tuple[float, float]:
        """
        每根K线更新止损止盈

        Args:
            high: 当前K线最高价
            low: 当前K线最低价
            close: 当前K线收盘价
            atr: 当前 ATR 值 (用于动态 ATR 追踪)

        Returns:
            (stop_loss, take_profit) 当前止损止盈价格
        """
        if not self._in_position:
            return self._stop_loss, self._take_profit

        is_long = self._side == 'long'

        # 更新最高/最低价
        self._highest_since_entry = max(self._highest_since_entry, high)
        self._lowest_since_entry = min(self._lowest_since_entry, low)

        # 更新追踪止损
        if self.sl_mode == StopLossMode.TRAILING:
            if is_long:
                # 多头: 止损跟随最高价
                new_sl = self._highest_since_entry * (1 - self.sl_trail_pct / 100)
                self._stop_loss = max(self._stop_loss, new_sl)
            else:
                # 空头: 止损跟随最低价
                new_sl = self._lowest_since_entry * (1 + self.sl_trail_pct / 100)
                self._stop_loss = min(self._stop_loss, new_sl)

        elif self.sl_mode == StopLossMode.TRAILING_ATR and atr is not None:
            if is_long:
                new_sl = self._highest_since_entry - atr * self.sl_trail_atr_mult
                self._stop_loss = max(self._stop_loss, new_sl)
            else:
                new_sl = self._lowest_since_entry + atr * self.sl_trail_atr_mult
                self._stop_loss = min(self._stop_loss, new_sl)

        # 更新追踪止盈
        if self.tp_mode == TakeProfitMode.TRAILING:
            if is_long:
                # 多头: 当价格创新高后，设置追踪止盈
                if high > self._entry_price * (1 + self.tp_trail_pct / 100):
                    new_tp = self._highest_since_entry * (1 - self.tp_trail_pct / 100)
                    if self._take_profit == float('inf'):
                        self._take_profit = new_tp
                    else:
                        self._take_profit = max(self._take_profit, new_tp)
            else:
                # 空头: 当价格创新低后，设置追踪止盈
                if low < self._entry_price * (1 - self.tp_trail_pct / 100):
                    new_tp = self._lowest_since_entry * (1 + self.tp_trail_pct / 100)
                    if self._take_profit == 0.0:
                        self._take_profit = new_tp
                    else:
                        self._take_profit = min(self._take_profit, new_tp)

        return self._stop_loss, self._take_profit

    def check_exit(
        self,
        high: float,
        low: float
    ) -> Tuple[bool, float, str]:
        """
        检查是否触发止损止盈

        Args:
            high: 当前K线最高价
            low: 当前K线最低价

        Returns:
            (triggered, exit_price, reason)
            - triggered: 是否触发
            - exit_price: 出场价格
            - reason: 'stop_loss' or 'take_profit'
        """
        if not self._in_position:
            return False, 0.0, ''

        is_long = self._side == 'long'

        if is_long:
            # 多头: 低于止损 或 高于止盈
            if low <= self._stop_loss:
                return True, self._stop_loss, 'stop_loss'
            if self._take_profit != float('inf') and high >= self._take_profit:
                return True, self._take_profit, 'take_profit'
        else:
            # 空头: 高于止损 或 低于止盈
            if high >= self._stop_loss:
                return True, self._stop_loss, 'stop_loss'
            if self._take_profit != 0.0 and low <= self._take_profit:
                return True, self._take_profit, 'take_profit'

        return False, 0.0, ''

    def on_exit(self):
        """出场时调用，重置状态"""
        self._in_position = False
        self._side = None
        self._entry_price = 0.0
        self._highest_since_entry = 0.0
        self._lowest_since_entry = float('inf')

    @property
    def stop_loss(self) -> float:
        return self._stop_loss

    @property
    def take_profit(self) -> float:
        return self._take_profit

    @property
    def in_position(self) -> bool:
        return self._in_position


# =============================================================================
# 使用示例
# =============================================================================

def example_fixed_atr_stop():
    """示例: 固定 ATR 止损止盈"""
    print("=" * 50)
    print("固定 ATR 止损止盈")
    print("=" * 50)

    manager = StopLossManager(
        sl_mode=StopLossMode.ATR,
        sl_atr_mult=2.0,
        tp_mode=TakeProfitMode.ATR,
        tp_atr_mult=3.0
    )

    # 模拟开多仓
    entry_price = 100.0
    atr = 2.5
    manager.on_entry('long', entry_price, atr)

    print(f"入场价: {entry_price}")
    print(f"ATR: {atr}")
    print(f"止损: {manager.stop_loss} (入场价 - 2 × ATR)")
    print(f"止盈: {manager.take_profit} (入场价 + 3 × ATR)")


def example_trailing_stop():
    """示例: 追踪止损"""
    print("\n" + "=" * 50)
    print("追踪止损 (1.5% 回撤)")
    print("=" * 50)

    manager = StopLossManager(
        sl_mode=StopLossMode.TRAILING,
        sl_trail_pct=1.5,  # 1.5% 回撤止损
        tp_mode=TakeProfitMode.FIXED_PCT,
        tp_fixed_pct=10.0  # 10% 固定止盈
    )

    # 模拟开多仓
    entry_price = 100.0
    manager.on_entry('long', entry_price)

    print(f"入场价: {entry_price}")
    print(f"初始止损: {manager.stop_loss:.2f}")
    print(f"止盈: {manager.take_profit:.2f}")

    # 模拟价格上涨
    prices = [
        (101, 99.5, 100.5),   # 小幅波动
        (103, 100, 102),      # 上涨
        (105, 102, 104),      # 继续上涨
        (106, 103, 105),      # 新高
        (105, 103, 104),      # 回落
    ]

    print("\n价格变动:")
    for high, low, close in prices:
        sl, tp = manager.update(high, low, close)
        triggered, exit_price, reason = manager.check_exit(high, low)
        print(f"  H={high}, L={low}, C={close} -> 止损={sl:.2f}")
        if triggered:
            print(f"  触发{reason}: {exit_price:.2f}")
            break


def example_trailing_take_profit():
    """示例: 追踪止盈"""
    print("\n" + "=" * 50)
    print("追踪止盈 (2% 回撤锁定利润)")
    print("=" * 50)

    manager = StopLossManager(
        sl_mode=StopLossMode.ATR,
        sl_atr_mult=2.0,
        tp_mode=TakeProfitMode.TRAILING,
        tp_trail_pct=2.0  # 2% 回撤止盈
    )

    # 模拟开多仓
    entry_price = 100.0
    atr = 2.0
    manager.on_entry('long', entry_price, atr)

    print(f"入场价: {entry_price}")
    print(f"初始止损: {manager.stop_loss:.2f}")
    print(f"追踪止盈: 价格上涨超过 {manager.tp_trail_pct}% 后激活")

    # 模拟价格大幅上涨后回落
    prices = [
        (102, 99, 101),      # 小涨
        (105, 101, 104),     # 涨超过 2%, 激活追踪止盈
        (110, 104, 109),     # 继续上涨
        (112, 108, 111),     # 新高
        (111, 107, 108),     # 回落触发追踪止盈
    ]

    print("\n价格变动:")
    for high, low, close in prices:
        sl, tp = manager.update(high, low, close)
        triggered, exit_price, reason = manager.check_exit(high, low)
        tp_display = f"{tp:.2f}" if tp != float('inf') else "未激活"
        print(f"  H={high}, L={low}, C={close} -> 止损={sl:.2f}, 追踪止盈={tp_display}")
        if triggered:
            print(f"  触发{reason}: {exit_price:.2f}")
            break


def example_risk_reward():
    """示例: 风险回报比"""
    print("\n" + "=" * 50)
    print("风险回报比 (1:2)")
    print("=" * 50)

    manager = StopLossManager(
        sl_mode=StopLossMode.FIXED_PCT,
        sl_fixed_pct=2.0,  # 2% 固定止损
        tp_mode=TakeProfitMode.RR_RATIO,
        tp_rr_ratio=2.0  # 2:1 风险回报比
    )

    # 模拟开多仓
    entry_price = 100.0
    manager.on_entry('long', entry_price)

    risk = entry_price * 0.02  # 2% = 2.0
    reward = risk * 2.0  # 2:1 = 4.0

    print(f"入场价: {entry_price}")
    print(f"止损: {manager.stop_loss:.2f} (风险 = {risk:.2f})")
    print(f"止盈: {manager.take_profit:.2f} (回报 = {reward:.2f})")
    print(f"风险回报比: 1:{manager.tp_rr_ratio}")


if __name__ == "__main__":
    example_fixed_atr_stop()
    example_trailing_stop()
    example_trailing_take_profit()
    example_risk_reward()
