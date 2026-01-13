"""
期权策略模板 - Lean Engine

常用期权策略的 Lean 实现
"""

# =============================================================================
# 1. 备兑看涨 (Covered Call)
# =============================================================================

COVERED_CALL = '''
from AlgorithmImports import *

class CoveredCallStrategy(QCAlgorithm):
    """
    备兑看涨期权策略

    - 持有股票
    - 卖出 OTM 看涨期权收取权利金
    - 期权到期后滚动到下一个周期
    """

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # 标的股票
        equity = self.AddEquity("SPY", Resolution.Minute)
        self.symbol = equity.Symbol

        # 期权链
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-5, 5)
                                    .Expiration(timedelta(days=25), timedelta(days=40)))
        self.option_symbol = option.Symbol

        # 状态
        self.sold_call = None

        # 参数
        self.otm_percent = 0.02  # 2% OTM
        self.target_delta = 0.30  # 目标 Delta

    def OnData(self, data):
        # 1. 如果没有股票，先买入
        if not self.Portfolio[self.symbol].Invested:
            self.SetHoldings(self.symbol, 0.9)  # 90% 仓位
            return

        # 2. 如果没有期权仓位，卖出看涨期权
        if self.sold_call is None:
            chain = data.OptionChains.get(self.option_symbol)
            if chain:
                self._sell_covered_call(chain)

    def _sell_covered_call(self, chain):
        """卖出备兑看涨期权"""
        underlying_price = chain.Underlying.Price

        # 筛选 OTM 看涨期权
        target_strike = underlying_price * (1 + self.otm_percent)
        calls = [x for x in chain
                 if x.Right == OptionRight.Call
                 and x.Strike >= target_strike]

        if not calls:
            return

        # 选择最近到期的
        calls = sorted(calls, key=lambda x: (x.Expiry, x.Strike))

        # 可选: 按 Delta 筛选
        # calls = [x for x in calls if abs(x.Greeks.Delta) < self.target_delta + 0.1]

        if calls:
            contract = calls[0]
            # 卖出数量 = 持股数量 / 100
            quantity = self.Portfolio[self.symbol].Quantity // 100
            if quantity > 0:
                self.Sell(contract.Symbol, quantity)
                self.sold_call = contract.Symbol
                self.Debug(f"Sold Call: {contract.Symbol}, Strike: {contract.Strike}, Expiry: {contract.Expiry}")

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            # 期权平仓或到期后重置
            if self.sold_call and orderEvent.Symbol == self.sold_call:
                if orderEvent.Direction == OrderDirection.Buy:
                    self.sold_call = None
'''


# =============================================================================
# 2. 保护性看跌 (Protective Put)
# =============================================================================

PROTECTIVE_PUT = '''
from AlgorithmImports import *

class ProtectivePutStrategy(QCAlgorithm):
    """
    保护性看跌期权策略

    - 持有股票
    - 买入看跌期权保护下行风险
    """

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        equity = self.AddEquity("SPY", Resolution.Minute)
        self.symbol = equity.Symbol

        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-10, 0)
                                    .Expiration(timedelta(days=30), timedelta(days=60)))

        self.bought_put = None
        self.protection_level = 0.05  # 5% 保护

    def OnData(self, data):
        # 买入股票
        if not self.Portfolio[self.symbol].Invested:
            self.SetHoldings(self.symbol, 0.9)

        # 买入保护性看跌
        if self.bought_put is None:
            chain = data.OptionChains.get(self.symbol)
            if chain:
                self._buy_protective_put(chain)

    def _buy_protective_put(self, chain):
        underlying_price = chain.Underlying.Price
        target_strike = underlying_price * (1 - self.protection_level)

        puts = [x for x in chain
                if x.Right == OptionRight.Put
                and x.Strike <= target_strike]

        if puts:
            # 选择最接近目标行权价的
            contract = max(puts, key=lambda x: x.Strike)
            quantity = self.Portfolio[self.symbol].Quantity // 100
            if quantity > 0:
                self.Buy(contract.Symbol, quantity)
                self.bought_put = contract.Symbol
'''


# =============================================================================
# 3. 铁鹰策略 (Iron Condor)
# =============================================================================

IRON_CONDOR = '''
from AlgorithmImports import *

class IronCondorStrategy(QCAlgorithm):
    """
    铁鹰策略 (Iron Condor)

    - 卖出 OTM Put + 卖出 OTM Call (收取权利金)
    - 买入更远 OTM Put + Call (限制风险)
    - 适合低波动、区间震荡市场
    """

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        equity = self.AddEquity("SPY", Resolution.Minute)
        self.symbol = equity.Symbol

        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-15, 15)
                                    .Expiration(timedelta(days=30), timedelta(days=45)))
        self.option_symbol = option.Symbol

        self.has_position = False
        self.wing_width = 5  # 翼宽 (行权价间距)

    def OnData(self, data):
        if self.has_position:
            return

        chain = data.OptionChains.get(self.option_symbol)
        if not chain:
            return

        underlying_price = chain.Underlying.Price

        # 找到 ATM 行权价
        atm_strike = min([x.Strike for x in chain],
                        key=lambda s: abs(s - underlying_price))

        # 构建四条腿
        short_put_strike = atm_strike - self.wing_width
        long_put_strike = short_put_strike - self.wing_width
        short_call_strike = atm_strike + self.wing_width
        long_call_strike = short_call_strike + self.wing_width

        # 查找合约
        expiry = min(set(x.Expiry for x in chain))

        contracts = {
            'short_put': None,
            'long_put': None,
            'short_call': None,
            'long_call': None
        }

        for x in chain:
            if x.Expiry == expiry:
                if x.Right == OptionRight.Put:
                    if x.Strike == short_put_strike:
                        contracts['short_put'] = x
                    elif x.Strike == long_put_strike:
                        contracts['long_put'] = x
                elif x.Right == OptionRight.Call:
                    if x.Strike == short_call_strike:
                        contracts['short_call'] = x
                    elif x.Strike == long_call_strike:
                        contracts['long_call'] = x

        # 检查是否找到所有合约
        if all(contracts.values()):
            # 执行铁鹰
            self.Sell(contracts['short_put'].Symbol, 1)   # 卖出看跌
            self.Buy(contracts['long_put'].Symbol, 1)     # 买入更低看跌
            self.Sell(contracts['short_call'].Symbol, 1)  # 卖出看涨
            self.Buy(contracts['long_call'].Symbol, 1)    # 买入更高看涨

            self.has_position = True
            self.Debug(f"Iron Condor opened: {short_put_strike}/{long_put_strike} - {short_call_strike}/{long_call_strike}")

    def OnOrderEvent(self, orderEvent):
        self.Debug(f"Order: {orderEvent}")
'''


# =============================================================================
# 4. 跨式策略 (Straddle)
# =============================================================================

STRADDLE = '''
from AlgorithmImports import *

class StraddleStrategy(QCAlgorithm):
    """
    跨式策略 (Straddle)

    - 同时买入 ATM Call + ATM Put
    - 预期大幅波动但方向不确定
    - 需要波动超过权利金成本才能盈利
    """

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        equity = self.AddEquity("SPY", Resolution.Minute)
        self.symbol = equity.Symbol

        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-2, 2)
                                    .Expiration(timedelta(days=7), timedelta(days=14)))

        self.has_position = False

        # 波动率阈值 - 只在低 IV 时建仓
        self.iv_threshold = 0.20

    def OnData(self, data):
        if self.has_position:
            return

        chain = data.OptionChains.get(self.symbol)
        if not chain:
            return

        underlying_price = chain.Underlying.Price

        # 找 ATM 期权
        atm_strike = min([x.Strike for x in chain],
                        key=lambda s: abs(s - underlying_price))

        expiry = min(set(x.Expiry for x in chain))

        atm_call = None
        atm_put = None

        for x in chain:
            if x.Strike == atm_strike and x.Expiry == expiry:
                if x.Right == OptionRight.Call:
                    atm_call = x
                elif x.Right == OptionRight.Put:
                    atm_put = x

        if atm_call and atm_put:
            # 检查 IV
            avg_iv = (atm_call.ImpliedVolatility + atm_put.ImpliedVolatility) / 2
            if avg_iv < self.iv_threshold:
                self.Buy(atm_call.Symbol, 1)
                self.Buy(atm_put.Symbol, 1)
                self.has_position = True
                self.Debug(f"Straddle opened at {atm_strike}, IV: {avg_iv:.2%}")
'''


# =============================================================================
# 5. 垂直价差 (Vertical Spread)
# =============================================================================

BULL_CALL_SPREAD = '''
from AlgorithmImports import *

class BullCallSpreadStrategy(QCAlgorithm):
    """
    牛市看涨价差 (Bull Call Spread)

    - 买入较低行权价 Call
    - 卖出较高行权价 Call
    - 看涨但限制成本和收益
    """

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        equity = self.AddEquity("SPY", Resolution.Minute)
        self.symbol = equity.Symbol

        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-5, 10)
                                    .Expiration(timedelta(days=30), timedelta(days=45)))

        # 技术指标判断趋势
        self.ema_fast = self.EMA(self.symbol, 20)
        self.ema_slow = self.EMA(self.symbol, 50)

        self.has_position = False
        self.spread_width = 5  # 价差宽度

    def OnData(self, data):
        if not self.ema_fast.IsReady:
            return

        if self.has_position:
            return

        # 只在上升趋势时建仓
        if self.ema_fast.Current.Value <= self.ema_slow.Current.Value:
            return

        chain = data.OptionChains.get(self.symbol)
        if not chain:
            return

        underlying_price = chain.Underlying.Price

        # 找 ATM 和 OTM Call
        atm_strike = min([x.Strike for x in chain if x.Right == OptionRight.Call],
                        key=lambda s: abs(s - underlying_price))
        otm_strike = atm_strike + self.spread_width

        expiry = min(set(x.Expiry for x in chain))

        long_call = None
        short_call = None

        for x in chain:
            if x.Expiry == expiry and x.Right == OptionRight.Call:
                if x.Strike == atm_strike:
                    long_call = x
                elif x.Strike == otm_strike:
                    short_call = x

        if long_call and short_call:
            self.Buy(long_call.Symbol, 1)
            self.Sell(short_call.Symbol, 1)
            self.has_position = True
            self.Debug(f"Bull Call Spread: Long {atm_strike}, Short {otm_strike}")
'''


# =============================================================================
# 策略选择器
# =============================================================================

STRATEGIES = {
    'covered_call': COVERED_CALL,
    'protective_put': PROTECTIVE_PUT,
    'iron_condor': IRON_CONDOR,
    'straddle': STRADDLE,
    'bull_call_spread': BULL_CALL_SPREAD,
}


def get_strategy(name: str) -> str:
    """获取策略代码"""
    return STRATEGIES.get(name, '')


def list_strategies() -> list:
    """列出所有策略"""
    return list(STRATEGIES.keys())


if __name__ == "__main__":
    print("可用期权策略:")
    for name in list_strategies():
        print(f"  - {name}")
