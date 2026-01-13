"""
期权定价和 Greeks 计算

Black-Scholes 模型实现
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionGreeks:
    """期权 Greeks"""
    delta: float      # 价格敏感度
    gamma: float      # Delta 变化率
    theta: float      # 时间衰减 (每天)
    vega: float       # 波动率敏感度
    rho: float        # 利率敏感度


class BlackScholes:
    """
    Black-Scholes 期权定价模型

    用法:
        bs = BlackScholes()

        # 计算期权价格
        price = bs.price(
            S=100,      # 标的价格
            K=105,      # 行权价
            T=30/365,   # 到期时间 (年)
            r=0.05,     # 无风险利率
            sigma=0.20, # 波动率
            option_type=OptionType.CALL
        )

        # 计算 Greeks
        greeks = bs.greeks(S=100, K=105, T=30/365, r=0.05, sigma=0.20)
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """计算 d1"""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """计算 d2"""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        计算期权价格

        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间 (年)
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型

        Returns:
            期权理论价格
        """
        if T <= 0:
            # 已到期
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType = OptionType.CALL
    ) -> OptionGreeks:
        """
        计算期权 Greeks

        Args:
            S: 标的价格
            K: 行权价
            T: 到期时间 (年)
            r: 无风险利率
            sigma: 波动率
            option_type: 期权类型

        Returns:
            OptionGreeks 对象
        """
        if T <= 0:
            # 已到期，Greeks 为极端值或 0
            intrinsic = max(S - K, 0) if option_type == OptionType.CALL else max(K - S, 0)
            delta = 1.0 if intrinsic > 0 else 0.0
            if option_type == OptionType.PUT:
                delta = -delta
            return OptionGreeks(delta=delta, gamma=0, theta=0, vega=0, rho=0)

        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(S, K, T, r, sigma)

        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (Call 和 Put 相同)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = (term1 + term2) / 365  # 每天

        # Vega (Call 和 Put 相同)
        vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # 每 1% 波动率

        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType = OptionType.CALL,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        计算隐含波动率 (牛顿迭代法)

        Args:
            market_price: 市场价格
            S: 标的价格
            K: 行权价
            T: 到期时间 (年)
            r: 无风险利率
            option_type: 期权类型

        Returns:
            隐含波动率
        """
        sigma = 0.20  # 初始猜测

        for _ in range(max_iterations):
            price = self.price(S, K, T, r, sigma, option_type)
            diff = market_price - price

            if abs(diff) < tolerance:
                return sigma

            # Vega
            d1 = self.d1(S, K, T, r, sigma)
            vega = S * np.sqrt(T) * norm.pdf(d1)

            if vega < 1e-10:
                break

            sigma += diff / vega

            # 边界检查
            sigma = max(0.001, min(5.0, sigma))

        return sigma


class OptionAnalyzer:
    """
    期权分析工具

    用法:
        analyzer = OptionAnalyzer()

        # 分析单个期权
        analysis = analyzer.analyze_option(
            S=100, K=105, T=30/365, r=0.05, sigma=0.20,
            option_type=OptionType.CALL, market_price=2.50
        )

        # 分析价差策略
        pnl = analyzer.spread_pnl(
            long_strike=100, short_strike=105,
            long_premium=3.0, short_premium=1.5,
            underlying_at_expiry=102
        )
    """

    def __init__(self):
        self.bs = BlackScholes()

    def analyze_option(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        market_price: float = None
    ) -> dict:
        """
        分析单个期权

        Returns:
            包含价格、Greeks、内在价值等的字典
        """
        theoretical_price = self.bs.price(S, K, T, r, sigma, option_type)
        greeks = self.bs.greeks(S, K, T, r, sigma, option_type)

        # 内在价值
        if option_type == OptionType.CALL:
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)

        # 时间价值
        time_value = theoretical_price - intrinsic

        result = {
            'theoretical_price': theoretical_price,
            'intrinsic_value': intrinsic,
            'time_value': time_value,
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'theta': greeks.theta,
            'vega': greeks.vega,
            'rho': greeks.rho,
        }

        # 如果有市场价格，计算隐含波动率
        if market_price:
            iv = self.bs.implied_volatility(market_price, S, K, T, r, option_type)
            result['implied_volatility'] = iv
            result['market_price'] = market_price
            result['mispricing'] = market_price - theoretical_price

        return result

    def breakeven(
        self,
        K: float,
        premium: float,
        option_type: OptionType
    ) -> float:
        """计算盈亏平衡点"""
        if option_type == OptionType.CALL:
            return K + premium
        else:
            return K - premium

    def max_profit_loss(
        self,
        strategy: str,
        **kwargs
    ) -> Tuple[float, float]:
        """
        计算策略最大盈亏

        Args:
            strategy: 策略名称
            **kwargs: 策略参数

        Returns:
            (最大收益, 最大损失)
        """
        if strategy == 'long_call':
            premium = kwargs['premium']
            return (float('inf'), -premium)

        elif strategy == 'long_put':
            premium = kwargs['premium']
            strike = kwargs['strike']
            return (strike - premium, -premium)

        elif strategy == 'covered_call':
            stock_price = kwargs['stock_price']
            strike = kwargs['strike']
            premium = kwargs['premium']
            max_profit = strike - stock_price + premium
            max_loss = stock_price - premium  # 股票归零
            return (max_profit, -max_loss)

        elif strategy == 'bull_call_spread':
            long_premium = kwargs['long_premium']
            short_premium = kwargs['short_premium']
            long_strike = kwargs['long_strike']
            short_strike = kwargs['short_strike']
            net_debit = long_premium - short_premium
            max_profit = short_strike - long_strike - net_debit
            max_loss = net_debit
            return (max_profit, -max_loss)

        elif strategy == 'iron_condor':
            credit = kwargs['credit']
            wing_width = kwargs['wing_width']
            max_profit = credit
            max_loss = wing_width - credit
            return (max_profit, -max_loss)

        return (0, 0)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    bs = BlackScholes()
    analyzer = OptionAnalyzer()

    print("=" * 60)
    print("期权定价示例")
    print("=" * 60)

    # 参数
    S = 100      # 标的价格
    K = 105      # 行权价
    T = 30/365   # 30天到期
    r = 0.05     # 5% 无风险利率
    sigma = 0.20 # 20% 波动率

    # Call 价格
    call_price = bs.price(S, K, T, r, sigma, OptionType.CALL)
    print(f"\nCall 期权 (S={S}, K={K}, T=30天, σ=20%)")
    print(f"  理论价格: ${call_price:.2f}")

    # Greeks
    greeks = bs.greeks(S, K, T, r, sigma, OptionType.CALL)
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.4f}")
    print(f"  Theta: ${greeks.theta:.4f}/天")
    print(f"  Vega:  ${greeks.vega:.4f}/1%σ")

    # Put 价格
    put_price = bs.price(S, K, T, r, sigma, OptionType.PUT)
    print(f"\nPut 期权 (S={S}, K={K}, T=30天, σ=20%)")
    print(f"  理论价格: ${put_price:.2f}")

    # 隐含波动率
    print("\n隐含波动率计算:")
    market_price = 2.50
    iv = bs.implied_volatility(market_price, S, K, T, r, OptionType.CALL)
    print(f"  市场价格: ${market_price}")
    print(f"  隐含波动率: {iv:.2%}")

    # 策略分析
    print("\n" + "=" * 60)
    print("策略盈亏分析")
    print("=" * 60)

    # Bull Call Spread
    max_profit, max_loss = analyzer.max_profit_loss(
        'bull_call_spread',
        long_strike=100,
        short_strike=105,
        long_premium=3.0,
        short_premium=1.5
    )
    print(f"\nBull Call Spread (100/105):")
    print(f"  净支出: $1.50")
    print(f"  最大收益: ${max_profit:.2f}")
    print(f"  最大损失: ${max_loss:.2f}")
    print(f"  盈亏比: {abs(max_profit/max_loss):.1f}:1")
