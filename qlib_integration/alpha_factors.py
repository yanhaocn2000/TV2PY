"""
Alpha Factor Mining with Qlib
使用 Qlib 进行 Alpha 因子挖掘和评估
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime


class FactorCategory(Enum):
    """因子类别"""
    MOMENTUM = "momentum"       # 动量因子
    VALUE = "value"            # 价值因子
    QUALITY = "quality"        # 质量因子
    VOLATILITY = "volatility"  # 波动率因子
    LIQUIDITY = "liquidity"    # 流动性因子
    TECHNICAL = "technical"    # 技术因子
    SENTIMENT = "sentiment"    # 情绪因子


@dataclass
class AlphaFactor:
    """Alpha 因子定义"""
    name: str
    expression: str  # Qlib 表达式
    category: FactorCategory
    description: str = ""
    ic: float = 0.0  # 信息系数
    ir: float = 0.0  # 信息比率
    turnover: float = 0.0  # 换手率
    returns: float = 0.0  # 因子收益

    @property
    def quality_score(self) -> float:
        """因子质量评分 (0-100)"""
        # IC 贡献 (40%)
        ic_score = min(abs(self.ic) * 10, 1) * 40

        # IR 贡献 (30%)
        ir_score = min(abs(self.ir) / 2, 1) * 30

        # 换手率惩罚 (15%) - 低换手率更好
        turnover_score = max(0, (1 - self.turnover)) * 15

        # 收益贡献 (15%)
        return_score = min(max(self.returns, 0) * 5, 1) * 15

        return ic_score + ir_score + turnover_score + return_score


class AlphaFactorMiner:
    """
    Alpha 因子挖掘器

    功能:
    1. 从 TV2PY 指标生成候选因子
    2. 使用 Qlib 评估因子质量
    3. 因子组合优化
    """

    # 预定义的技术因子表达式 (基于 TV 指标)
    TECHNICAL_FACTORS = {
        # 价格动量
        "mom_5d": ("Ref($close, 5)/$close - 1", FactorCategory.MOMENTUM),
        "mom_10d": ("Ref($close, 10)/$close - 1", FactorCategory.MOMENTUM),
        "mom_20d": ("Ref($close, 20)/$close - 1", FactorCategory.MOMENTUM),
        "mom_60d": ("Ref($close, 60)/$close - 1", FactorCategory.MOMENTUM),

        # 反转因子
        "reversal_5d": ("1 - Ref($close, 5)/$close", FactorCategory.MOMENTUM),
        "reversal_20d": ("1 - Ref($close, 20)/$close", FactorCategory.MOMENTUM),

        # 波动率因子
        "volatility_5d": ("Std($close/Ref($close, 1)-1, 5)", FactorCategory.VOLATILITY),
        "volatility_20d": ("Std($close/Ref($close, 1)-1, 20)", FactorCategory.VOLATILITY),
        "volatility_60d": ("Std($close/Ref($close, 1)-1, 60)", FactorCategory.VOLATILITY),

        # ATR
        "atr_14": ("Mean(Max(Max($high-$low, Abs($high-Ref($close,1))), "
                   "Abs($low-Ref($close,1))), 14)", FactorCategory.VOLATILITY),

        # RSI
        "rsi_14": ("100 - 100/(1 + Mean(Max($close-Ref($close,1),0),14)/"
                   "Mean(Max(Ref($close,1)-$close,0),14))", FactorCategory.TECHNICAL),

        # MACD
        "macd": ("EMA($close, 12) - EMA($close, 26)", FactorCategory.TECHNICAL),
        "macd_signal": ("EMA(EMA($close, 12) - EMA($close, 26), 9)", FactorCategory.TECHNICAL),
        "macd_hist": ("(EMA($close, 12) - EMA($close, 26)) - "
                      "EMA(EMA($close, 12) - EMA($close, 26), 9)", FactorCategory.TECHNICAL),

        # 布林带
        "bbands_width": ("(Mean($close, 20) + 2*Std($close, 20) - "
                         "(Mean($close, 20) - 2*Std($close, 20))) / Mean($close, 20)",
                         FactorCategory.VOLATILITY),
        "bbands_position": ("($close - (Mean($close, 20) - 2*Std($close, 20))) / "
                            "(4*Std($close, 20))", FactorCategory.TECHNICAL),

        # 成交量因子
        "volume_ratio": ("$volume / Mean($volume, 20)", FactorCategory.LIQUIDITY),
        "volume_ma_ratio": ("Mean($volume, 5) / Mean($volume, 20)", FactorCategory.LIQUIDITY),
        "turnover_rate": ("$volume * $close / $market_cap", FactorCategory.LIQUIDITY),

        # 价格位置
        "high_low_position": ("($close - $low) / ($high - $low + 1e-8)",
                               FactorCategory.TECHNICAL),
        "close_to_high": ("$close / Max($high, 20) - 1", FactorCategory.TECHNICAL),
        "close_to_low": ("$close / Min($low, 20) - 1", FactorCategory.TECHNICAL),

        # 均线因子
        "ma_cross_5_20": ("EMA($close, 5) / EMA($close, 20) - 1", FactorCategory.TECHNICAL),
        "ma_cross_10_60": ("EMA($close, 10) / EMA($close, 60) - 1", FactorCategory.TECHNICAL),

        # 价格加速度
        "price_acceleration": ("Ref($close, 5)/$close - 2*Ref($close, 10)/$close + "
                                "Ref($close, 15)/$close", FactorCategory.MOMENTUM),
    }

    def __init__(self, market: str = "csi300"):
        """
        初始化因子挖掘器

        Args:
            market: 市场 (csi300, csi500, csi800)
        """
        self.market = market
        self.factors: List[AlphaFactor] = []
        self.qlib_initialized = False

    def init_qlib(self, provider_uri: str = "~/.qlib/qlib_data/cn_data",
                  region: str = "cn"):
        """初始化 Qlib"""
        try:
            import qlib
            qlib.init(provider_uri=provider_uri, region=region)
            self.qlib_initialized = True
        except ImportError:
            raise ImportError("请先安装 qlib: pip install pyqlib")

    def add_tv_indicator_factor(self, name: str, indicator_func: Callable,
                                 category: FactorCategory = FactorCategory.TECHNICAL,
                                 description: str = ""):
        """
        将 TV/PyneCore 指标转换为因子

        Args:
            name: 因子名称
            indicator_func: 指标计算函数
            category: 因子类别
            description: 因子描述
        """
        # 注册为自定义因子
        factor = AlphaFactor(
            name=name,
            expression=f"Custom({name})",  # 自定义表达式
            category=category,
            description=description
        )
        self.factors.append(factor)

    def mine_factors(self,
                     stock_pool: str = "CSI300",
                     start_date: str = "2020-01-01",
                     end_date: str = "2024-01-01",
                     categories: Optional[List[FactorCategory]] = None,
                     min_ic: float = 0.02,
                     min_ir: float = 0.3) -> List[AlphaFactor]:
        """
        挖掘有效因子

        Args:
            stock_pool: 股票池
            start_date: 开始日期
            end_date: 结束日期
            categories: 因子类别过滤
            min_ic: 最小 IC
            min_ir: 最小 IR

        Returns:
            有效因子列表
        """
        valid_factors = []

        for name, (expr, category) in self.TECHNICAL_FACTORS.items():
            if categories and category not in categories:
                continue

            # 评估因子
            ic, ir, turnover, returns = self._evaluate_factor(
                expr, start_date, end_date
            )

            if abs(ic) >= min_ic and abs(ir) >= min_ir:
                factor = AlphaFactor(
                    name=name,
                    expression=expr,
                    category=category,
                    ic=ic,
                    ir=ir,
                    turnover=turnover,
                    returns=returns
                )
                valid_factors.append(factor)

        # 按质量评分排序
        valid_factors.sort(key=lambda f: f.quality_score, reverse=True)

        return valid_factors

    def _evaluate_factor(self, expression: str,
                         start_date: str, end_date: str) -> Tuple[float, float, float, float]:
        """
        评估单个因子

        Returns:
            (IC, IR, Turnover, Returns)
        """
        if not self.qlib_initialized:
            # 模拟评估 (用于演示)
            return self._mock_evaluate()

        try:
            from qlib.data import D
            from qlib.contrib.evaluate import risk_analysis

            # 获取因子值
            factor_data = D.features(
                instruments=self.market,
                fields=[expression],
                start_time=start_date,
                end_time=end_date
            )

            # 获取收益率
            returns = D.features(
                instruments=self.market,
                fields=["Ref($close, -1)/$close - 1"],
                start_time=start_date,
                end_time=end_date
            )

            # 计算 IC
            ic = self._calculate_ic(factor_data, returns)

            # 计算 IR
            ir = self._calculate_ir(factor_data, returns)

            # 计算换手率
            turnover = self._calculate_turnover(factor_data)

            # 计算因子收益
            factor_returns = self._calculate_factor_returns(factor_data, returns)

            return ic, ir, turnover, factor_returns

        except Exception as e:
            print(f"因子评估失败: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def _mock_evaluate(self) -> Tuple[float, float, float, float]:
        """模拟因子评估 (演示用)"""
        ic = np.random.uniform(-0.1, 0.1)
        ir = ic * 20 + np.random.uniform(-0.5, 0.5)
        turnover = np.random.uniform(0.1, 0.5)
        returns = np.random.uniform(-0.1, 0.2)
        return ic, ir, turnover, returns

    def _calculate_ic(self, factor: pd.DataFrame,
                      returns: pd.DataFrame) -> float:
        """计算信息系数 (IC)"""
        # 按日期计算秩相关
        ics = []
        for date in factor.index.get_level_values(0).unique():
            f = factor.loc[date].values.flatten()
            r = returns.loc[date].values.flatten()
            # 去除 NaN
            mask = ~(np.isnan(f) | np.isnan(r))
            if mask.sum() > 10:
                ic = np.corrcoef(
                    pd.Series(f[mask]).rank(),
                    pd.Series(r[mask]).rank()
                )[0, 1]
                ics.append(ic)

        return np.mean(ics) if ics else 0.0

    def _calculate_ir(self, factor: pd.DataFrame,
                      returns: pd.DataFrame) -> float:
        """计算信息比率 (IR)"""
        # IR = mean(IC) / std(IC)
        ics = []
        for date in factor.index.get_level_values(0).unique():
            f = factor.loc[date].values.flatten()
            r = returns.loc[date].values.flatten()
            mask = ~(np.isnan(f) | np.isnan(r))
            if mask.sum() > 10:
                ic = np.corrcoef(
                    pd.Series(f[mask]).rank(),
                    pd.Series(r[mask]).rank()
                )[0, 1]
                ics.append(ic)

        if len(ics) < 2:
            return 0.0

        return np.mean(ics) / (np.std(ics) + 1e-8)

    def _calculate_turnover(self, factor: pd.DataFrame) -> float:
        """计算因子换手率"""
        # 简化: 计算因子排名变化
        turnovers = []
        dates = factor.index.get_level_values(0).unique()

        for i in range(1, len(dates)):
            prev = factor.loc[dates[i-1]]
            curr = factor.loc[dates[i]]

            # 共同股票
            common = prev.index.intersection(curr.index)
            if len(common) < 10:
                continue

            prev_rank = prev.loc[common].rank()
            curr_rank = curr.loc[common].rank()

            # 排名变化
            change = (prev_rank - curr_rank).abs().mean() / len(common)
            turnovers.append(change)

        return np.mean(turnovers) if turnovers else 0.5

    def _calculate_factor_returns(self, factor: pd.DataFrame,
                                  returns: pd.DataFrame) -> float:
        """计算因子收益 (多空组合)"""
        # 按因子分组, 计算 Top-Bottom 收益
        total_return = []
        dates = factor.index.get_level_values(0).unique()

        for date in dates:
            f = factor.loc[date]
            r = returns.loc[date]

            common = f.index.intersection(r.index)
            if len(common) < 20:
                continue

            f_common = f.loc[common].values.flatten()
            r_common = r.loc[common].values.flatten()

            # 去除 NaN
            mask = ~(np.isnan(f_common) | np.isnan(r_common))
            f_clean = f_common[mask]
            r_clean = r_common[mask]

            if len(f_clean) < 20:
                continue

            # Top 20% vs Bottom 20%
            n = len(f_clean) // 5
            top_idx = np.argsort(f_clean)[-n:]
            bottom_idx = np.argsort(f_clean)[:n]

            long_return = r_clean[top_idx].mean()
            short_return = r_clean[bottom_idx].mean()

            total_return.append(long_return - short_return)

        if not total_return:
            return 0.0

        # 年化收益
        daily_return = np.mean(total_return)
        annual_return = daily_return * 252

        return annual_return


class FactorCombiner:
    """因子组合器"""

    def __init__(self, factors: List[AlphaFactor]):
        self.factors = factors

    def equal_weight(self) -> Dict[str, float]:
        """等权组合"""
        weight = 1.0 / len(self.factors)
        return {f.name: weight for f in self.factors}

    def ic_weight(self) -> Dict[str, float]:
        """IC 加权"""
        total_ic = sum(abs(f.ic) for f in self.factors)
        return {f.name: abs(f.ic) / total_ic for f in self.factors}

    def ir_weight(self) -> Dict[str, float]:
        """IR 加权"""
        total_ir = sum(abs(f.ir) for f in self.factors)
        return {f.name: abs(f.ir) / total_ir for f in self.factors}

    def optimize_weights(self,
                         correlation_matrix: Optional[np.ndarray] = None,
                         risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        优化因子权重 (均值-方差优化)

        Args:
            correlation_matrix: 因子相关矩阵
            risk_aversion: 风险厌恶系数

        Returns:
            优化后的权重
        """
        n = len(self.factors)

        if correlation_matrix is None:
            # 假设因子不相关
            correlation_matrix = np.eye(n)

        # 因子预期收益 (使用 IC 作为代理)
        expected_returns = np.array([f.ic for f in self.factors])

        # 因子波动率 (使用 1/IR 作为代理)
        volatilities = np.array([1.0 / (abs(f.ir) + 0.1) for f in self.factors])

        # 协方差矩阵
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

        try:
            # 优化: max(w'μ - λ/2 * w'Σw)
            # 解析解: w = (λΣ)^(-1) μ
            inv_cov = np.linalg.inv(cov_matrix + np.eye(n) * 1e-6)
            weights = inv_cov @ expected_returns / risk_aversion

            # 归一化
            weights = np.abs(weights)
            weights = weights / weights.sum()

            return {self.factors[i].name: weights[i] for i in range(n)}

        except np.linalg.LinAlgError:
            # 如果矩阵奇异, 返回等权
            return self.equal_weight()


class RDAgentIntegration:
    """
    RD-Agent 集成 (Qlib 2025 新功能)

    使用 LLM 自动发现和优化因子
    """

    def __init__(self, llm_model: str = "gpt-4"):
        """
        初始化 RD-Agent

        Args:
            llm_model: LLM 模型名称
        """
        self.llm_model = llm_model
        self.discovered_factors: List[AlphaFactor] = []

    def discover_factors(self,
                         market: str = "cn",
                         objective: str = "momentum",
                         max_iterations: int = 100,
                         seed_factors: Optional[List[str]] = None) -> List[AlphaFactor]:
        """
        使用 LLM 自动发现因子

        Args:
            market: 市场
            objective: 目标 (momentum, reversal, value, etc.)
            max_iterations: 最大迭代次数
            seed_factors: 种子因子表达式

        Returns:
            发现的因子列表
        """
        try:
            from qlib.contrib.rd_agent import RDAgent

            agent = RDAgent(llm_model=self.llm_model)

            # 运行因子发现
            results = agent.run(
                task="factor_discovery",
                market=market,
                objective=objective,
                max_iterations=max_iterations,
                seed_expressions=seed_factors
            )

            # 转换为 AlphaFactor
            for r in results:
                factor = AlphaFactor(
                    name=r["name"],
                    expression=r["expression"],
                    category=FactorCategory.TECHNICAL,
                    description=r.get("description", ""),
                    ic=r.get("ic", 0),
                    ir=r.get("ir", 0)
                )
                self.discovered_factors.append(factor)

            return self.discovered_factors

        except ImportError:
            print("RD-Agent 需要额外安装: pip install rd-agent")
            return []

    def optimize_factor(self, factor: AlphaFactor,
                        optimization_target: str = "ic") -> AlphaFactor:
        """
        使用 LLM 优化因子表达式

        Args:
            factor: 原始因子
            optimization_target: 优化目标 (ic, ir, returns)

        Returns:
            优化后的因子
        """
        try:
            from qlib.contrib.rd_agent import RDAgent

            agent = RDAgent(llm_model=self.llm_model)

            result = agent.optimize(
                expression=factor.expression,
                target=optimization_target
            )

            return AlphaFactor(
                name=f"{factor.name}_optimized",
                expression=result["expression"],
                category=factor.category,
                description=f"Optimized from {factor.name}",
                ic=result.get("ic", 0),
                ir=result.get("ir", 0)
            )

        except ImportError:
            print("RD-Agent 需要额外安装: pip install rd-agent")
            return factor
