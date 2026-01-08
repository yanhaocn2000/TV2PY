"""
TV2PY ROMA Integration

ROMA (Recursive Open Meta-Agents) 框架集成

ROMA 是一个用于构建递归智能代理系统的框架，本模块将 ROMA 与 TV2PY 结合，
提供自动化的 TradingView 指标分析、转换和策略优化能力。

主要组件:
    - Atomizer: 将复杂任务分解为原子操作
    - Planner: 规划任务执行顺序
    - Executor: 执行原子任务
    - Aggregator: 聚合执行结果
    - Verifier: 验证输出正确性

交易特化 Agents:
    - IndicatorAnalyzer: 分析 Pine Script 指标
    - StrategyOptimizer: 优化交易策略参数
    - BacktestAgent: 自动化回测
    - SignalAggregator: 多指标信号聚合

使用示例:
    from integrations.roma import ROMAOrchestrator, TradingMetaAgent

    # 创建编排器
    orchestrator = ROMAOrchestrator()

    # 创建交易元代理
    agent = TradingMetaAgent(
        indicators=["smc", "squeeze", "supertrend"],
        timeframe="4h"
    )

    # 执行任务
    result = orchestrator.execute(
        agent=agent,
        task="Analyze ETH-USDT and generate trading signals"
    )
"""

from .core.atomizer import Atomizer, AtomicTask
from .core.planner import Planner, ExecutionPlan
from .core.executor import Executor, ExecutionResult
from .core.aggregator import Aggregator
from .core.verifier import Verifier, VerificationResult

from .orchestrator import ROMAOrchestrator

from .agents.trading_agent import TradingMetaAgent
from .agents.indicator_analyzer import IndicatorAnalyzerAgent
from .agents.strategy_optimizer import StrategyOptimizerAgent

__all__ = [
    # Core components
    "Atomizer",
    "AtomicTask",
    "Planner",
    "ExecutionPlan",
    "Executor",
    "ExecutionResult",
    "Aggregator",
    "Verifier",
    "VerificationResult",
    # Orchestrator
    "ROMAOrchestrator",
    # Trading Agents
    "TradingMetaAgent",
    "IndicatorAnalyzerAgent",
    "StrategyOptimizerAgent",
]

__version__ = "1.0.0"
