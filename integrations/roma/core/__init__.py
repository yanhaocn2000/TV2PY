"""
ROMA Core Components

核心组件模块，包含 ROMA 框架的五个主要组件。
"""

from .atomizer import Atomizer, AtomicTask, TaskType, TradingAtomizer
from .planner import Planner, ExecutionPlan, PlanStep, TradingPlanner
from .executor import Executor, ExecutionResult, ExecutionStatus, TradingTaskExecutor, PlanExecutionResult
from .aggregator import Aggregator, AggregatedResult, TradingAggregator, AggregationMethod
from .verifier import Verifier, VerificationResult, VerificationStatus, TradingVerifier

__all__ = [
    "Atomizer", "AtomicTask", "TaskType", "TradingAtomizer",
    "Planner", "ExecutionPlan", "PlanStep", "TradingPlanner",
    "Executor", "ExecutionResult", "ExecutionStatus", "TradingTaskExecutor", "PlanExecutionResult",
    "Aggregator", "AggregatedResult", "TradingAggregator", "AggregationMethod",
    "Verifier", "VerificationResult", "VerificationStatus", "TradingVerifier",
]
