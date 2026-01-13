"""
ROMA Strategy Optimizer Agent

策略优化代理，用于自动化参数优化和策略改进。
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import itertools

# 添加 TV2PY 路径
TV2PY_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TV2PY_PATH))

from ..orchestrator import BaseAgent
from ..core.atomizer import AtomicTask, TaskType
from ..core.executor import PlanExecutionResult


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    param_type: str = "float"  # float, int, bool

    def get_values(self) -> List[Any]:
        """获取参数值列表"""
        if self.param_type == "bool":
            return [True, False]
        elif self.param_type == "int":
            return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
        else:
            values = []
            v = self.min_value
            while v <= self.max_value:
                values.append(v)
                v += self.step
            return values


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: str = "grid_search"  # grid_search, random_search, bayesian
    metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, max_drawdown, win_rate
    maximize: bool = True
    max_iterations: int = 100
    early_stopping: int = 20
    parameter_ranges: List[ParameterRange] = field(default_factory=list)


class StrategyOptimizerAgent(BaseAgent):
    """
    策略优化代理

    自动化参数优化和策略改进。

    优化方法:
        - grid_search: 网格搜索 (穷举)
        - random_search: 随机搜索
        - bayesian: 贝叶斯优化 (需要额外依赖)

    优化指标:
        - sharpe_ratio: 夏普比率
        - total_return: 总收益率
        - max_drawdown: 最大回撤
        - win_rate: 胜率
        - profit_factor: 盈亏比

    使用示例:
        agent = StrategyOptimizerAgent(
            method="grid_search",
            metric="sharpe_ratio",
        )

        agent.add_parameter("sma_period", 10, 50, step=5)
        agent.add_parameter("rsi_threshold", 20, 40, step=5)

        result = orchestrator.execute_with_agent(
            agent=agent,
            task_description="优化策略参数",
            context={"strategy": my_strategy}
        )
    """

    def __init__(
        self,
        method: str = "grid_search",
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        max_iterations: int = 100,
    ):
        super().__init__(name="StrategyOptimizerAgent")

        self.config = OptimizationConfig(
            method=method,
            metric=metric,
            maximize=maximize,
            max_iterations=max_iterations,
        )

    def add_parameter(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step: float = 1.0,
        param_type: str = "float"
    ) -> "StrategyOptimizerAgent":
        """添加要优化的参数"""
        self.config.parameter_ranges.append(ParameterRange(
            name=name,
            min_value=min_value,
            max_value=max_value,
            step=step,
            param_type=param_type,
        ))
        return self

    def prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """准备优化上下文"""
        prepared = context.copy()

        prepared["optimization"] = {
            "method": self.config.method,
            "metric": self.config.metric,
            "maximize": self.config.maximize,
            "max_iterations": self.config.max_iterations,
            "parameters": [
                {
                    "name": p.name,
                    "range": (p.min_value, p.max_value),
                    "step": p.step,
                }
                for p in self.config.parameter_ranges
            ],
        }

        return prepared

    def customize_tasks(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Optional[List[AtomicTask]]:
        """自定义优化任务"""
        tasks = []
        task_counter = 0

        def make_id(prefix: str) -> str:
            nonlocal task_counter
            task_counter += 1
            return f"{prefix}_{task_counter:04d}"

        # 数据获取
        data_task = AtomicTask(
            id=make_id("data_fetch"),
            task_type=TaskType.DATA_FETCH,
            name="Fetch Data for Optimization",
            description="获取优化所需的历史数据",
            inputs={
                "symbol": context.get("symbol", "ETH-USDT"),
                "timeframe": context.get("timeframe", "4h"),
                "limit": context.get("limit", 500),
            },
            expected_outputs=["candles_df"],
            priority=100,
        )
        tasks.append(data_task)

        # 生成参数组合
        param_combinations = self._generate_param_combinations()

        # 为每个参数组合创建回测任务
        backtest_tasks = []
        for i, params in enumerate(param_combinations[:self.config.max_iterations]):
            bt_task = AtomicTask(
                id=make_id("strategy_bt"),
                task_type=TaskType.STRATEGY_BACKTEST,
                name=f"Backtest #{i+1}",
                description=f"回测参数组合 {params}",
                inputs={
                    "params": params,
                    "initial_capital": context.get("initial_capital", 10000),
                },
                dependencies=[data_task.id],
                expected_outputs=["backtest_results"],
                priority=50,
                metadata={"params": params, "iteration": i},
            )
            backtest_tasks.append(bt_task)
            tasks.append(bt_task)

        # 汇总优化结果
        opt_task = AtomicTask(
            id=make_id("strategy_opt"),
            task_type=TaskType.STRATEGY_OPTIMIZE,
            name="Aggregate Optimization Results",
            description="汇总优化结果，找出最佳参数",
            inputs={
                "metric": self.config.metric,
                "maximize": self.config.maximize,
            },
            dependencies=[t.id for t in backtest_tasks],
            expected_outputs=["optimal_params", "optimization_report"],
            priority=30,
        )
        tasks.append(opt_task)

        return tasks

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """生成参数组合"""
        if not self.config.parameter_ranges:
            return [{}]

        if self.config.method == "grid_search":
            return self._grid_search_combinations()
        elif self.config.method == "random_search":
            return self._random_search_combinations()
        else:
            return self._grid_search_combinations()

    def _grid_search_combinations(self) -> List[Dict[str, Any]]:
        """网格搜索参数组合"""
        param_values = {
            p.name: p.get_values()
            for p in self.config.parameter_ranges
        }

        keys = list(param_values.keys())
        values = list(param_values.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _random_search_combinations(self) -> List[Dict[str, Any]]:
        """随机搜索参数组合"""
        import random

        combinations = []
        for _ in range(self.config.max_iterations):
            params = {}
            for p in self.config.parameter_ranges:
                if p.param_type == "bool":
                    params[p.name] = random.choice([True, False])
                elif p.param_type == "int":
                    params[p.name] = random.randint(int(p.min_value), int(p.max_value))
                else:
                    params[p.name] = random.uniform(p.min_value, p.max_value)
            combinations.append(params)

        return combinations

    def process_results(
        self,
        execution_result: PlanExecutionResult
    ) -> Dict[str, Any]:
        """处理优化结果"""
        output = {
            "agent": self.name,
            "method": self.config.method,
            "metric": self.config.metric,
            "success_rate": execution_result.success_rate,
        }

        # 收集所有回测结果
        backtest_results = []

        for task_id, result in execution_result.results.items():
            if not result.success:
                continue

            if result.metadata.get("params"):
                backtest_results.append({
                    "params": result.metadata["params"],
                    "iteration": result.metadata.get("iteration", 0),
                    "results": result.output.get("backtest_results", {}),
                })

        # 找出最佳参数
        if backtest_results:
            best = self._find_best_params(backtest_results)
            output["best_params"] = best["params"]
            output["best_metrics"] = best["metrics"]
            output["total_iterations"] = len(backtest_results)

            # 生成优化报告
            output["optimization_report"] = self._generate_report(backtest_results, best)
        else:
            output["best_params"] = {}
            output["error"] = "No successful backtests completed"

        return output

    def _find_best_params(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """找出最佳参数"""
        metric = self.config.metric
        maximize = self.config.maximize

        best = None
        best_value = float("-inf") if maximize else float("inf")

        for r in results:
            metrics = r.get("results", {})
            value = metrics.get(metric, 0)

            if maximize:
                if value > best_value:
                    best_value = value
                    best = {
                        "params": r["params"],
                        "metrics": metrics,
                        "iteration": r["iteration"],
                    }
            else:
                if value < best_value:
                    best_value = value
                    best = {
                        "params": r["params"],
                        "metrics": metrics,
                        "iteration": r["iteration"],
                    }

        return best or {"params": {}, "metrics": {}}

    def _generate_report(
        self,
        results: List[Dict[str, Any]],
        best: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成优化报告"""
        metric = self.config.metric

        # 收集指标值
        values = [r["results"].get(metric, 0) for r in results]

        import numpy as np

        report = {
            "total_iterations": len(results),
            "metric_stats": {
                "mean": np.mean(values) if values else 0,
                "std": np.std(values) if values else 0,
                "min": np.min(values) if values else 0,
                "max": np.max(values) if values else 0,
            },
            "best_iteration": best.get("iteration", 0),
            "improvement_ratio": (
                (np.max(values) - np.mean(values)) / np.mean(values)
                if values and np.mean(values) != 0 else 0
            ),
        }

        return report


class QuickOptimizer(StrategyOptimizerAgent):
    """
    快速优化器

    使用较大步长进行快速参数扫描。
    """

    def __init__(self, metric: str = "sharpe_ratio"):
        super().__init__(
            method="random_search",
            metric=metric,
            max_iterations=20,
        )
        self.name = "QuickOptimizer"


class ThoroughOptimizer(StrategyOptimizerAgent):
    """
    全面优化器

    使用精细步长进行全面参数优化。
    """

    def __init__(self, metric: str = "sharpe_ratio"):
        super().__init__(
            method="grid_search",
            metric=metric,
            max_iterations=200,
        )
        self.name = "ThoroughOptimizer"
