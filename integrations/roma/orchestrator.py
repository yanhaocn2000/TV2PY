"""
ROMA Orchestrator - 编排器

编排 ROMA 组件协同工作，提供统一的执行接口。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

from .core.atomizer import Atomizer, AtomicTask, TradingAtomizer
from .core.planner import Planner, ExecutionPlan, TradingPlanner
from .core.executor import Executor, ExecutionResult, PlanExecutionResult, TradingTaskExecutor
from .core.aggregator import Aggregator, AggregatedResult, TradingAggregator
from .core.verifier import Verifier, VerificationResult, TradingVerifier


@dataclass
class OrchestrationResult:
    """
    编排结果

    包含完整的执行链路信息。
    """
    orchestration_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # 各阶段结果
    tasks: List[AtomicTask] = field(default_factory=list)
    plan: Optional[ExecutionPlan] = None
    execution_result: Optional[PlanExecutionResult] = None
    aggregated_result: Optional[AggregatedResult] = None
    verification_result: Optional[VerificationResult] = None

    # 最终输出
    final_output: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orchestration_id": self.orchestration_id,
            "status": self.status,
            "duration": self.duration,
            "task_count": len(self.tasks),
            "plan": self.plan.to_dict() if self.plan else None,
            "execution_success_rate": self.execution_result.success_rate if self.execution_result else 0,
            "verification_score": self.verification_result.score if self.verification_result else 0,
            "final_output": self.final_output,
        }


class ROMAOrchestrator:
    """
    ROMA 编排器

    协调 Atomizer, Planner, Executor, Aggregator, Verifier 组件工作。

    使用示例:
        orchestrator = ROMAOrchestrator()

        result = orchestrator.execute(
            task_description="分析 ETH-USDT 并生成交易信号",
            context={
                "symbol": "ETH-USDT",
                "timeframe": "4h",
                "indicators": ["smc", "squeeze", "supertrend"],
            }
        )

        print(result.final_output)
    """

    def __init__(
        self,
        atomizer: Optional[Atomizer] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        aggregator: Optional[Aggregator] = None,
        verifier: Optional[Verifier] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化编排器

        Args:
            atomizer: 原子化器 (默认 TradingAtomizer)
            planner: 规划器 (默认 TradingPlanner)
            executor: 执行器 (默认 Executor)
            aggregator: 聚合器 (默认 TradingAggregator)
            verifier: 验证器 (默认 TradingVerifier)
            logger: 日志记录器
        """
        self.atomizer = atomizer or TradingAtomizer()
        self.planner = planner or TradingPlanner()
        self.executor = executor or Executor()
        self.aggregator = aggregator or TradingAggregator()
        self.verifier = verifier or TradingVerifier()
        self.logger = logger or logging.getLogger(__name__)

        self._orchestration_counter = 0

        # 注册默认执行器
        self._register_default_executors()

    def _register_default_executors(self):
        """注册默认任务执行器"""
        from .core.atomizer import TaskType
        trading_executor = TradingTaskExecutor()

        for task_type in TaskType:
            self.executor.register_executor(task_type, trading_executor)

    def execute(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        skip_verification: bool = False,
        on_progress: Optional[Callable[[str, Any], None]] = None,
    ) -> OrchestrationResult:
        """
        执行任务

        完整的 ROMA 工作流:
        1. Atomize - 分解任务
        2. Plan - 规划执行
        3. Execute - 执行任务
        4. Aggregate - 聚合结果
        5. Verify - 验证结果

        Args:
            task_description: 任务描述
            context: 执行上下文
            skip_verification: 是否跳过验证
            on_progress: 进度回调

        Returns:
            OrchestrationResult: 编排结果
        """
        self._orchestration_counter += 1
        orch_id = f"orch_{self._orchestration_counter:04d}"
        context = context or {}

        result = OrchestrationResult(
            orchestration_id=orch_id,
            status="running",
            start_time=datetime.now(),
        )

        try:
            # 1. Atomize - 分解任务
            self._report_progress(on_progress, "atomizing", {"task": task_description})
            tasks = self.atomizer.atomize(task_description, context)
            result.tasks = tasks
            self.logger.info(f"Atomized into {len(tasks)} tasks")

            if not tasks:
                result.status = "completed"
                result.final_output = {"message": "No tasks to execute"}
                result.end_time = datetime.now()
                return result

            # 2. Plan - 规划执行
            self._report_progress(on_progress, "planning", {"task_count": len(tasks)})
            plan = self.planner.create_plan(tasks, context)
            result.plan = plan
            self.logger.info(f"Created plan with {plan.total_steps} steps")

            # 3. Execute - 执行任务
            self._report_progress(on_progress, "executing", {"plan": plan.plan_id})
            execution_result = self.executor.execute_plan(
                plan,
                context,
                on_step_complete=lambda step_id, results: self._report_progress(
                    on_progress, "step_complete", {"step": step_id, "results": len(results)}
                ),
            )
            result.execution_result = execution_result
            self.logger.info(f"Execution completed: {execution_result.success_rate:.1%} success rate")

            # 4. Aggregate - 聚合结果
            self._report_progress(on_progress, "aggregating", {})
            successful_results = [r for r in execution_result.results.values() if r.success]
            if successful_results:
                aggregated = self.aggregator.aggregate(successful_results)
                result.aggregated_result = aggregated
            else:
                result.aggregated_result = AggregatedResult(
                    aggregation_id="agg_empty",
                    method=None,
                    value={},
                    confidence=0.0,
                    source_count=0,
                )

            # 5. Verify - 验证结果
            if not skip_verification:
                self._report_progress(on_progress, "verifying", {})
                verification = self.verifier.verify(execution_result)
                result.verification_result = verification
                self.logger.info(f"Verification: {verification.status.value} (score: {verification.score:.2f})")

            # 生成最终输出
            result.final_output = self._generate_final_output(result)
            result.status = "completed"

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            result.status = "failed"
            result.final_output = {"error": str(e)}
            result.metadata["exception"] = str(e)

        result.end_time = datetime.now()
        self._report_progress(on_progress, "completed", {"duration": result.duration})

        return result

    def _report_progress(
        self,
        callback: Optional[Callable],
        stage: str,
        data: Any
    ):
        """报告进度"""
        if callback:
            callback(stage, data)

    def _generate_final_output(self, result: OrchestrationResult) -> Dict[str, Any]:
        """生成最终输出"""
        output = {
            "orchestration_id": result.orchestration_id,
            "task_count": len(result.tasks),
        }

        # 添加执行摘要
        if result.execution_result:
            output["execution_summary"] = {
                "success_rate": result.execution_result.success_rate,
                "total_duration": result.execution_result.total_duration,
                "success_count": result.execution_result.success_count,
                "failure_count": result.execution_result.failure_count,
            }

        # 添加聚合结果
        if result.aggregated_result and result.aggregated_result.value:
            output["aggregated"] = result.aggregated_result.value
            output["confidence"] = result.aggregated_result.confidence

        # 添加验证结果
        if result.verification_result:
            output["verification"] = {
                "status": result.verification_result.status.value,
                "score": result.verification_result.score,
                "warnings": result.verification_result.warnings,
            }

        return output

    def execute_with_agent(
        self,
        agent: "BaseAgent",
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        使用代理执行任务

        代理可以提供自定义的任务分解和执行策略。

        Args:
            agent: 代理实例
            task_description: 任务描述
            context: 执行上下文

        Returns:
            OrchestrationResult
        """
        # 让代理准备上下文
        prepared_context = agent.prepare_context(context or {})

        # 让代理自定义任务分解
        custom_tasks = agent.customize_tasks(task_description, prepared_context)

        if custom_tasks:
            # 使用代理提供的任务
            self._orchestration_counter += 1
            orch_id = f"orch_{self._orchestration_counter:04d}"

            result = OrchestrationResult(
                orchestration_id=orch_id,
                status="running",
                start_time=datetime.now(),
            )
            result.tasks = custom_tasks

            # 规划和执行
            plan = self.planner.create_plan(custom_tasks, prepared_context)
            result.plan = plan

            execution_result = self.executor.execute_plan(plan, prepared_context)
            result.execution_result = execution_result

            # 让代理处理结果
            final_output = agent.process_results(execution_result)
            result.final_output = final_output
            result.status = "completed"
            result.end_time = datetime.now()

            return result
        else:
            # 使用标准流程
            return self.execute(task_description, prepared_context)


class BaseAgent:
    """
    代理基类

    代理可以自定义 ROMA 工作流的各个阶段。
    """

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    def prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """准备执行上下文"""
        return context

    def customize_tasks(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Optional[List[AtomicTask]]:
        """
        自定义任务列表

        返回 None 使用标准原子化流程。
        """
        return None

    def process_results(
        self,
        execution_result: PlanExecutionResult
    ) -> Dict[str, Any]:
        """处理执行结果"""
        return {
            "agent": self.name,
            "success_rate": execution_result.success_rate,
            "results": {
                k: v.output for k, v in execution_result.results.items()
            }
        }
