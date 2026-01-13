"""
ROMA Executor - 任务执行组件

负责执行原子任务，支持:
    - 同步/异步执行
    - 错误处理和重试
    - 执行状态跟踪
    - 结果缓存
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

from .atomizer import AtomicTask, TaskType
from .planner import ExecutionPlan, PlanStep


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"         # 等待执行
    RUNNING = "running"         # 执行中
    COMPLETED = "completed"     # 完成
    FAILED = "failed"           # 失败
    TIMEOUT = "timeout"         # 超时
    SKIPPED = "skipped"         # 跳过


@dataclass
class ExecutionResult:
    """
    任务执行结果
    """
    task_id: str
    status: ExecutionStatus
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """执行时长(秒)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED


@dataclass
class PlanExecutionResult:
    """
    计划执行结果
    """
    plan_id: str
    status: ExecutionStatus
    results: Dict[str, ExecutionResult] = field(default_factory=dict)
    total_duration: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class TaskExecutorBase(ABC):
    """任务执行器基类"""

    @abstractmethod
    def execute(
        self,
        task: AtomicTask,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """执行单个任务"""
        pass


class Executor:
    """
    执行器

    负责执行计划中的所有任务。
    """

    def __init__(
        self,
        max_workers: int = 4,
        enable_cache: bool = True,
    ):
        """
        Args:
            max_workers: 最大并行工作线程数
            enable_cache: 是否启用结果缓存
        """
        self.max_workers = max_workers
        self.enable_cache = enable_cache

        # 任务执行器注册表
        self._task_executors: Dict[TaskType, TaskExecutorBase] = {}

        # 结果缓存
        self._result_cache: Dict[str, ExecutionResult] = {}

        # 执行上下文 (任务间共享数据)
        self._shared_context: Dict[str, Any] = {}

    def register_executor(
        self,
        task_type: TaskType,
        executor: TaskExecutorBase
    ):
        """注册任务执行器"""
        self._task_executors[task_type] = executor

    def execute_plan(
        self,
        plan: ExecutionPlan,
        context: Dict[str, Any] = None,
        on_step_complete: Callable[[int, List[ExecutionResult]], None] = None,
        on_task_complete: Callable[[ExecutionResult], None] = None,
    ) -> PlanExecutionResult:
        """
        执行完整计划

        Args:
            plan: 执行计划
            context: 初始上下文
            on_step_complete: 步骤完成回调
            on_task_complete: 任务完成回调

        Returns:
            PlanExecutionResult: 计划执行结果
        """
        # 初始化共享上下文
        self._shared_context = context.copy() if context else {}

        results = {}
        success_count = 0
        failure_count = 0
        start_time = time.time()

        # 按步骤执行
        for step in plan.steps:
            step_results = self._execute_step(
                step,
                on_task_complete=on_task_complete
            )

            for result in step_results:
                results[result.task_id] = result
                if result.success:
                    success_count += 1
                else:
                    failure_count += 1

            if on_step_complete:
                on_step_complete(step.step_id, step_results)

            # 如果有任务失败，检查是否需要终止
            failed_tasks = [r for r in step_results if not r.success]
            if failed_tasks:
                # 这里可以添加更复杂的失败处理逻辑
                pass

        total_duration = time.time() - start_time

        # 确定整体状态
        if failure_count == 0:
            status = ExecutionStatus.COMPLETED
        elif success_count == 0:
            status = ExecutionStatus.FAILED
        else:
            status = ExecutionStatus.COMPLETED  # 部分成功也算完成

        return PlanExecutionResult(
            plan_id=plan.plan_id,
            status=status,
            results=results,
            total_duration=total_duration,
            success_count=success_count,
            failure_count=failure_count,
            metadata={
                "context": self._shared_context,
            }
        )

    def _execute_step(
        self,
        step: PlanStep,
        on_task_complete: Callable[[ExecutionResult], None] = None,
    ) -> List[ExecutionResult]:
        """执行单个步骤中的所有任务"""
        results = []

        if len(step.tasks) == 1:
            # 单任务顺序执行
            result = self._execute_task(step.tasks[0])
            results.append(result)
            if on_task_complete:
                on_task_complete(result)
        else:
            # 多任务并行执行
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._execute_task, task): task
                    for task in step.tasks
                }

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = ExecutionResult(
                            task_id=task.id,
                            status=ExecutionStatus.FAILED,
                            error=str(e),
                        )
                    results.append(result)
                    if on_task_complete:
                        on_task_complete(result)

        return results

    def _execute_task(self, task: AtomicTask) -> ExecutionResult:
        """执行单个任务"""
        # 检查缓存
        if self.enable_cache and task.id in self._result_cache:
            return self._result_cache[task.id]

        start_time = datetime.now()
        retry_count = 0

        while retry_count <= task.retry_count:
            try:
                # 准备执行上下文
                exec_context = self._prepare_context(task)

                # 获取执行器
                executor = self._task_executors.get(task.task_type)

                if executor:
                    # 使用注册的执行器
                    result = executor.execute(task, exec_context)
                else:
                    # 使用默认执行
                    result = self._default_execute(task, exec_context)

                result.start_time = start_time
                result.end_time = datetime.now()
                result.retry_count = retry_count

                # 更新共享上下文
                if result.success:
                    self._update_shared_context(task, result)

                    # 缓存结果
                    if self.enable_cache:
                        self._result_cache[task.id] = result

                return result

            except Exception as e:
                retry_count += 1
                if retry_count > task.retry_count:
                    return ExecutionResult(
                        task_id=task.id,
                        status=ExecutionStatus.FAILED,
                        error=f"{str(e)}\n{traceback.format_exc()}",
                        start_time=start_time,
                        end_time=datetime.now(),
                        retry_count=retry_count - 1,
                    )
                # 等待后重试
                time.sleep(0.5 * retry_count)

        # 不应该到达这里
        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.FAILED,
            error="Unexpected execution path",
        )

    def _prepare_context(self, task: AtomicTask) -> Dict[str, Any]:
        """准备任务执行上下文"""
        context = self._shared_context.copy()

        # 解析输入引用
        resolved_inputs = {}
        for key, value in task.inputs.items():
            if isinstance(value, str) and value.endswith("_ref"):
                # 引用其他任务的输出
                ref_task_id = value.replace("_ref", "").replace("_", "_")
                if ref_task_id in self._result_cache:
                    resolved_inputs[key] = self._result_cache[ref_task_id].output
                else:
                    resolved_inputs[key] = value
            else:
                resolved_inputs[key] = value

        context["inputs"] = resolved_inputs
        context["params"] = task.params
        return context

    def _update_shared_context(self, task: AtomicTask, result: ExecutionResult):
        """更新共享上下文"""
        # 将任务输出添加到共享上下文
        for output_name in task.expected_outputs:
            if output_name in result.output:
                self._shared_context[f"{task.id}_{output_name}"] = result.output[output_name]

    def _default_execute(
        self,
        task: AtomicTask,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """默认执行逻辑"""
        # 返回一个模拟成功的结果
        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.COMPLETED,
            output={name: None for name in task.expected_outputs},
            metadata={"default_execution": True},
        )

    def clear_cache(self):
        """清除结果缓存"""
        self._result_cache.clear()

    def get_cached_result(self, task_id: str) -> Optional[ExecutionResult]:
        """获取缓存的结果"""
        return self._result_cache.get(task_id)


class TradingTaskExecutor(TaskExecutorBase):
    """
    交易任务执行器

    执行交易相关的原子任务。
    """

    def __init__(self, tv2py_indicators=None):
        """
        Args:
            tv2py_indicators: TV2PY 指标实例字典
        """
        self.indicators = tv2py_indicators or {}

    def execute(
        self,
        task: AtomicTask,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """执行交易任务"""
        inputs = context.get("inputs", {})
        params = context.get("params", {})

        try:
            if task.task_type == TaskType.DATA_FETCH:
                return self._execute_data_fetch(task, inputs, params)
            elif task.task_type == TaskType.DATA_VALIDATE:
                return self._execute_data_validate(task, inputs, context)
            elif task.task_type == TaskType.INDICATOR_CALCULATE:
                return self._execute_indicator_calc(task, inputs, context)
            elif task.task_type == TaskType.SIGNAL_GENERATE:
                return self._execute_signal_gen(task, inputs, context)
            elif task.task_type == TaskType.STRATEGY_BACKTEST:
                return self._execute_backtest(task, inputs, context)
            else:
                return ExecutionResult(
                    task_id=task.id,
                    status=ExecutionStatus.COMPLETED,
                    output={"message": f"Task type {task.task_type} executed with default handler"},
                )

        except Exception as e:
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.FAILED,
                error=str(e),
            )

    def _execute_data_fetch(
        self,
        task: AtomicTask,
        inputs: Dict,
        params: Dict
    ) -> ExecutionResult:
        """执行数据获取"""
        import pandas as pd
        import numpy as np

        symbol = inputs.get("symbol", "ETH-USDT")
        timeframe = inputs.get("timeframe", "4h")
        limit = inputs.get("limit", 200)

        # 生成模拟数据 (实际使用时替换为真实数据源)
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq=timeframe)

        base = 3000
        trend = np.cumsum(np.random.randn(limit) * 20)
        close = base + trend
        high = close + np.abs(np.random.randn(limit)) * 30
        low = close - np.abs(np.random.randn(limit)) * 30
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = 1000000 + np.abs(np.random.randn(limit)) * 500000

        df = pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)

        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.COMPLETED,
            output={
                "candles_df": df,
                "symbol": symbol,
                "timeframe": timeframe,
                "records": len(df),
            },
        )

    def _execute_data_validate(
        self,
        task: AtomicTask,
        inputs: Dict,
        context: Dict
    ) -> ExecutionResult:
        """执行数据验证"""
        # 从上下文获取数据
        data_ref = inputs.get("data_ref", "")
        df = context.get(f"{data_ref}_candles_df") if data_ref else inputs.get("candles_df")

        if df is None:
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.FAILED,
                error="No data found to validate",
            )

        # 基本验证
        quality_report = {
            "total_rows": len(df),
            "null_count": df.isnull().sum().sum(),
            "has_required_columns": all(col in df.columns for col in ["open", "high", "low", "close", "volume"]),
            "date_range": f"{df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "N/A",
        }

        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.COMPLETED,
            output={
                "validated_df": df,
                "data_quality_report": quality_report,
            },
        )

    def _execute_indicator_calc(
        self,
        task: AtomicTask,
        inputs: Dict,
        context: Dict
    ) -> ExecutionResult:
        """执行指标计算"""
        indicator_name = inputs.get("indicator_name", "")
        params = inputs.get("params", {})

        # 获取数据
        df = None
        for key, value in context.items():
            if "candles_df" in key or "validated_df" in key:
                df = value
                break

        if df is None:
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.FAILED,
                error="No candle data available for indicator calculation",
            )

        # 计算指标
        try:
            result = self._calculate_indicator(indicator_name, df, params)
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.COMPLETED,
                output={f"{indicator_name}_result": result},
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.id,
                status=ExecutionStatus.FAILED,
                error=f"Indicator calculation failed: {e}",
            )

    def _calculate_indicator(
        self,
        name: str,
        df,
        params: Dict
    ) -> Dict[str, Any]:
        """计算具体指标"""
        # 这里集成 TV2PY 指标
        result = {
            "indicator": name,
            "params": params,
            "calculated": True,
            "last_value": df["close"].iloc[-1] if "close" in df.columns else None,
        }
        return result

    def _execute_signal_gen(
        self,
        task: AtomicTask,
        inputs: Dict,
        context: Dict
    ) -> ExecutionResult:
        """执行信号生成"""
        strategy_type = inputs.get("strategy_type", "smart_money")
        min_confidence = inputs.get("min_confidence", 0.6)

        # 生成模拟信号
        signal = {
            "type": "HOLD",
            "strength": 0.5,
            "confidence": 0.65,
            "reason": f"Generated by {strategy_type} strategy",
        }

        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.COMPLETED,
            output={"raw_signals": [signal]},
        )

    def _execute_backtest(
        self,
        task: AtomicTask,
        inputs: Dict,
        context: Dict
    ) -> ExecutionResult:
        """执行回测"""
        initial_capital = inputs.get("initial_capital", 10000)

        # 生成模拟回测结果
        results = {
            "initial_capital": initial_capital,
            "final_capital": initial_capital * 1.25,
            "total_return": 0.25,
            "max_drawdown": 0.15,
            "sharpe_ratio": 1.5,
            "total_trades": 20,
            "win_rate": 0.6,
        }

        return ExecutionResult(
            task_id=task.id,
            status=ExecutionStatus.COMPLETED,
            output={
                "backtest_results": results,
                "trades": [],
                "metrics": results,
            },
        )
