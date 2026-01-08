"""
ROMA Planner - 执行计划组件

基于原子任务列表生成优化的执行计划。
支持:
    - 拓扑排序确保依赖顺序
    - 并行执行优化
    - 资源约束考虑
    - 动态重规划
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from collections import defaultdict
import heapq

from .atomizer import AtomicTask, TaskType


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"   # 顺序执行
    PARALLEL = "parallel"       # 并行执行
    HYBRID = "hybrid"           # 混合执行


@dataclass
class PlanStep:
    """
    执行计划步骤

    每个步骤可以包含多个可并行执行的任务。
    """
    step_id: int
    tasks: List[AtomicTask]
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    estimated_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_ids(self) -> List[str]:
        return [t.id for t in self.tasks]

    def __repr__(self):
        task_names = [t.name for t in self.tasks]
        return f"Step {self.step_id}: {task_names} ({self.execution_mode.value})"


@dataclass
class ExecutionPlan:
    """
    执行计划

    包含完整的任务执行顺序和优化信息。
    """
    plan_id: str
    steps: List[PlanStep]
    total_tasks: int
    estimated_duration: float
    parallel_factor: float  # 并行化程度 (1.0 = 完全顺序)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_task_order(self) -> List[str]:
        """获取任务执行顺序"""
        order = []
        for step in self.steps:
            order.extend(step.task_ids)
        return order

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "plan_id": self.plan_id,
            "total_steps": self.total_steps,
            "total_tasks": self.total_tasks,
            "estimated_duration": self.estimated_duration,
            "parallel_factor": self.parallel_factor,
            "steps": [
                {
                    "step_id": step.step_id,
                    "tasks": [t.name for t in step.tasks],
                    "mode": step.execution_mode.value,
                }
                for step in self.steps
            ],
        }


class Planner:
    """
    执行计划器

    负责将原子任务列表转换为优化的执行计划。
    """

    def __init__(
        self,
        max_parallel: int = 4,
        prefer_parallel: bool = True,
    ):
        """
        Args:
            max_parallel: 最大并行任务数
            prefer_parallel: 是否优先并行执行
        """
        self.max_parallel = max_parallel
        self.prefer_parallel = prefer_parallel
        self._plan_counter = 0

    def create_plan(
        self,
        tasks: List[AtomicTask],
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        创建执行计划

        Args:
            tasks: 原子任务列表
            context: 上下文信息

        Returns:
            ExecutionPlan: 优化后的执行计划
        """
        if not tasks:
            return self._create_empty_plan()

        # 1. 构建依赖图
        dep_graph = self._build_dependency_graph(tasks)

        # 2. 拓扑排序
        sorted_tasks = self._topological_sort(tasks, dep_graph)

        # 3. 分组并行任务
        steps = self._group_parallel_tasks(sorted_tasks, dep_graph)

        # 4. 估算执行时间
        total_duration = self._estimate_duration(steps)

        # 5. 计算并行因子
        sequential_duration = sum(t.timeout for t in tasks)
        parallel_factor = sequential_duration / max(total_duration, 0.001)

        # 生成计划ID
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter:04d}"

        return ExecutionPlan(
            plan_id=plan_id,
            steps=steps,
            total_tasks=len(tasks),
            estimated_duration=total_duration,
            parallel_factor=parallel_factor,
            metadata={
                "max_parallel": self.max_parallel,
                "context": context or {},
            }
        )

    def _create_empty_plan(self) -> ExecutionPlan:
        """创建空计划"""
        self._plan_counter += 1
        return ExecutionPlan(
            plan_id=f"plan_{self._plan_counter:04d}",
            steps=[],
            total_tasks=0,
            estimated_duration=0.0,
            parallel_factor=1.0,
        )

    def _build_dependency_graph(
        self,
        tasks: List[AtomicTask]
    ) -> Dict[str, Set[str]]:
        """
        构建依赖图

        Returns:
            Dict[task_id, Set[dependent_task_ids]]
        """
        graph = defaultdict(set)
        task_ids = {t.id for t in tasks}

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_ids:
                    graph[task.id].add(dep_id)

        return graph

    def _topological_sort(
        self,
        tasks: List[AtomicTask],
        dep_graph: Dict[str, Set[str]]
    ) -> List[AtomicTask]:
        """
        拓扑排序 (考虑优先级)

        使用 Kahn's algorithm 结合优先级堆
        """
        task_map = {t.id: t for t in tasks}

        # 计算入度
        in_degree = defaultdict(int)
        for task_id in task_map:
            for dep_id in dep_graph.get(task_id, set()):
                in_degree[task_id] += 1

        # 初始化优先队列 (使用负优先级因为 heapq 是最小堆)
        # 格式: (-priority, task_id)
        ready_queue = []
        for task in tasks:
            if in_degree[task.id] == 0:
                heapq.heappush(ready_queue, (-task.priority, task.id))

        sorted_tasks = []
        while ready_queue:
            _, task_id = heapq.heappop(ready_queue)
            task = task_map[task_id]
            sorted_tasks.append(task)

            # 更新依赖此任务的其他任务的入度
            for other_task in tasks:
                if task_id in dep_graph.get(other_task.id, set()):
                    in_degree[other_task.id] -= 1
                    if in_degree[other_task.id] == 0:
                        heapq.heappush(ready_queue, (-other_task.priority, other_task.id))

        # 检查是否有循环依赖
        if len(sorted_tasks) != len(tasks):
            raise ValueError("Circular dependency detected in tasks")

        return sorted_tasks

    def _group_parallel_tasks(
        self,
        sorted_tasks: List[AtomicTask],
        dep_graph: Dict[str, Set[str]]
    ) -> List[PlanStep]:
        """
        将任务分组为可并行执行的步骤
        """
        if not sorted_tasks:
            return []

        steps = []
        completed = set()
        remaining = list(sorted_tasks)
        step_id = 0

        while remaining:
            # 找出所有可以执行的任务 (依赖已完成)
            ready = []
            for task in remaining:
                deps = dep_graph.get(task.id, set())
                if deps.issubset(completed):
                    ready.append(task)

            if not ready:
                # 不应该发生，如果发生说明拓扑排序有问题
                raise RuntimeError("No ready tasks found but tasks remaining")

            # 限制并行数
            if self.prefer_parallel:
                batch = ready[:self.max_parallel]
            else:
                batch = [ready[0]]

            # 确定执行模式
            mode = ExecutionMode.PARALLEL if len(batch) > 1 else ExecutionMode.SEQUENTIAL

            # 创建步骤
            step = PlanStep(
                step_id=step_id,
                tasks=batch,
                execution_mode=mode,
                estimated_duration=max(t.timeout for t in batch),
            )
            steps.append(step)

            # 更新状态
            for task in batch:
                completed.add(task.id)
                remaining.remove(task)

            step_id += 1

        return steps

    def _estimate_duration(self, steps: List[PlanStep]) -> float:
        """估算总执行时间"""
        return sum(step.estimated_duration for step in steps)

    def replan(
        self,
        original_plan: ExecutionPlan,
        completed_tasks: Set[str],
        failed_tasks: Set[str],
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        动态重规划

        当有任务失败或需要调整时重新生成计划。

        Args:
            original_plan: 原始计划
            completed_tasks: 已完成的任务ID集合
            failed_tasks: 失败的任务ID集合
            context: 新的上下文信息

        Returns:
            新的执行计划
        """
        # 收集剩余任务
        remaining_tasks = []
        for step in original_plan.steps:
            for task in step.tasks:
                if task.id not in completed_tasks and task.id not in failed_tasks:
                    remaining_tasks.append(task)

        # 移除依赖失败任务的任务
        valid_tasks = []
        for task in remaining_tasks:
            deps = set(task.dependencies)
            if not deps.intersection(failed_tasks):
                valid_tasks.append(task)

        # 重新规划
        return self.create_plan(valid_tasks, context)


class TradingPlanner(Planner):
    """
    交易任务专用规划器

    针对交易任务的特殊优化。
    """

    # 任务类型优先级 (数字越大优先级越高)
    TYPE_PRIORITY = {
        TaskType.DATA_FETCH: 100,
        TaskType.DATA_VALIDATE: 95,
        TaskType.DATA_TRANSFORM: 90,
        TaskType.INDICATOR_LOAD: 85,
        TaskType.INDICATOR_CALCULATE: 80,
        TaskType.INDICATOR_COMBINE: 75,
        TaskType.SIGNAL_GENERATE: 70,
        TaskType.SIGNAL_FILTER: 65,
        TaskType.SIGNAL_AGGREGATE: 60,
        TaskType.STRATEGY_ANALYZE: 55,
        TaskType.STRATEGY_BACKTEST: 50,
        TaskType.STRATEGY_OPTIMIZE: 45,
        TaskType.PINE_PARSE: 100,
        TaskType.PINE_CONVERT: 95,
        TaskType.CODE_GENERATE: 90,
        TaskType.CUSTOM: 50,
    }

    def __init__(self, max_parallel: int = 4):
        super().__init__(max_parallel=max_parallel, prefer_parallel=True)

    def create_plan(
        self,
        tasks: List[AtomicTask],
        context: Dict[str, Any] = None
    ) -> ExecutionPlan:
        """
        创建交易任务执行计划

        会自动调整任务优先级。
        """
        # 根据任务类型调整优先级
        for task in tasks:
            base_priority = self.TYPE_PRIORITY.get(task.task_type, 50)
            task.priority = base_priority + task.priority

        return super().create_plan(tasks, context)

    def optimize_for_latency(
        self,
        plan: ExecutionPlan,
        critical_path: List[str] = None
    ) -> ExecutionPlan:
        """
        针对延迟优化计划

        确保关键路径任务优先执行。
        """
        if not critical_path:
            return plan

        critical_set = set(critical_path)

        # 重新排序每个步骤中的任务
        for step in plan.steps:
            critical_tasks = [t for t in step.tasks if t.id in critical_set]
            other_tasks = [t for t in step.tasks if t.id not in critical_set]
            step.tasks = critical_tasks + other_tasks

        return plan
