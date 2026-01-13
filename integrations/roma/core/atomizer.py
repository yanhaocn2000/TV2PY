"""
ROMA Atomizer - 任务原子化组件

将复杂任务分解为可执行的原子操作。
对于 TV2PY，这包括:
    - Pine Script 代码分析
    - 指标参数提取
    - 数据获取任务
    - 计算任务
    - 信号生成任务
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import re
from abc import ABC, abstractmethod


class TaskType(Enum):
    """原子任务类型"""
    # 数据相关
    DATA_FETCH = "data_fetch"           # 获取市场数据
    DATA_TRANSFORM = "data_transform"   # 数据转换
    DATA_VALIDATE = "data_validate"     # 数据验证

    # 指标相关
    INDICATOR_LOAD = "indicator_load"       # 加载指标
    INDICATOR_CALCULATE = "indicator_calc"  # 计算指标
    INDICATOR_COMBINE = "indicator_combine" # 组合指标

    # 信号相关
    SIGNAL_GENERATE = "signal_gen"      # 生成信号
    SIGNAL_FILTER = "signal_filter"     # 过滤信号
    SIGNAL_AGGREGATE = "signal_agg"     # 聚合信号

    # 策略相关
    STRATEGY_ANALYZE = "strategy_analyze"   # 分析策略
    STRATEGY_OPTIMIZE = "strategy_opt"      # 优化策略
    STRATEGY_BACKTEST = "strategy_bt"       # 回测策略

    # 转换相关
    PINE_PARSE = "pine_parse"           # 解析 Pine Script
    PINE_CONVERT = "pine_convert"       # 转换 Pine Script
    CODE_GENERATE = "code_gen"          # 生成代码

    # 通用
    CUSTOM = "custom"                   # 自定义任务


@dataclass
class AtomicTask:
    """
    原子任务定义

    原子任务是不可再分的最小执行单元。
    """
    id: str
    task_type: TaskType
    name: str
    description: str

    # 输入输出
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)

    # 依赖关系
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID

    # 执行参数
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0  # 超时时间(秒)
    retry_count: int = 3   # 重试次数

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0      # 优先级 (越高越优先)

    def __hash__(self):
        return hash(self.id)


class Atomizer(ABC):
    """
    原子化器基类

    负责将高层任务分解为原子任务。
    """

    def __init__(self):
        self._task_counter = 0
        self._task_registry: Dict[str, AtomicTask] = {}

    def _generate_task_id(self, prefix: str = "task") -> str:
        """生成唯一任务ID"""
        self._task_counter += 1
        return f"{prefix}_{self._task_counter:04d}"

    @abstractmethod
    def atomize(self, task_description: str, context: Dict[str, Any] = None) -> List[AtomicTask]:
        """
        将任务描述分解为原子任务列表

        Args:
            task_description: 任务描述
            context: 上下文信息

        Returns:
            原子任务列表
        """
        pass

    def create_task(
        self,
        task_type: TaskType,
        name: str,
        description: str,
        inputs: Dict[str, Any] = None,
        dependencies: List[str] = None,
        **kwargs
    ) -> AtomicTask:
        """创建原子任务"""
        task_id = self._generate_task_id(task_type.value)
        task = AtomicTask(
            id=task_id,
            task_type=task_type,
            name=name,
            description=description,
            inputs=inputs or {},
            dependencies=dependencies or [],
            **kwargs
        )
        self._task_registry[task_id] = task
        return task


class TradingAtomizer(Atomizer):
    """
    交易任务原子化器

    专门处理交易相关任务的分解。
    """

    # 任务模式识别
    TASK_PATTERNS = {
        "data": [
            r"(fetch|get|download|load)\s+(data|candles?|bars?|ohlc)",
            r"(market|price|historical)\s+data",
        ],
        "indicator": [
            r"(calculate|compute|run)\s+(\w+\s+)?(indicator|smc|squeeze|supertrend|bollinger|atr|rsi|macd)",
            r"(add|use|apply)\s+(\w+\s+)?indicator",
        ],
        "signal": [
            r"(generate|create|get)\s+(trading\s+)?signal",
            r"(buy|sell|entry|exit)\s+signal",
        ],
        "backtest": [
            r"(run|execute|perform)\s+(back)?test",
            r"(analyze|evaluate)\s+(strategy|performance)",
        ],
        "optimize": [
            r"(optimize|tune|improve)\s+(parameter|strategy|indicator)",
            r"(find|search)\s+(best|optimal)\s+(parameter|setting)",
        ],
        "convert": [
            r"(convert|translate|transform)\s+(pine|tradingview|script)",
            r"(pine\s*script|tradingview)\s+(to|->)\s+python",
        ],
    }

    def atomize(self, task_description: str, context: Dict[str, Any] = None) -> List[AtomicTask]:
        """
        分解交易任务

        将自然语言任务描述转换为原子任务序列。
        """
        context = context or {}
        tasks = []
        task_lower = task_description.lower()

        # 识别任务类型
        identified_types = self._identify_task_types(task_lower)

        # 根据识别的类型生成原子任务
        if "data" in identified_types or not identified_types:
            # 数据获取任务
            tasks.extend(self._create_data_tasks(task_description, context))

        if "indicator" in identified_types:
            # 指标计算任务
            tasks.extend(self._create_indicator_tasks(task_description, context))

        if "signal" in identified_types:
            # 信号生成任务
            tasks.extend(self._create_signal_tasks(task_description, context))

        if "backtest" in identified_types:
            # 回测任务
            tasks.extend(self._create_backtest_tasks(task_description, context))

        if "optimize" in identified_types:
            # 优化任务
            tasks.extend(self._create_optimization_tasks(task_description, context))

        if "convert" in identified_types:
            # 转换任务
            tasks.extend(self._create_conversion_tasks(task_description, context))

        # 建立依赖关系
        self._establish_dependencies(tasks)

        return tasks

    def _identify_task_types(self, task_lower: str) -> List[str]:
        """识别任务中包含的类型"""
        identified = []
        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    identified.append(task_type)
                    break
        return identified

    def _create_data_tasks(self, description: str, context: Dict) -> List[AtomicTask]:
        """创建数据相关任务"""
        tasks = []

        # 数据获取
        fetch_task = self.create_task(
            task_type=TaskType.DATA_FETCH,
            name="Fetch Market Data",
            description="获取市场 OHLCV 数据",
            inputs={
                "symbol": context.get("symbol", "ETH-USDT"),
                "timeframe": context.get("timeframe", "4h"),
                "limit": context.get("limit", 200),
            },
            expected_outputs=["candles_df"],
            priority=10,
        )
        tasks.append(fetch_task)

        # 数据验证
        validate_task = self.create_task(
            task_type=TaskType.DATA_VALIDATE,
            name="Validate Data",
            description="验证数据完整性和质量",
            inputs={"data_ref": fetch_task.id},
            dependencies=[fetch_task.id],
            expected_outputs=["validated_df", "data_quality_report"],
            priority=9,
        )
        tasks.append(validate_task)

        return tasks

    def _create_indicator_tasks(self, description: str, context: Dict) -> List[AtomicTask]:
        """创建指标计算任务"""
        tasks = []
        indicators = context.get("indicators", ["smc", "squeeze", "supertrend"])

        # 为每个指标创建计算任务
        for indicator in indicators:
            task = self.create_task(
                task_type=TaskType.INDICATOR_CALCULATE,
                name=f"Calculate {indicator.upper()}",
                description=f"计算 {indicator} 指标",
                inputs={
                    "indicator_name": indicator,
                    "params": context.get(f"{indicator}_params", {}),
                },
                expected_outputs=[f"{indicator}_result"],
                priority=8,
            )
            tasks.append(task)

        # 如果有多个指标，添加组合任务
        if len(indicators) > 1:
            combine_task = self.create_task(
                task_type=TaskType.INDICATOR_COMBINE,
                name="Combine Indicators",
                description="组合多个指标结果",
                inputs={"indicators": indicators},
                dependencies=[t.id for t in tasks],
                expected_outputs=["combined_indicators"],
                priority=7,
            )
            tasks.append(combine_task)

        return tasks

    def _create_signal_tasks(self, description: str, context: Dict) -> List[AtomicTask]:
        """创建信号生成任务"""
        tasks = []

        # 信号生成
        gen_task = self.create_task(
            task_type=TaskType.SIGNAL_GENERATE,
            name="Generate Trading Signals",
            description="基于指标生成交易信号",
            inputs={
                "strategy_type": context.get("strategy_type", "smart_money"),
                "min_confidence": context.get("min_confidence", 0.6),
            },
            expected_outputs=["raw_signals"],
            priority=6,
        )
        tasks.append(gen_task)

        # 信号过滤
        filter_task = self.create_task(
            task_type=TaskType.SIGNAL_FILTER,
            name="Filter Signals",
            description="过滤低质量信号",
            inputs={
                "signal_ref": gen_task.id,
                "filters": context.get("signal_filters", ["confidence", "strength"]),
            },
            dependencies=[gen_task.id],
            expected_outputs=["filtered_signals"],
            priority=5,
        )
        tasks.append(filter_task)

        return tasks

    def _create_backtest_tasks(self, description: str, context: Dict) -> List[AtomicTask]:
        """创建回测任务"""
        tasks = []

        # 策略分析
        analyze_task = self.create_task(
            task_type=TaskType.STRATEGY_ANALYZE,
            name="Analyze Strategy",
            description="分析策略配置和历史表现",
            inputs={
                "strategy_config": context.get("strategy_config", {}),
            },
            expected_outputs=["strategy_analysis"],
            priority=5,
        )
        tasks.append(analyze_task)

        # 执行回测
        backtest_task = self.create_task(
            task_type=TaskType.STRATEGY_BACKTEST,
            name="Run Backtest",
            description="执行策略回测",
            inputs={
                "analysis_ref": analyze_task.id,
                "start_date": context.get("start_date"),
                "end_date": context.get("end_date"),
                "initial_capital": context.get("initial_capital", 10000),
            },
            dependencies=[analyze_task.id],
            expected_outputs=["backtest_results", "trades", "metrics"],
            priority=4,
        )
        tasks.append(backtest_task)

        return tasks

    def _create_optimization_tasks(self, description: str, context: Dict) -> List[AtomicTask]:
        """创建优化任务"""
        tasks = []

        # 参数优化
        opt_task = self.create_task(
            task_type=TaskType.STRATEGY_OPTIMIZE,
            name="Optimize Parameters",
            description="优化策略参数",
            inputs={
                "param_ranges": context.get("param_ranges", {}),
                "optimization_method": context.get("opt_method", "grid_search"),
                "metric": context.get("opt_metric", "sharpe_ratio"),
            },
            expected_outputs=["optimal_params", "optimization_report"],
            priority=3,
        )
        tasks.append(opt_task)

        return tasks

    def _create_conversion_tasks(self, description: str, context: Dict) -> List[AtomicTask]:
        """创建 Pine Script 转换任务"""
        tasks = []

        # 解析 Pine Script
        parse_task = self.create_task(
            task_type=TaskType.PINE_PARSE,
            name="Parse Pine Script",
            description="解析 Pine Script 代码结构",
            inputs={
                "pine_code": context.get("pine_code", ""),
            },
            expected_outputs=["ast", "parsed_elements"],
            priority=10,
        )
        tasks.append(parse_task)

        # 转换为 Python
        convert_task = self.create_task(
            task_type=TaskType.PINE_CONVERT,
            name="Convert to Python",
            description="将 Pine Script 转换为 Python 代码",
            inputs={
                "ast_ref": parse_task.id,
            },
            dependencies=[parse_task.id],
            expected_outputs=["python_code", "conversion_notes"],
            priority=9,
        )
        tasks.append(convert_task)

        # 生成最终代码
        gen_task = self.create_task(
            task_type=TaskType.CODE_GENERATE,
            name="Generate Final Code",
            description="生成可用的 Python 指标代码",
            inputs={
                "code_ref": convert_task.id,
                "template": context.get("template", "indicator"),
            },
            dependencies=[convert_task.id],
            expected_outputs=["final_code", "test_code"],
            priority=8,
        )
        tasks.append(gen_task)

        return tasks

    def _establish_dependencies(self, tasks: List[AtomicTask]):
        """建立任务间的依赖关系"""
        task_map = {t.id: t for t in tasks}

        # 按类型建立依赖
        data_tasks = [t for t in tasks if t.task_type in [TaskType.DATA_FETCH, TaskType.DATA_VALIDATE]]
        indicator_tasks = [t for t in tasks if t.task_type in [TaskType.INDICATOR_CALCULATE, TaskType.INDICATOR_COMBINE]]
        signal_tasks = [t for t in tasks if t.task_type in [TaskType.SIGNAL_GENERATE, TaskType.SIGNAL_FILTER]]

        # 指标任务依赖数据任务
        if data_tasks and indicator_tasks:
            data_task_ids = [t.id for t in data_tasks]
            for task in indicator_tasks:
                if task.task_type == TaskType.INDICATOR_CALCULATE:
                    task.dependencies.extend([tid for tid in data_task_ids if tid not in task.dependencies])

        # 信号任务依赖指标任务
        if indicator_tasks and signal_tasks:
            indicator_task_ids = [t.id for t in indicator_tasks]
            for task in signal_tasks:
                if task.task_type == TaskType.SIGNAL_GENERATE:
                    task.dependencies.extend([tid for tid in indicator_task_ids if tid not in task.dependencies])
