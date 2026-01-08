# TV2PY ROMA Integration

ROMA (Recursive Open Meta-Agents) 框架集成，为 TV2PY 提供智能代理能力。

## 概述

ROMA 是一个用于构建递归智能代理系统的框架。本集成将 ROMA 与 TV2PY 结合，提供:

- **自动化任务分解**: 将复杂交易任务分解为可执行的原子操作
- **智能执行规划**: 优化任务执行顺序，支持并行处理
- **多指标信号聚合**: 综合多个指标生成统一信号
- **结果验证**: 确保输出的正确性和一致性
- **Pine Script 分析**: 自动分析和转换 TradingView 指标

## 架构

```
ROMA Framework
├── Core Components (核心组件)
│   ├── Atomizer     - 任务原子化
│   ├── Planner      - 执行规划
│   ├── Executor     - 任务执行
│   ├── Aggregator   - 结果聚合
│   └── Verifier     - 结果验证
│
├── Orchestrator (编排器)
│   └── ROMAOrchestrator - 协调所有组件
│
└── Trading Agents (交易代理)
    ├── TradingMetaAgent      - 交易元代理
    ├── IndicatorAnalyzerAgent - 指标分析代理
    └── StrategyOptimizerAgent - 策略优化代理
```

## 快速开始

### 基本使用

```python
from integrations.roma import ROMAOrchestrator, TradingMetaAgent

# 创建编排器
orchestrator = ROMAOrchestrator()

# 创建交易元代理
agent = TradingMetaAgent(
    indicators=["smc", "squeeze", "supertrend"],
    timeframe="4h",
)

# 执行分析任务
result = orchestrator.execute_with_agent(
    agent=agent,
    task_description="分析 ETH-USDT 并生成交易信号",
    context={
        "symbol": "ETH-USDT",
        "limit": 200,
    }
)

print(result.final_output)
```

### 使用不同代理

```python
from integrations.roma import (
    ROMAOrchestrator,
    TradingMetaAgent,
    IndicatorAnalyzerAgent,
    StrategyOptimizerAgent,
)

orchestrator = ROMAOrchestrator()

# 1. 快速分析
from integrations.roma.agents.trading_agent import QuickAnalysisAgent
quick_agent = QuickAnalysisAgent(timeframe="1h")
result = orchestrator.execute_with_agent(quick_agent, "快速分析市场")

# 2. 深度分析
from integrations.roma.agents.trading_agent import DeepAnalysisAgent
deep_agent = DeepAnalysisAgent(timeframe="4h")
result = orchestrator.execute_with_agent(deep_agent, "深度分析市场")

# 3. 指标分析
analyzer = IndicatorAnalyzerAgent()
result = orchestrator.execute_with_agent(
    analyzer,
    "分析 Pine Script 代码",
    context={"pine_code": pine_script_code}
)

# 4. 策略优化
optimizer = StrategyOptimizerAgent(metric="sharpe_ratio")
optimizer.add_parameter("sma_period", 10, 50, step=5)
optimizer.add_parameter("rsi_threshold", 20, 40, step=5)
result = orchestrator.execute_with_agent(optimizer, "优化策略参数")
```

## 核心组件

### Atomizer (原子化器)

将高层任务分解为原子任务:

```python
from integrations.roma.core import TradingAtomizer, TaskType

atomizer = TradingAtomizer()
tasks = atomizer.atomize(
    "分析 ETH 市场并生成交易信号",
    context={"indicators": ["smc", "squeeze"]}
)

for task in tasks:
    print(f"{task.name}: {task.task_type}")
```

支持的任务类型:
- `DATA_FETCH` - 数据获取
- `INDICATOR_CALCULATE` - 指标计算
- `SIGNAL_GENERATE` - 信号生成
- `STRATEGY_BACKTEST` - 策略回测
- `PINE_PARSE` - Pine Script 解析
- `PINE_CONVERT` - Pine Script 转换

### Planner (规划器)

生成优化的执行计划:

```python
from integrations.roma.core import TradingPlanner

planner = TradingPlanner(max_parallel=4)
plan = planner.create_plan(tasks)

print(f"计划包含 {plan.total_steps} 个步骤")
print(f"预估耗时: {plan.estimated_duration}s")
print(f"并行因子: {plan.parallel_factor:.2f}x")
```

### Executor (执行器)

执行任务并收集结果:

```python
from integrations.roma.core import Executor

executor = Executor(max_workers=4)
result = executor.execute_plan(plan, context)

print(f"成功率: {result.success_rate:.1%}")
print(f"总耗时: {result.total_duration:.2f}s")
```

### Aggregator (聚合器)

聚合多个任务的结果:

```python
from integrations.roma.core import TradingAggregator, AggregationMethod

aggregator = TradingAggregator()

# 聚合信号
signal_result = aggregator.aggregate_signals(signal_results)
print(f"聚合信号: {signal_result.value}")

# 聚合指标
indicator_result = aggregator.aggregate_indicators(indicator_results)
```

### Verifier (验证器)

验证结果的正确性:

```python
from integrations.roma.core import TradingVerifier

verifier = TradingVerifier()
verification = verifier.verify(result)

print(f"验证状态: {verification.status}")
print(f"验证分数: {verification.score:.2f}")
```

## 交易代理

### TradingMetaAgent

通用交易分析代理:

```python
agent = TradingMetaAgent(
    indicators=["smc", "squeeze", "supertrend"],
    timeframe="4h",
    strategy_type="smart_money",
    min_confidence=0.6,
    risk_per_trade=0.02,
)
```

### IndicatorAnalyzerAgent

分析 Pine Script 代码:

```python
from integrations.roma.agents import IndicatorAnalyzerAgent

agent = IndicatorAnalyzerAgent()
analysis = agent.analyze_pine_code(pine_code)

print(f"指标名称: {analysis['name']}")
print(f"复杂度: {analysis['complexity']}")
print(f"使用的函数: {analysis['functions_used']}")
```

### StrategyOptimizerAgent

策略参数优化:

```python
from integrations.roma.agents import StrategyOptimizerAgent

optimizer = StrategyOptimizerAgent(
    method="grid_search",  # 或 "random_search"
    metric="sharpe_ratio",
    maximize=True,
)

optimizer.add_parameter("period", 10, 50, step=5)
optimizer.add_parameter("multiplier", 1.0, 4.0, step=0.5)
```

## 与 TV2PY 指标集成

ROMA 可以直接使用 TV2PY 的 50+ 个指标:

```python
# 通过 Hummingbot 适配器使用
from integrations.hummingbot import IndicatorAdapter

adapter = IndicatorAdapter()
adapter.add_indicator("smc", swing_length=10)
adapter.add_indicator("squeeze", bb_length=20)
adapter.add_indicator("supertrend", period=10, multiplier=3)

# 在 ROMA 任务中使用
context = {
    "indicators": ["smc", "squeeze", "supertrend"],
    "smc_params": {"swing_length": 10},
}
```

## 进度回调

监控任务执行进度:

```python
def on_progress(stage: str, data: dict):
    print(f"[{stage}] {data}")

result = orchestrator.execute(
    task_description="分析市场",
    on_progress=on_progress,
)
```

## 自定义组件

### 自定义原子化器

```python
from integrations.roma.core import Atomizer, AtomicTask

class MyAtomizer(Atomizer):
    def atomize(self, task_description, context):
        # 自定义任务分解逻辑
        tasks = []
        # ...
        return tasks
```

### 自定义验证规则

```python
from integrations.roma.core import Verifier, VerificationRule

verifier = TradingVerifier()
verifier.add_rule(VerificationRule(
    name="custom_check",
    description="自定义验证",
    check_fn=lambda d: d.get("value") > 0,
    severity="warning",
))
```

## 示例用例

### 1. 市场分析工作流

```python
# 创建分析工作流
orchestrator = ROMAOrchestrator()
agent = DeepAnalysisAgent()

result = orchestrator.execute_with_agent(
    agent,
    "全面分析 ETH-USDT 市场状态，包括趋势、动量和支撑阻力",
    context={
        "symbol": "ETH-USDT",
        "timeframe": "4h",
    }
)

# 获取分析结果
print("指标分析:", result.final_output.get("indicators"))
print("交易建议:", result.final_output.get("recommendation"))
```

### 2. 参数优化工作流

```python
# 优化 SuperTrend 参数
optimizer = StrategyOptimizerAgent(metric="sharpe_ratio")
optimizer.add_parameter("period", 7, 21, step=2)
optimizer.add_parameter("multiplier", 1.5, 4.0, step=0.5)

result = orchestrator.execute_with_agent(
    optimizer,
    "优化 SuperTrend 策略参数",
)

print("最佳参数:", result.final_output.get("best_params"))
print("优化报告:", result.final_output.get("optimization_report"))
```

### 3. Pine Script 转换工作流

```python
from integrations.roma.agents.indicator_analyzer import PineScriptConverter

converter = PineScriptConverter()
conversion = converter.convert(pine_code)

print("Python 代码:")
print(conversion["python_code"])
print("\n转换注意事项:")
for note in conversion["notes"]:
    print(f"- {note}")
```

## 配置选项

### 编排器配置

```python
orchestrator = ROMAOrchestrator(
    atomizer=TradingAtomizer(),      # 自定义原子化器
    planner=TradingPlanner(max_parallel=8),  # 增加并行度
    executor=Executor(max_workers=8),
    aggregator=TradingAggregator(),
    verifier=TradingVerifier(),
)
```

### 执行配置

```python
result = orchestrator.execute(
    task_description="分析任务",
    context={"symbol": "ETH-USDT"},
    skip_verification=False,  # 是否跳过验证
)
```

## 错误处理

```python
result = orchestrator.execute(task_description, context)

if not result.success:
    print(f"执行失败: {result.final_output.get('error')}")
    print(f"失败任务数: {result.execution_result.failure_count}")

# 检查验证结果
if result.verification_result and not result.verification_result.is_valid:
    print(f"验证失败: {result.verification_result.failed_rules}")
```

## 依赖

- TV2PY 指标库
- NumPy
- Pandas
- (可选) Hummingbot 适配器

## 参考

- [ROMA 论文](https://arxiv.org/abs/2402.01623)
- [TV2PY 文档](../README.md)
- [Hummingbot 集成](../hummingbot/README.md)
