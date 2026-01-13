"""
ROMA Integration Tests

测试 ROMA 框架与 TV2PY 的集成。
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def test_atomizer():
    """测试任务原子化器"""
    print("\n" + "=" * 60)
    print("Testing Atomizer")
    print("=" * 60)

    from integrations.roma.core import TradingAtomizer

    atomizer = TradingAtomizer()

    # 测试不同任务描述
    test_cases = [
        "Fetch ETH market data",
        "Calculate SMC and Squeeze indicators",
        "Generate trading signals",
        "Run backtest on strategy",
        "Optimize strategy parameters",
        "Convert Pine Script to Python",
    ]

    for task in test_cases:
        tasks = atomizer.atomize(task, context={
            "indicators": ["smc", "squeeze"],
            "symbol": "ETH-USDT",
        })
        print(f"\n'{task}':")
        print(f"  Generated {len(tasks)} atomic tasks")
        for t in tasks:
            print(f"    - {t.name} ({t.task_type.value})")

    print("\nAtomizer test passed!")


def test_planner():
    """测试执行规划器"""
    print("\n" + "=" * 60)
    print("Testing Planner")
    print("=" * 60)

    from integrations.roma.core import TradingAtomizer, TradingPlanner

    atomizer = TradingAtomizer()
    planner = TradingPlanner(max_parallel=4)

    # 创建任务
    tasks = atomizer.atomize(
        "Fetch data, calculate indicators, and generate signals",
        context={"indicators": ["smc", "squeeze", "supertrend"]}
    )

    # 创建执行计划
    plan = planner.create_plan(tasks)

    print(f"\nExecution Plan: {plan.plan_id}")
    print(f"  Total tasks: {plan.total_tasks}")
    print(f"  Total steps: {plan.total_steps}")
    print(f"  Estimated duration: {plan.estimated_duration:.1f}s")
    print(f"  Parallel factor: {plan.parallel_factor:.2f}x")

    print("\nSteps:")
    for step in plan.steps:
        task_names = [t.name for t in step.tasks]
        print(f"  Step {step.step_id}: {task_names} ({step.execution_mode.value})")

    print("\nPlanner test passed!")


def test_executor():
    """测试任务执行器"""
    print("\n" + "=" * 60)
    print("Testing Executor")
    print("=" * 60)

    from integrations.roma.core import (
        TradingAtomizer,
        TradingPlanner,
        Executor,
        TradingTaskExecutor,
        TaskType,
    )

    atomizer = TradingAtomizer()
    planner = TradingPlanner(max_parallel=4)
    executor = Executor(max_workers=4)

    # 注册交易任务执行器
    trading_executor = TradingTaskExecutor()
    for task_type in TaskType:
        executor.register_executor(task_type, trading_executor)

    # 创建并执行计划
    tasks = atomizer.atomize(
        "Fetch data and calculate indicators",
        context={"indicators": ["smc", "squeeze"]}
    )
    plan = planner.create_plan(tasks)
    result = executor.execute_plan(plan, context={"symbol": "ETH-USDT"})

    print(f"\nExecution Result:")
    print(f"  Status: {result.status.value}")
    print(f"  Success rate: {result.success_rate:.1%}")
    print(f"  Total duration: {result.total_duration:.2f}s")
    print(f"  Success: {result.success_count}, Failed: {result.failure_count}")

    print("\nTask Results:")
    for task_id, task_result in result.results.items():
        print(f"  {task_id}: {task_result.status.value}")

    print("\nExecutor test passed!")


def test_aggregator():
    """测试结果聚合器"""
    print("\n" + "=" * 60)
    print("Testing Aggregator")
    print("=" * 60)

    from integrations.roma.core import (
        TradingAggregator,
        ExecutionResult,
        ExecutionStatus,
        AggregationMethod,
    )

    aggregator = TradingAggregator()

    # 创建模拟结果
    results = [
        ExecutionResult(
            task_id="signal_1",
            status=ExecutionStatus.COMPLETED,
            output={
                "raw_signals": [{"type": "LONG", "confidence": 0.7, "strength": 0.6}]
            }
        ),
        ExecutionResult(
            task_id="signal_2",
            status=ExecutionStatus.COMPLETED,
            output={
                "raw_signals": [{"type": "LONG", "confidence": 0.8, "strength": 0.7}]
            }
        ),
        ExecutionResult(
            task_id="signal_3",
            status=ExecutionStatus.COMPLETED,
            output={
                "raw_signals": [{"type": "SHORT", "confidence": 0.5, "strength": 0.4}]
            }
        ),
    ]

    # 聚合信号
    aggregated = aggregator.aggregate_signals(results)

    print(f"\nAggregated Signal:")
    print(f"  Method: {aggregated.method}")
    print(f"  Confidence: {aggregated.confidence:.2f}")
    print(f"  Source count: {aggregated.source_count}")
    print(f"  Value: {aggregated.value}")

    print("\nAggregator test passed!")


def test_verifier():
    """测试结果验证器"""
    print("\n" + "=" * 60)
    print("Testing Verifier")
    print("=" * 60)

    from integrations.roma.core import (
        TradingVerifier,
        ExecutionResult,
        ExecutionStatus,
    )

    verifier = TradingVerifier()

    # 测试有效信号
    valid_result = ExecutionResult(
        task_id="test_1",
        status=ExecutionStatus.COMPLETED,
        output={
            "signal": "LONG",
            "confidence": 0.75,
            "strength": 0.6,
        }
    )

    verification = verifier.verify(valid_result)
    print(f"\nValid Signal Verification:")
    print(f"  Status: {verification.status.value}")
    print(f"  Score: {verification.score:.2f}")
    print(f"  Passed rules: {verification.passed_rules}")

    # 测试无效信号
    invalid_result = ExecutionResult(
        task_id="test_2",
        status=ExecutionStatus.COMPLETED,
        output={
            "signal": "INVALID",
            "confidence": 1.5,  # 超出范围
            "strength": -0.1,  # 负数
        }
    )

    verification = verifier.verify(invalid_result)
    print(f"\nInvalid Signal Verification:")
    print(f"  Status: {verification.status.value}")
    print(f"  Score: {verification.score:.2f}")
    print(f"  Failed rules: {verification.failed_rules}")

    print("\nVerifier test passed!")


def test_orchestrator():
    """测试编排器"""
    print("\n" + "=" * 60)
    print("Testing Orchestrator")
    print("=" * 60)

    from integrations.roma import ROMAOrchestrator

    orchestrator = ROMAOrchestrator()

    # 执行简单任务
    result = orchestrator.execute(
        task_description="Fetch market data and calculate SMC indicator",
        context={
            "symbol": "ETH-USDT",
            "timeframe": "4h",
            "indicators": ["smc"],
        },
        on_progress=lambda stage, data: print(f"  [{stage}] {data}"),
    )

    print(f"\nOrchestration Result:")
    print(f"  ID: {result.orchestration_id}")
    print(f"  Status: {result.status}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Tasks: {len(result.tasks)}")

    if result.plan:
        print(f"  Plan steps: {result.plan.total_steps}")

    if result.execution_result:
        print(f"  Execution success rate: {result.execution_result.success_rate:.1%}")

    if result.verification_result:
        print(f"  Verification score: {result.verification_result.score:.2f}")

    print(f"\nFinal Output:")
    for key, value in result.final_output.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    print("\nOrchestrator test passed!")


def test_trading_agent():
    """测试交易代理"""
    print("\n" + "=" * 60)
    print("Testing Trading Agent")
    print("=" * 60)

    from integrations.roma import ROMAOrchestrator, TradingMetaAgent

    orchestrator = ROMAOrchestrator()

    # 创建交易代理
    agent = TradingMetaAgent(
        indicators=["smc", "squeeze", "supertrend"],
        timeframe="4h",
        min_confidence=0.6,
    )

    # 使用代理执行
    result = orchestrator.execute_with_agent(
        agent=agent,
        task_description="分析 ETH-USDT 并生成交易信号",
        context={
            "symbol": "ETH-USDT",
            "limit": 200,
        }
    )

    print(f"\nAgent Result:")
    print(f"  Status: {result.status}")
    print(f"  Tasks: {len(result.tasks)}")

    print(f"\nFinal Output:")
    output = result.final_output
    print(f"  Agent: {output.get('agent')}")
    print(f"  Symbol: {output.get('symbol')}")
    print(f"  Timeframe: {output.get('timeframe')}")
    print(f"  Indicator count: {output.get('indicator_count', 0)}")

    if "aggregated_signal" in output:
        signal = output["aggregated_signal"]
        print(f"\n  Aggregated Signal:")
        print(f"    Signal: {signal.get('signal')}")
        print(f"    Confidence: {signal.get('confidence', 0):.2%}")

    if "recommendation" in output:
        print(f"\n  Recommendation: {output['recommendation']}")

    print("\nTrading Agent test passed!")


def test_indicator_analyzer():
    """测试指标分析代理"""
    print("\n" + "=" * 60)
    print("Testing Indicator Analyzer")
    print("=" * 60)

    from integrations.roma.agents import IndicatorAnalyzerAgent

    agent = IndicatorAnalyzerAgent()

    # 测试 Pine Script 代码分析
    pine_code = """
//@version=5
indicator("SuperTrend", overlay=true)

// Inputs
atr_period = input(10, "ATR Period")
multiplier = input(3.0, "ATR Multiplier")

// Calculate ATR
atr = ta.atr(atr_period)

// Calculate SuperTrend
upper_band = (high + low) / 2 + multiplier * atr
lower_band = (high + low) / 2 - multiplier * atr

// Determine trend
var trend = 1
trend := close > upper_band[1] ? 1 : close < lower_band[1] ? -1 : trend[1]

// Plot
plot(trend == 1 ? lower_band : upper_band, color=trend == 1 ? color.green : color.red)
    """

    analysis = agent.analyze_pine_code(pine_code)

    print(f"\nPine Script Analysis:")
    print(f"  Name: {analysis['name']}")
    print(f"  Version: {analysis['version']}")
    print(f"  Complexity: {analysis['complexity']}")
    print(f"  Conversion Difficulty: {analysis['conversion_difficulty']}/3")

    print(f"\n  Inputs: {len(analysis['inputs'])}")
    for inp in analysis['inputs']:
        print(f"    - {inp['name']}")

    print(f"\n  Functions used: {analysis['functions_used']}")
    print(f"  Plots: {len(analysis['plots'])}")

    # 测试函数映射建议
    mapping = agent.suggest_python_mapping(analysis['functions_used'])
    print(f"\n  Suggested Python mappings:")
    for pine_fn, py_fn in mapping.items():
        print(f"    {pine_fn} -> {py_fn}")

    print("\nIndicator Analyzer test passed!")


def test_strategy_optimizer():
    """测试策略优化代理"""
    print("\n" + "=" * 60)
    print("Testing Strategy Optimizer")
    print("=" * 60)

    from integrations.roma import ROMAOrchestrator
    from integrations.roma.agents import StrategyOptimizerAgent

    orchestrator = ROMAOrchestrator()

    # 创建优化代理
    optimizer = StrategyOptimizerAgent(
        method="grid_search",
        metric="sharpe_ratio",
        max_iterations=10,  # 限制迭代次数用于测试
    )

    optimizer.add_parameter("period", 10, 20, step=5)
    optimizer.add_parameter("multiplier", 2.0, 3.0, step=0.5)

    print(f"\nOptimizer Configuration:")
    print(f"  Method: {optimizer.config.method}")
    print(f"  Metric: {optimizer.config.metric}")
    print(f"  Max iterations: {optimizer.config.max_iterations}")
    print(f"  Parameters: {len(optimizer.config.parameter_ranges)}")

    # 生成参数组合
    combinations = optimizer._generate_param_combinations()
    print(f"\n  Generated {len(combinations)} parameter combinations")
    print(f"  Sample combinations:")
    for combo in combinations[:3]:
        print(f"    {combo}")

    print("\nStrategy Optimizer test passed!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TV2PY ROMA Integration Tests")
    print("=" * 60)

    tests = [
        ("Atomizer", test_atomizer),
        ("Planner", test_planner),
        ("Executor", test_executor),
        ("Aggregator", test_aggregator),
        ("Verifier", test_verifier),
        ("Orchestrator", test_orchestrator),
        ("Trading Agent", test_trading_agent),
        ("Indicator Analyzer", test_indicator_analyzer),
        ("Strategy Optimizer", test_strategy_optimizer),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n{name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
