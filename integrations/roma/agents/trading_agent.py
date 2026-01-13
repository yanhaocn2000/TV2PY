"""
ROMA Trading Meta-Agent

交易元代理，整合 TV2PY 指标进行自动化交易分析。
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# 添加 TV2PY 路径
TV2PY_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TV2PY_PATH))

from ..orchestrator import BaseAgent
from ..core.atomizer import AtomicTask, TaskType
from ..core.executor import PlanExecutionResult


@dataclass
class TradingContext:
    """交易上下文"""
    symbol: str = "ETH-USDT"
    timeframe: str = "4h"
    indicators: List[str] = field(default_factory=lambda: ["smc", "squeeze", "supertrend"])
    strategy_type: str = "smart_money"
    min_confidence: float = 0.6
    risk_per_trade: float = 0.02
    data_limit: int = 200


class TradingMetaAgent(BaseAgent):
    """
    交易元代理

    使用 TV2PY 指标进行多维度市场分析。

    支持的分析模式:
        - signal: 生成交易信号
        - analysis: 市场分析
        - backtest: 策略回测
        - optimize: 参数优化

    使用示例:
        agent = TradingMetaAgent(
            indicators=["smc", "squeeze", "supertrend"],
            timeframe="4h",
        )

        result = orchestrator.execute_with_agent(
            agent=agent,
            task_description="分析 ETH-USDT 市场状态",
        )
    """

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        timeframe: str = "4h",
        strategy_type: str = "smart_money",
        min_confidence: float = 0.6,
        risk_per_trade: float = 0.02,
    ):
        super().__init__(name="TradingMetaAgent")

        self.config = TradingContext(
            indicators=indicators or ["smc", "squeeze", "supertrend"],
            timeframe=timeframe,
            strategy_type=strategy_type,
            min_confidence=min_confidence,
            risk_per_trade=risk_per_trade,
        )

        # 尝试导入 TV2PY 组件
        self._load_tv2py_components()

    def _load_tv2py_components(self):
        """加载 TV2PY 组件"""
        try:
            from integrations.hummingbot import IndicatorAdapter, TV2PYSignalGenerator
            self.indicator_adapter = IndicatorAdapter()
            self.signal_generator = None  # 延迟初始化
            self._tv2py_available = True
        except ImportError:
            self.indicator_adapter = None
            self.signal_generator = None
            self._tv2py_available = False

    def prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备交易上下文

        合并用户提供的上下文和默认配置。
        """
        prepared = {
            "symbol": context.get("symbol", self.config.symbol),
            "timeframe": context.get("timeframe", self.config.timeframe),
            "indicators": context.get("indicators", self.config.indicators),
            "strategy_type": context.get("strategy_type", self.config.strategy_type),
            "min_confidence": context.get("min_confidence", self.config.min_confidence),
            "risk_per_trade": context.get("risk_per_trade", self.config.risk_per_trade),
            "limit": context.get("limit", self.config.data_limit),
        }

        # 传递其他上下文
        for key, value in context.items():
            if key not in prepared:
                prepared[key] = value

        return prepared

    def customize_tasks(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Optional[List[AtomicTask]]:
        """
        自定义任务列表

        根据任务描述生成优化的任务序列。
        """
        tasks = []
        task_lower = task_description.lower()
        task_counter = 0

        def make_id(prefix: str) -> str:
            nonlocal task_counter
            task_counter += 1
            return f"{prefix}_{task_counter:04d}"

        # 1. 数据获取任务 (总是需要)
        data_task = AtomicTask(
            id=make_id("data_fetch"),
            task_type=TaskType.DATA_FETCH,
            name="Fetch Market Data",
            description=f"获取 {context['symbol']} {context['timeframe']} 数据",
            inputs={
                "symbol": context["symbol"],
                "timeframe": context["timeframe"],
                "limit": context.get("limit", 200),
            },
            expected_outputs=["candles_df"],
            priority=100,
        )
        tasks.append(data_task)

        # 2. 指标计算任务
        indicator_tasks = []
        for indicator in context.get("indicators", []):
            task = AtomicTask(
                id=make_id("indicator_calc"),
                task_type=TaskType.INDICATOR_CALCULATE,
                name=f"Calculate {indicator.upper()}",
                description=f"计算 {indicator} 指标",
                inputs={
                    "indicator_name": indicator,
                    "params": context.get(f"{indicator}_params", {}),
                },
                dependencies=[data_task.id],
                expected_outputs=[f"{indicator}_result"],
                priority=80,
            )
            indicator_tasks.append(task)
            tasks.append(task)

        # 3. 根据任务类型添加后续任务
        if any(word in task_lower for word in ["signal", "信号", "交易"]):
            # 信号生成
            signal_task = AtomicTask(
                id=make_id("signal_gen"),
                task_type=TaskType.SIGNAL_GENERATE,
                name="Generate Trading Signals",
                description="基于指标生成交易信号",
                inputs={
                    "strategy_type": context["strategy_type"],
                    "min_confidence": context["min_confidence"],
                },
                dependencies=[t.id for t in indicator_tasks],
                expected_outputs=["raw_signals"],
                priority=60,
            )
            tasks.append(signal_task)

            # 信号过滤
            filter_task = AtomicTask(
                id=make_id("signal_filter"),
                task_type=TaskType.SIGNAL_FILTER,
                name="Filter Signals",
                description="过滤低质量信号",
                inputs={"min_confidence": context["min_confidence"]},
                dependencies=[signal_task.id],
                expected_outputs=["filtered_signals"],
                priority=50,
            )
            tasks.append(filter_task)

        if any(word in task_lower for word in ["backtest", "回测", "测试"]):
            # 回测任务
            backtest_task = AtomicTask(
                id=make_id("strategy_bt"),
                task_type=TaskType.STRATEGY_BACKTEST,
                name="Run Backtest",
                description="执行策略回测",
                inputs={
                    "initial_capital": context.get("initial_capital", 10000),
                    "risk_per_trade": context["risk_per_trade"],
                },
                dependencies=[t.id for t in indicator_tasks],
                expected_outputs=["backtest_results", "trades", "metrics"],
                priority=40,
            )
            tasks.append(backtest_task)

        if any(word in task_lower for word in ["optimize", "优化", "调参"]):
            # 优化任务
            opt_task = AtomicTask(
                id=make_id("strategy_opt"),
                task_type=TaskType.STRATEGY_OPTIMIZE,
                name="Optimize Parameters",
                description="优化策略参数",
                inputs={
                    "param_ranges": context.get("param_ranges", {}),
                    "opt_metric": context.get("opt_metric", "sharpe_ratio"),
                },
                dependencies=[t.id for t in indicator_tasks],
                expected_outputs=["optimal_params", "optimization_report"],
                priority=30,
            )
            tasks.append(opt_task)

        return tasks if tasks else None

    def process_results(
        self,
        execution_result: PlanExecutionResult
    ) -> Dict[str, Any]:
        """
        处理执行结果

        生成交易分析报告。
        """
        output = {
            "agent": self.name,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "success_rate": execution_result.success_rate,
        }

        # 收集指标结果
        indicators = {}
        signals = []
        backtest = None

        for task_id, result in execution_result.results.items():
            if not result.success:
                continue

            for key, value in result.output.items():
                if "_result" in key:
                    indicator_name = key.replace("_result", "")
                    indicators[indicator_name] = value
                elif "signals" in key:
                    if isinstance(value, list):
                        signals.extend(value)
                    else:
                        signals.append(value)
                elif "backtest" in key or "metrics" in key:
                    backtest = value

        # 生成分析摘要
        output["indicators"] = indicators
        output["indicator_count"] = len(indicators)

        if signals:
            output["signals"] = signals
            # 聚合信号
            output["aggregated_signal"] = self._aggregate_signals(signals)

        if backtest:
            output["backtest"] = backtest

        # 生成建议
        output["recommendation"] = self._generate_recommendation(output)

        return output

    def _aggregate_signals(self, signals: List[Dict]) -> Dict[str, Any]:
        """聚合多个信号"""
        if not signals:
            return {"signal": "HOLD", "confidence": 0.0}

        from collections import Counter

        types = [s.get("type", "HOLD") for s in signals]
        type_counts = Counter(types)

        most_common, count = type_counts.most_common(1)[0]
        agreement = count / len(signals)

        confidences = [s.get("confidence", 0.5) for s in signals]
        avg_confidence = sum(confidences) / len(confidences)

        return {
            "signal": most_common,
            "confidence": avg_confidence * agreement,
            "agreement": agreement,
            "distribution": dict(type_counts),
        }

    def _generate_recommendation(self, analysis: Dict[str, Any]) -> str:
        """生成交易建议"""
        signal_info = analysis.get("aggregated_signal", {})
        signal = signal_info.get("signal", "HOLD")
        confidence = signal_info.get("confidence", 0)

        if confidence < self.config.min_confidence:
            return f"信号置信度不足 ({confidence:.1%} < {self.config.min_confidence:.1%})，建议观望"

        if signal == "LONG":
            return f"看涨信号 (置信度: {confidence:.1%})，可考虑做多"
        elif signal == "SHORT":
            return f"看跌信号 (置信度: {confidence:.1%})，可考虑做空"
        else:
            return "无明确方向，建议观望"


class QuickAnalysisAgent(TradingMetaAgent):
    """
    快速分析代理

    简化版本，只使用核心指标进行快速分析。
    """

    def __init__(self, timeframe: str = "1h"):
        super().__init__(
            indicators=["supertrend", "bollinger"],
            timeframe=timeframe,
            min_confidence=0.5,
        )
        self.name = "QuickAnalysisAgent"


class DeepAnalysisAgent(TradingMetaAgent):
    """
    深度分析代理

    使用全套指标进行深度市场分析。
    """

    def __init__(self, timeframe: str = "4h"):
        super().__init__(
            indicators=["smc", "squeeze", "supertrend", "bollinger", "atr", "volume_profile", "choppiness"],
            timeframe=timeframe,
            min_confidence=0.7,
        )
        self.name = "DeepAnalysisAgent"
