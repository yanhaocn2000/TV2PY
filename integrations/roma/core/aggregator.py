"""
ROMA Aggregator - 结果聚合组件

聚合多个任务的执行结果，生成综合输出。
支持:
    - 多指标结果聚合
    - 信号综合评估
    - 回测结果汇总
    - 自定义聚合策略
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .executor import ExecutionResult, ExecutionStatus


class AggregationMethod(Enum):
    """聚合方法"""
    SIMPLE_AVERAGE = "simple_avg"       # 简单平均
    WEIGHTED_AVERAGE = "weighted_avg"   # 加权平均
    VOTING = "voting"                   # 投票法
    CONSENSUS = "consensus"             # 共识法
    PRIORITY = "priority"               # 优先级法
    CUSTOM = "custom"                   # 自定义


@dataclass
class AggregatedResult:
    """
    聚合结果
    """
    aggregation_id: str
    method: AggregationMethod
    value: Any
    confidence: float
    source_count: int
    source_results: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Aggregator:
    """
    结果聚合器

    聚合多个任务的执行结果。
    """

    def __init__(self):
        self._aggregation_counter = 0
        self._custom_aggregators: Dict[str, Callable] = {}

    def register_aggregator(
        self,
        name: str,
        aggregator_fn: Callable[[List[ExecutionResult]], Any]
    ):
        """注册自定义聚合函数"""
        self._custom_aggregators[name] = aggregator_fn

    def aggregate(
        self,
        results: List[ExecutionResult],
        method: AggregationMethod = AggregationMethod.SIMPLE_AVERAGE,
        weights: Optional[Dict[str, float]] = None,
        custom_fn: Optional[str] = None,
    ) -> AggregatedResult:
        """
        聚合执行结果

        Args:
            results: 执行结果列表
            method: 聚合方法
            weights: 权重字典 (task_id -> weight)
            custom_fn: 自定义聚合函数名称

        Returns:
            AggregatedResult: 聚合后的结果
        """
        self._aggregation_counter += 1
        agg_id = f"agg_{self._aggregation_counter:04d}"

        # 过滤成功的结果
        successful = [r for r in results if r.success]

        if not successful:
            return AggregatedResult(
                aggregation_id=agg_id,
                method=method,
                value=None,
                confidence=0.0,
                source_count=0,
                metadata={"error": "No successful results to aggregate"},
            )

        # 根据方法聚合
        if method == AggregationMethod.CUSTOM and custom_fn:
            if custom_fn in self._custom_aggregators:
                value = self._custom_aggregators[custom_fn](successful)
            else:
                value = None
        elif method == AggregationMethod.SIMPLE_AVERAGE:
            value = self._simple_average(successful)
        elif method == AggregationMethod.WEIGHTED_AVERAGE:
            value = self._weighted_average(successful, weights or {})
        elif method == AggregationMethod.VOTING:
            value = self._voting(successful)
        elif method == AggregationMethod.CONSENSUS:
            value = self._consensus(successful)
        elif method == AggregationMethod.PRIORITY:
            value = self._priority_based(successful)
        else:
            value = self._simple_average(successful)

        # 计算置信度
        confidence = len(successful) / len(results) if results else 0.0

        return AggregatedResult(
            aggregation_id=agg_id,
            method=method,
            value=value,
            confidence=confidence,
            source_count=len(successful),
            source_results=[r.task_id for r in successful],
        )

    def _simple_average(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """简单平均聚合"""
        aggregated = {}

        # 收集所有数值型输出
        numeric_outputs = {}
        for result in results:
            for key, value in result.output.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_outputs:
                        numeric_outputs[key] = []
                    numeric_outputs[key].append(value)
                elif key not in aggregated:
                    aggregated[key] = value

        # 计算平均值
        for key, values in numeric_outputs.items():
            aggregated[key] = np.mean(values)

        return aggregated

    def _weighted_average(
        self,
        results: List[ExecutionResult],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """加权平均聚合"""
        aggregated = {}
        numeric_outputs = {}

        for result in results:
            weight = weights.get(result.task_id, 1.0)
            for key, value in result.output.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_outputs:
                        numeric_outputs[key] = {"values": [], "weights": []}
                    numeric_outputs[key]["values"].append(value)
                    numeric_outputs[key]["weights"].append(weight)
                elif key not in aggregated:
                    aggregated[key] = value

        # 计算加权平均
        for key, data in numeric_outputs.items():
            total_weight = sum(data["weights"])
            if total_weight > 0:
                aggregated[key] = sum(
                    v * w for v, w in zip(data["values"], data["weights"])
                ) / total_weight

        return aggregated

    def _voting(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """投票法聚合"""
        from collections import Counter

        aggregated = {}
        votes = {}

        for result in results:
            for key, value in result.output.items():
                if isinstance(value, (str, int, bool)):
                    if key not in votes:
                        votes[key] = []
                    votes[key].append(value)
                elif key not in aggregated:
                    aggregated[key] = value

        # 选择多数票
        for key, values in votes.items():
            counter = Counter(values)
            most_common = counter.most_common(1)
            if most_common:
                aggregated[key] = most_common[0][0]

        return aggregated

    def _consensus(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """共识法聚合 - 只保留所有结果一致的值"""
        if not results:
            return {}

        # 使用第一个结果作为基准
        base = results[0].output.copy()

        # 检查其他结果是否一致
        for result in results[1:]:
            keys_to_remove = []
            for key, value in base.items():
                if key not in result.output or result.output[key] != value:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del base[key]

        return base

    def _priority_based(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """优先级法聚合 - 使用第一个成功结果"""
        if results:
            return results[0].output.copy()
        return {}


class TradingAggregator(Aggregator):
    """
    交易结果聚合器

    专门用于聚合交易相关的结果。
    """

    def __init__(self):
        super().__init__()
        # 注册交易特化的聚合函数
        self.register_aggregator("signal_consensus", self._aggregate_signals)
        self.register_aggregator("indicator_combine", self._combine_indicators)
        self.register_aggregator("backtest_summary", self._summarize_backtests)

    def aggregate_signals(
        self,
        signal_results: List[ExecutionResult],
        min_agreement: float = 0.6
    ) -> AggregatedResult:
        """
        聚合交易信号

        Args:
            signal_results: 信号结果列表
            min_agreement: 最小一致性要求

        Returns:
            聚合后的交易信号
        """
        result = self.aggregate(
            signal_results,
            method=AggregationMethod.CUSTOM,
            custom_fn="signal_consensus",
        )
        result.metadata["min_agreement"] = min_agreement
        return result

    def aggregate_indicators(
        self,
        indicator_results: List[ExecutionResult]
    ) -> AggregatedResult:
        """聚合指标结果"""
        return self.aggregate(
            indicator_results,
            method=AggregationMethod.CUSTOM,
            custom_fn="indicator_combine",
        )

    def aggregate_backtests(
        self,
        backtest_results: List[ExecutionResult]
    ) -> AggregatedResult:
        """聚合回测结果"""
        return self.aggregate(
            backtest_results,
            method=AggregationMethod.CUSTOM,
            custom_fn="backtest_summary",
        )

    def _aggregate_signals(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """聚合交易信号"""
        signals = []
        for result in results:
            raw_signals = result.output.get("raw_signals", [])
            signals.extend(raw_signals)

        if not signals:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "strength": 0.0,
                "reasons": [],
            }

        # 统计信号类型
        signal_types = [s.get("type", "HOLD") for s in signals]
        from collections import Counter
        type_counts = Counter(signal_types)

        # 确定主导信号
        most_common_type, count = type_counts.most_common(1)[0]
        agreement = count / len(signals)

        # 计算平均置信度和强度
        confidences = [s.get("confidence", 0.5) for s in signals]
        strengths = [s.get("strength", 0.5) for s in signals]

        return {
            "signal": most_common_type,
            "confidence": np.mean(confidences) * agreement,
            "strength": np.mean(strengths),
            "agreement": agreement,
            "signal_distribution": dict(type_counts),
            "reasons": [s.get("reason", "") for s in signals if s.get("reason")],
        }

    def _combine_indicators(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """组合多个指标结果"""
        combined = {
            "indicators": {},
            "summary": {
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
            }
        }

        for result in results:
            for key, value in result.output.items():
                if "_result" in key:
                    indicator_name = key.replace("_result", "")
                    combined["indicators"][indicator_name] = value

                    # 统计方向
                    if isinstance(value, dict):
                        signal = value.get("signal", 0)
                        if signal > 0:
                            combined["summary"]["bullish_count"] += 1
                        elif signal < 0:
                            combined["summary"]["bearish_count"] += 1
                        else:
                            combined["summary"]["neutral_count"] += 1

        # 计算综合方向
        bullish = combined["summary"]["bullish_count"]
        bearish = combined["summary"]["bearish_count"]

        if bullish > bearish:
            combined["summary"]["overall_bias"] = "BULLISH"
        elif bearish > bullish:
            combined["summary"]["overall_bias"] = "BEARISH"
        else:
            combined["summary"]["overall_bias"] = "NEUTRAL"

        return combined

    def _summarize_backtests(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """汇总多个回测结果"""
        metrics_list = []

        for result in results:
            metrics = result.output.get("backtest_results", result.output.get("metrics", {}))
            if metrics:
                metrics_list.append(metrics)

        if not metrics_list:
            return {"error": "No backtest results to summarize"}

        # 计算汇总统计
        summary = {
            "backtest_count": len(metrics_list),
            "avg_return": np.mean([m.get("total_return", 0) for m in metrics_list]),
            "avg_sharpe": np.mean([m.get("sharpe_ratio", 0) for m in metrics_list]),
            "avg_max_drawdown": np.mean([m.get("max_drawdown", 0) for m in metrics_list]),
            "avg_win_rate": np.mean([m.get("win_rate", 0) for m in metrics_list]),
            "best_return": max(m.get("total_return", 0) for m in metrics_list),
            "worst_return": min(m.get("total_return", 0) for m in metrics_list),
            "individual_results": metrics_list,
        }

        return summary
