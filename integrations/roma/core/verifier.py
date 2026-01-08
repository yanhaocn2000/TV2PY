"""
ROMA Verifier - 验证组件

验证执行结果的正确性和一致性。
支持:
    - 输出格式验证
    - 数值范围检查
    - 一致性验证
    - 业务规则验证
    - 自定义验证器
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

from .executor import ExecutionResult, ExecutionStatus, PlanExecutionResult
from .aggregator import AggregatedResult


class VerificationStatus(Enum):
    """验证状态"""
    PASSED = "passed"           # 通过
    FAILED = "failed"           # 失败
    WARNING = "warning"         # 警告
    SKIPPED = "skipped"         # 跳过


@dataclass
class VerificationRule:
    """验证规则"""
    name: str
    description: str
    check_fn: Callable[[Any], bool]
    severity: str = "error"  # error, warning, info
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """
    验证结果
    """
    verification_id: str
    status: VerificationStatus
    passed_rules: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 0.0  # 验证分数 0-1
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.status in [VerificationStatus.PASSED, VerificationStatus.WARNING]


class Verifier:
    """
    验证器

    验证执行结果的正确性。
    """

    def __init__(self):
        self._verification_counter = 0
        self._rules: Dict[str, VerificationRule] = {}

    def add_rule(self, rule: VerificationRule):
        """添加验证规则"""
        self._rules[rule.name] = rule

    def remove_rule(self, rule_name: str):
        """移除验证规则"""
        if rule_name in self._rules:
            del self._rules[rule_name]

    def verify(
        self,
        result: Union[ExecutionResult, PlanExecutionResult, AggregatedResult],
        rules: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        验证结果

        Args:
            result: 要验证的结果
            rules: 要应用的规则名称列表 (None 表示所有规则)

        Returns:
            VerificationResult: 验证结果
        """
        self._verification_counter += 1
        ver_id = f"ver_{self._verification_counter:04d}"

        # 确定要应用的规则
        rules_to_apply = (
            [self._rules[r] for r in rules if r in self._rules]
            if rules
            else list(self._rules.values())
        )

        if not rules_to_apply:
            return VerificationResult(
                verification_id=ver_id,
                status=VerificationStatus.SKIPPED,
                score=1.0,
                details={"reason": "No rules to apply"},
            )

        # 获取要验证的数据
        data = self._extract_data(result)

        # 应用规则
        passed = []
        failed = []
        warnings = []

        for rule in rules_to_apply:
            try:
                if rule.check_fn(data):
                    passed.append(rule.name)
                else:
                    if rule.severity == "warning":
                        warnings.append(f"{rule.name}: {rule.description}")
                    else:
                        failed.append(f"{rule.name}: {rule.description}")
            except Exception as e:
                warnings.append(f"{rule.name}: Check failed with error: {e}")

        # 计算分数
        total = len(rules_to_apply)
        score = len(passed) / total if total > 0 else 0.0

        # 确定状态
        if not failed:
            if warnings:
                status = VerificationStatus.WARNING
            else:
                status = VerificationStatus.PASSED
        else:
            status = VerificationStatus.FAILED

        return VerificationResult(
            verification_id=ver_id,
            status=status,
            passed_rules=passed,
            failed_rules=failed,
            warnings=warnings,
            score=score,
            details={
                "total_rules": total,
                "passed_count": len(passed),
                "failed_count": len(failed),
                "warning_count": len(warnings),
            },
        )

    def _extract_data(
        self,
        result: Union[ExecutionResult, PlanExecutionResult, AggregatedResult]
    ) -> Dict[str, Any]:
        """从结果中提取数据"""
        if isinstance(result, ExecutionResult):
            return {
                "status": result.status,
                "output": result.output,
                "success": result.success,
                "duration": result.duration,
                **result.output,
            }
        elif isinstance(result, PlanExecutionResult):
            return {
                "status": result.status,
                "success_rate": result.success_rate,
                "total_duration": result.total_duration,
                "results": {k: v.output for k, v in result.results.items()},
            }
        elif isinstance(result, AggregatedResult):
            return {
                "method": result.method,
                "value": result.value,
                "confidence": result.confidence,
                "source_count": result.source_count,
                **(result.value if isinstance(result.value, dict) else {}),
            }
        else:
            return {"raw": result}


class TradingVerifier(Verifier):
    """
    交易验证器

    专门用于验证交易相关结果。
    """

    def __init__(self):
        super().__init__()
        self._add_default_rules()

    def _add_default_rules(self):
        """添加默认的交易验证规则"""

        # 数据完整性规则
        self.add_rule(VerificationRule(
            name="data_not_empty",
            description="Data should not be empty",
            check_fn=lambda d: bool(d.get("output") or d.get("value")),
            severity="error",
        ))

        self.add_rule(VerificationRule(
            name="execution_success",
            description="Execution should be successful",
            check_fn=lambda d: d.get("success", True) or d.get("status") in [
                ExecutionStatus.COMPLETED,
                VerificationStatus.PASSED,
            ],
            severity="error",
        ))

        # 信号验证规则
        self.add_rule(VerificationRule(
            name="signal_valid_type",
            description="Signal type should be valid (LONG/SHORT/HOLD/CLOSE)",
            check_fn=lambda d: d.get("signal") in ["LONG", "SHORT", "HOLD", "CLOSE_LONG", "CLOSE_SHORT", None],
            severity="error",
        ))

        self.add_rule(VerificationRule(
            name="confidence_range",
            description="Confidence should be between 0 and 1",
            check_fn=lambda d: 0 <= d.get("confidence", 0.5) <= 1,
            severity="error",
        ))

        self.add_rule(VerificationRule(
            name="strength_range",
            description="Strength should be between 0 and 1",
            check_fn=lambda d: 0 <= d.get("strength", 0.5) <= 1,
            severity="error",
        ))

        # 回测验证规则
        self.add_rule(VerificationRule(
            name="positive_sharpe",
            description="Sharpe ratio should be positive for profitable strategy",
            check_fn=lambda d: d.get("sharpe_ratio", 0) > 0 if d.get("total_return", 0) > 0 else True,
            severity="warning",
        ))

        self.add_rule(VerificationRule(
            name="reasonable_drawdown",
            description="Max drawdown should be less than 50%",
            check_fn=lambda d: d.get("max_drawdown", 0) < 0.5,
            severity="warning",
        ))

        self.add_rule(VerificationRule(
            name="reasonable_win_rate",
            description="Win rate should be between 0 and 1",
            check_fn=lambda d: 0 <= d.get("win_rate", 0.5) <= 1,
            severity="error",
        ))

        # 指标验证规则
        self.add_rule(VerificationRule(
            name="indicator_calculated",
            description="Indicator should be calculated",
            check_fn=lambda d: d.get("calculated", True),
            severity="error",
        ))

    def verify_signal(self, signal: Dict[str, Any]) -> VerificationResult:
        """验证交易信号"""
        # 创建模拟执行结果
        result = type('Result', (), {'output': signal, 'success': True, 'status': ExecutionStatus.COMPLETED, 'duration': 0})()
        return self.verify(
            result,
            rules=["signal_valid_type", "confidence_range", "strength_range"],
        )

    def verify_backtest(self, backtest_result: Dict[str, Any]) -> VerificationResult:
        """验证回测结果"""
        result = type('Result', (), {'output': backtest_result, 'success': True, 'status': ExecutionStatus.COMPLETED, 'duration': 0})()
        return self.verify(
            result,
            rules=["positive_sharpe", "reasonable_drawdown", "reasonable_win_rate"],
        )

    def verify_indicator(self, indicator_result: Dict[str, Any]) -> VerificationResult:
        """验证指标结果"""
        result = type('Result', (), {'output': indicator_result, 'success': True, 'status': ExecutionStatus.COMPLETED, 'duration': 0})()
        return self.verify(
            result,
            rules=["indicator_calculated", "data_not_empty"],
        )

    def comprehensive_verify(
        self,
        plan_result: PlanExecutionResult,
    ) -> Dict[str, VerificationResult]:
        """
        全面验证计划执行结果

        Returns:
            Dict[task_id, VerificationResult]
        """
        verification_results = {}

        for task_id, exec_result in plan_result.results.items():
            ver_result = self.verify(exec_result)
            verification_results[task_id] = ver_result

        return verification_results
