"""
Signal Validator - 对比 TradingView 和 Python 交易信号
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalValidationResult:
    """信号验证结果"""
    strategy_name: str
    total_bars: int
    tv_long_signals: int
    py_long_signals: int
    tv_short_signals: int
    py_short_signals: int
    matching_long: int
    matching_short: int
    long_accuracy: float
    short_accuracy: float
    overall_accuracy: float
    passed: bool
    mismatches: Optional[pd.DataFrame] = None


class SignalValidator:
    """
    交易信号验证器 - 对比 TradingView 和 Python 的入场/出场信号
    """

    def __init__(self, accuracy_threshold: float = 0.95):
        """
        Args:
            accuracy_threshold: 信号匹配准确率阈值 (默认 95%)
        """
        self.accuracy_threshold = accuracy_threshold
        self.results: list[SignalValidationResult] = []

    def validate_signals(
        self,
        strategy_name: str,
        tv_long_signals: pd.Series,
        tv_short_signals: pd.Series,
        py_long_signals: pd.Series,
        py_short_signals: pd.Series,
    ) -> SignalValidationResult:
        """
        验证交易信号是否一致

        Args:
            strategy_name: 策略名称
            tv_long_signals: TradingView 多头信号 (布尔序列)
            tv_short_signals: TradingView 空头/平仓信号 (布尔序列)
            py_long_signals: Python 多头信号 (布尔序列)
            py_short_signals: Python 空头/平仓信号 (布尔序列)

        Returns:
            SignalValidationResult 验证结果
        """
        # 对齐数据
        min_len = min(len(tv_long_signals), len(py_long_signals))

        tv_long = tv_long_signals.iloc[:min_len].astype(bool).values
        tv_short = tv_short_signals.iloc[:min_len].astype(bool).values
        py_long = py_long_signals.iloc[:min_len].astype(bool).values
        py_short = py_short_signals.iloc[:min_len].astype(bool).values

        # 统计信号数量
        tv_long_count = int(np.sum(tv_long))
        py_long_count = int(np.sum(py_long))
        tv_short_count = int(np.sum(tv_short))
        py_short_count = int(np.sum(py_short))

        # 匹配信号
        matching_long = int(np.sum(tv_long & py_long))
        matching_short = int(np.sum(tv_short & py_short))

        # 计算准确率
        if tv_long_count > 0:
            long_accuracy = matching_long / tv_long_count
        else:
            long_accuracy = 1.0 if py_long_count == 0 else 0.0

        if tv_short_count > 0:
            short_accuracy = matching_short / tv_short_count
        else:
            short_accuracy = 1.0 if py_short_count == 0 else 0.0

        total_tv_signals = tv_long_count + tv_short_count
        if total_tv_signals > 0:
            overall_accuracy = (matching_long + matching_short) / total_tv_signals
        else:
            overall_accuracy = 1.0

        passed = overall_accuracy >= self.accuracy_threshold

        # 找出不匹配的位置
        long_mismatch = tv_long != py_long
        short_mismatch = tv_short != py_short
        any_mismatch = long_mismatch | short_mismatch

        mismatch_indices = np.where(any_mismatch)[0]
        if len(mismatch_indices) > 0:
            mismatches = pd.DataFrame({
                "bar_index": mismatch_indices,
                "tv_long": tv_long[mismatch_indices],
                "py_long": py_long[mismatch_indices],
                "tv_short": tv_short[mismatch_indices],
                "py_short": py_short[mismatch_indices],
            })
        else:
            mismatches = None

        result = SignalValidationResult(
            strategy_name=strategy_name,
            total_bars=min_len,
            tv_long_signals=tv_long_count,
            py_long_signals=py_long_count,
            tv_short_signals=tv_short_count,
            py_short_signals=py_short_count,
            matching_long=matching_long,
            matching_short=matching_short,
            long_accuracy=long_accuracy,
            short_accuracy=short_accuracy,
            overall_accuracy=overall_accuracy,
            passed=passed,
            mismatches=mismatches,
        )

        self.results.append(result)
        return result

    def validate_from_dataframe(
        self,
        strategy_name: str,
        tv_df: pd.DataFrame,
        py_df: pd.DataFrame,
        long_col: str = "long_signal",
        short_col: str = "short_signal",
    ) -> SignalValidationResult:
        """
        从 DataFrame 验证信号

        Args:
            strategy_name: 策略名称
            tv_df: TradingView 导出的数据
            py_df: Python 生成的数据
            long_col: 多头信号列名
            short_col: 空头信号列名
        """
        return self.validate_signals(
            strategy_name,
            tv_df[long_col],
            tv_df[short_col],
            py_df[long_col],
            py_df[short_col],
        )

    def generate_report(self) -> str:
        """生成验证报告"""
        lines = [
            "=" * 60,
            "TradingView vs Python 信号验证报告",
            "=" * 60,
            "",
        ]

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"{status} {result.strategy_name}")
            lines.append(f"  总K线数: {result.total_bars}")
            lines.append(f"  TradingView 多头信号: {result.tv_long_signals}")
            lines.append(f"  Python 多头信号: {result.py_long_signals}")
            lines.append(f"  匹配多头信号: {result.matching_long} ({result.long_accuracy*100:.2f}%)")
            lines.append(f"  TradingView 空头信号: {result.tv_short_signals}")
            lines.append(f"  Python 空头信号: {result.py_short_signals}")
            lines.append(f"  匹配空头信号: {result.matching_short} ({result.short_accuracy*100:.2f}%)")
            lines.append(f"  整体准确率: {result.overall_accuracy*100:.2f}%")
            lines.append(f"  阈值: {self.accuracy_threshold*100:.0f}%")

            if result.mismatches is not None and len(result.mismatches) > 0:
                lines.append(f"  不匹配数量: {len(result.mismatches)}")
                if len(result.mismatches) <= 10:
                    lines.append("  不匹配位置:")
                    for _, row in result.mismatches.iterrows():
                        lines.append(f"    Bar {int(row['bar_index'])}: TV(L={row['tv_long']}, S={row['tv_short']}) vs PY(L={row['py_long']}, S={row['py_short']})")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"总结: {passed_count}/{total_count} 策略通过验证")
        lines.append("=" * 60)

        return "\n".join(lines)
