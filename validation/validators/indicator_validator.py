"""
Indicator Validator - 对比 TradingView 和 Python 指标计算结果
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """验证结果"""
    indicator_name: str
    total_bars: int
    matching_bars: int
    max_deviation: float
    mean_deviation: float
    correlation: float
    passed: bool
    tolerance: float
    details: Optional[pd.DataFrame] = None


class IndicatorValidator:
    """
    指标验证器 - 对比 TradingView 导出数据与 Python 计算结果
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Args:
            tolerance: 允许的最大误差 (默认 0.000001)
        """
        self.tolerance = tolerance
        self.results: list[ValidationResult] = []

    def validate_series(
        self,
        name: str,
        tradingview_values: pd.Series,
        python_values: pd.Series,
        tolerance: Optional[float] = None,
    ) -> ValidationResult:
        """
        验证两个序列是否一致

        Args:
            name: 指标名称
            tradingview_values: TradingView 导出的指标值
            python_values: Python 计算的指标值
            tolerance: 自定义容差

        Returns:
            ValidationResult 验证结果
        """
        tol = tolerance or self.tolerance

        # 对齐数据
        tv = tradingview_values.dropna().reset_index(drop=True)
        py = python_values.dropna().reset_index(drop=True)

        # 取共同长度
        min_len = min(len(tv), len(py))
        tv = tv.iloc[:min_len]
        py = py.iloc[:min_len]

        # 计算偏差
        deviation = np.abs(tv.values - py.values)
        max_dev = float(np.max(deviation))
        mean_dev = float(np.mean(deviation))

        # 计算相关性
        if len(tv) > 1:
            correlation = float(np.corrcoef(tv.values, py.values)[0, 1])
        else:
            correlation = 1.0

        # 判断是否通过
        matching = int(np.sum(deviation <= tol))
        passed = max_dev <= tol

        # 详细对比
        details = pd.DataFrame({
            "tradingview": tv.values,
            "python": py.values,
            "deviation": deviation,
            "match": deviation <= tol,
        })

        result = ValidationResult(
            indicator_name=name,
            total_bars=min_len,
            matching_bars=matching,
            max_deviation=max_dev,
            mean_deviation=mean_dev,
            correlation=correlation,
            passed=passed,
            tolerance=tol,
            details=details,
        )

        self.results.append(result)
        return result

    def validate_ema(
        self,
        close: pd.Series,
        tradingview_ema: pd.Series,
        length: int,
    ) -> ValidationResult:
        """验证 EMA 计算"""
        # Python EMA 计算 (与 TradingView 一致)
        python_ema = self._calculate_ema(close, length)
        return self.validate_series(f"EMA({length})", tradingview_ema, python_ema)

    def validate_sma(
        self,
        close: pd.Series,
        tradingview_sma: pd.Series,
        length: int,
    ) -> ValidationResult:
        """验证 SMA 计算"""
        python_sma = close.rolling(window=length).mean()
        return self.validate_series(f"SMA({length})", tradingview_sma, python_sma)

    def validate_rsi(
        self,
        close: pd.Series,
        tradingview_rsi: pd.Series,
        length: int = 14,
    ) -> ValidationResult:
        """验证 RSI 计算"""
        python_rsi = self._calculate_rsi(close, length)
        return self.validate_series(f"RSI({length})", tradingview_rsi, python_rsi, tolerance=0.01)

    def validate_macd(
        self,
        close: pd.Series,
        tradingview_macd: pd.Series,
        tradingview_signal: pd.Series,
        tradingview_hist: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> list[ValidationResult]:
        """验证 MACD 计算"""
        python_macd, python_signal, python_hist = self._calculate_macd(
            close, fast, slow, signal
        )

        results = [
            self.validate_series(f"MACD({fast},{slow})", tradingview_macd, python_macd, tolerance=0.01),
            self.validate_series(f"MACD_Signal({signal})", tradingview_signal, python_signal, tolerance=0.01),
            self.validate_series("MACD_Hist", tradingview_hist, python_hist, tolerance=0.01),
        ]
        return results

    def validate_bollinger_bands(
        self,
        close: pd.Series,
        tradingview_upper: pd.Series,
        tradingview_middle: pd.Series,
        tradingview_lower: pd.Series,
        length: int = 20,
        mult: float = 2.0,
    ) -> list[ValidationResult]:
        """验证布林带计算"""
        middle = close.rolling(window=length).mean()
        std = close.rolling(window=length).std()
        upper = middle + mult * std
        lower = middle - mult * std

        results = [
            self.validate_series(f"BB_Upper({length},{mult})", tradingview_upper, upper, tolerance=0.01),
            self.validate_series(f"BB_Middle({length})", tradingview_middle, middle, tolerance=0.01),
            self.validate_series(f"BB_Lower({length},{mult})", tradingview_lower, lower, tolerance=0.01),
        ]
        return results

    def _calculate_ema(self, series: pd.Series, length: int) -> pd.Series:
        """
        计算 EMA (与 TradingView ta.ema 一致)
        TradingView 使用 RMA 方式: alpha = 1/length 对于 RMA, 2/(length+1) 对于 EMA
        """
        alpha = 2.0 / (length + 1)
        return series.ewm(alpha=alpha, adjust=False).mean()

    def _calculate_rsi(self, close: pd.Series, length: int) -> pd.Series:
        """
        计算 RSI (与 TradingView ta.rsi 一致)
        TradingView 使用 RMA (Wilder's smoothing)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # RMA (Wilder's smoothing) alpha = 1/length
        alpha = 1.0 / length
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, close: pd.Series, fast: int, slow: int, signal: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """计算 MACD"""
        fast_ema = self._calculate_ema(close, fast)
        slow_ema = self._calculate_ema(close, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_report(self) -> str:
        """生成验证报告"""
        lines = [
            "=" * 60,
            "TradingView vs Python 指标验证报告",
            "=" * 60,
            "",
        ]

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"{status} {result.indicator_name}")
            lines.append(f"  总条数: {result.total_bars}")
            lines.append(f"  匹配条数: {result.matching_bars} ({result.matching_bars/result.total_bars*100:.2f}%)")
            lines.append(f"  最大偏差: {result.max_deviation:.8f}")
            lines.append(f"  平均偏差: {result.mean_deviation:.8f}")
            lines.append(f"  相关系数: {result.correlation:.6f}")
            lines.append(f"  容差阈值: {result.tolerance}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"总结: {passed_count}/{total_count} 指标通过验证")
        lines.append("=" * 60)

        return "\n".join(lines)

    def export_details(self, output_path: str) -> None:
        """导出详细对比数据到 CSV"""
        for result in self.results:
            if result.details is not None:
                filename = f"{output_path}/{result.indicator_name.replace('(', '_').replace(')', '_').replace(',', '_')}.csv"
                result.details.to_csv(filename, index=False)
