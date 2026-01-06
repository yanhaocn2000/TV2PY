"""
Keltner Channel 策略验证测试

对比 Python 实现与 TradingView 参考值的精度
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Optional
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.keltner_channel import (
    KeltnerChannelIndicator,
    KeltnerChannelStrategy,
    KeltnerChannelParams,
    SqueezeType
)


@dataclass
class ValidationResult:
    """验证结果"""
    name: str
    total_points: int
    max_error: float
    mean_error: float
    std_error: float
    max_pct_error: float  # 最大百分比误差
    correlation: float
    passed: bool
    tolerance: float


class TradingViewReference:
    """
    TradingView 参考实现

    这些实现完全按照 TradingView Pine Script 文档来写,
    用于验证我们的策略实现是否正确
    """

    @staticmethod
    def sma(data: np.ndarray, length: int) -> np.ndarray:
        """
        Simple Moving Average - TradingView 实现

        Pine Script:
            ta.sma(source, length) → series float
        """
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            result[i] = np.mean(data[i - length + 1:i + 1])
        return result

    @staticmethod
    def ema(data: np.ndarray, length: int) -> np.ndarray:
        """
        Exponential Moving Average - TradingView 实现

        Pine Script:
            ta.ema(source, length) → series float
            alpha = 2 / (length + 1)
            EMA = alpha * source + (1 - alpha) * EMA[1]
        """
        result = np.full_like(data, np.nan, dtype=float)
        alpha = 2.0 / (length + 1)

        # 第一个有效值使用 SMA 作为种子
        result[length - 1] = np.mean(data[:length])

        for i in range(length, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    def stdev(data: np.ndarray, length: int) -> np.ndarray:
        """
        Standard Deviation - TradingView 实现

        Pine Script:
            ta.stdev(source, length) → series float
            使用 population stdev (ddof=0)
        """
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            result[i] = np.std(window, ddof=0)  # TradingView 使用 population std
        return result

    @staticmethod
    def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        True Range - TradingView 实现

        Pine Script:
            ta.tr(handle_na) → series float
            tr = max(high - low, abs(high - close[1]), abs(low - close[1]))
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        return np.maximum(np.maximum(tr1, tr2), tr3)

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            length: int) -> np.ndarray:
        """
        Average True Range - TradingView 实现

        Pine Script:
            ta.atr(length) → series float
            ATR = RMA(TR, length)  # 使用 RMA 不是 EMA
        """
        tr = TradingViewReference.true_range(high, low, close)
        return TradingViewReference.rma(tr, length)

    @staticmethod
    def rma(data: np.ndarray, length: int) -> np.ndarray:
        """
        Relative Moving Average (Wilder's Smoothing) - TradingView 实现

        Pine Script:
            ta.rma(source, length) → series float
            alpha = 1 / length
            RMA = alpha * source + (1 - alpha) * RMA[1]
        """
        result = np.full_like(data, np.nan, dtype=float)
        alpha = 1.0 / length

        # 第一个有效值使用 SMA 作为种子
        result[length - 1] = np.mean(data[:length])

        for i in range(length, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    def bollinger_bands(close: np.ndarray, length: int = 20,
                        mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bands - TradingView 实现

        Pine Script:
            basis = ta.sma(source, length)
            dev = mult * ta.stdev(source, length)
            upper = basis + dev
            lower = basis - dev
        """
        basis = TradingViewReference.sma(close, length)
        dev = mult * TradingViewReference.stdev(close, length)
        upper = basis + dev
        lower = basis - dev
        return upper, basis, lower

    @staticmethod
    def keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                        length: int = 20, mult: float = 1.5,
                        use_true_range: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Keltner Channel - TradingView 实现

        Pine Script:
            basis = ta.ema(source, length)
            range = useTrueRange ? ta.tr : high - low
            rangema = ta.ema(range, length)  # 注意: TradingView 内置用 EMA
            upper = basis + rangema * mult
            lower = basis - rangema * mult
        """
        basis = TradingViewReference.ema(close, length)

        if use_true_range:
            tr = TradingViewReference.true_range(high, low, close)
        else:
            tr = high - low

        # TradingView Keltner Channel 使用 EMA 计算 ATR
        rangema = TradingViewReference.ema(tr, length)

        upper = basis + rangema * mult
        lower = basis - rangema * mult

        return upper, basis, lower


class KeltnerChannelValidator:
    """Keltner Channel 策略验证器"""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.results: List[ValidationResult] = []

    def validate_indicator(self, name: str, our_values: np.ndarray,
                           ref_values: np.ndarray,
                           tolerance: Optional[float] = None) -> ValidationResult:
        """验证单个指标"""
        tol = tolerance or self.tolerance

        # 只比较有效值
        mask = ~(np.isnan(our_values) | np.isnan(ref_values))
        our_valid = our_values[mask]
        ref_valid = ref_values[mask]

        if len(our_valid) == 0:
            return ValidationResult(
                name=name, total_points=0, max_error=0, mean_error=0,
                std_error=0, max_pct_error=0, correlation=1.0,
                passed=True, tolerance=tol
            )

        # 计算误差
        errors = np.abs(our_valid - ref_valid)
        max_error = float(np.max(errors))
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))

        # 百分比误差 (避免除以零)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = np.abs(errors / ref_valid) * 100
            pct_errors = np.where(np.isfinite(pct_errors), pct_errors, 0)
        max_pct_error = float(np.max(pct_errors))

        # 相关系数
        if len(our_valid) > 1:
            correlation = float(np.corrcoef(our_valid, ref_valid)[0, 1])
        else:
            correlation = 1.0

        passed = max_error <= tol

        result = ValidationResult(
            name=name,
            total_points=len(our_valid),
            max_error=max_error,
            mean_error=mean_error,
            std_error=std_error,
            max_pct_error=max_pct_error,
            correlation=correlation,
            passed=passed,
            tolerance=tol
        )

        self.results.append(result)
        return result

    def run_full_validation(self, df: pd.DataFrame,
                            params: KeltnerChannelParams) -> List[ValidationResult]:
        """运行完整验证"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 创建我们的指标计算器
        our_indicator = KeltnerChannelIndicator(params)

        # 计算我们的值
        our_bb_upper, our_bb_basis, our_bb_lower = our_indicator.calculate_bb(close)
        our_kc_upper, our_kc_basis, our_kc_lower = our_indicator.calculate_kc(high, low, close)

        # 计算参考值
        ref_bb_upper, ref_bb_basis, ref_bb_lower = TradingViewReference.bollinger_bands(
            close, params.length, params.bb_mult
        )
        ref_kc_upper, ref_kc_basis, ref_kc_lower = TradingViewReference.keltner_channel(
            high, low, close, params.length, params.kc_mult, params.use_true_range
        )

        # 验证 Bollinger Bands
        self.validate_indicator("BB_Upper", our_bb_upper, ref_bb_upper, tolerance=0.0001)
        self.validate_indicator("BB_Basis (SMA)", our_bb_basis, ref_bb_basis, tolerance=1e-10)
        self.validate_indicator("BB_Lower", our_bb_lower, ref_bb_lower, tolerance=0.0001)

        # 验证 Keltner Channel
        self.validate_indicator("KC_Upper", our_kc_upper, ref_kc_upper, tolerance=0.0001)
        self.validate_indicator("KC_Basis (EMA)", our_kc_basis, ref_kc_basis, tolerance=1e-10)
        self.validate_indicator("KC_Lower", our_kc_lower, ref_kc_lower, tolerance=0.0001)

        # 验证基础指标
        our_sma = our_indicator._sma(close, params.length)
        ref_sma = TradingViewReference.sma(close, params.length)
        self.validate_indicator("SMA", our_sma, ref_sma, tolerance=1e-10)

        our_ema = our_indicator._ema(close, params.length)
        ref_ema = TradingViewReference.ema(close, params.length)
        self.validate_indicator("EMA", our_ema, ref_ema, tolerance=1e-10)

        our_stdev = our_indicator._stdev(close, params.length)
        ref_stdev = TradingViewReference.stdev(close, params.length)
        self.validate_indicator("STDEV", our_stdev, ref_stdev, tolerance=1e-10)

        our_tr = our_indicator._true_range(high, low, close)
        ref_tr = TradingViewReference.true_range(high, low, close)
        self.validate_indicator("True Range", our_tr, ref_tr, tolerance=1e-10)

        return self.results

    def generate_report(self) -> str:
        """生成验证报告"""
        lines = [
            "=" * 70,
            "Keltner Channel 策略转换精度验证报告",
            "=" * 70,
            "",
        ]

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        # 按类别分组
        bb_results = [r for r in self.results if r.name.startswith("BB")]
        kc_results = [r for r in self.results if r.name.startswith("KC")]
        base_results = [r for r in self.results if not r.name.startswith("BB") and not r.name.startswith("KC")]

        # Bollinger Bands
        lines.append("【Bollinger Bands 验证】")
        lines.append("-" * 50)
        for result in bb_results:
            self._add_result_lines(lines, result)
        lines.append("")

        # Keltner Channel
        lines.append("【Keltner Channel 验证】")
        lines.append("-" * 50)
        for result in kc_results:
            self._add_result_lines(lines, result)
        lines.append("")

        # 基础指标
        lines.append("【基础指标验证】")
        lines.append("-" * 50)
        for result in base_results:
            self._add_result_lines(lines, result)
        lines.append("")

        # 总结
        lines.append("=" * 70)
        lines.append("验证总结")
        lines.append("=" * 70)
        lines.append(f"通过: {passed_count}/{total_count} 指标")
        lines.append(f"通过率: {passed_count/total_count*100:.1f}%")
        lines.append("")

        # 最大误差汇总
        if self.results:
            max_errors = [(r.name, r.max_error, r.max_pct_error) for r in self.results]
            max_errors.sort(key=lambda x: x[1], reverse=True)
            lines.append("最大误差排名:")
            for name, err, pct in max_errors[:5]:
                lines.append(f"  {name}: {err:.2e} ({pct:.6f}%)")

        lines.append("")
        lines.append("结论:")
        if passed_count == total_count:
            lines.append("  ✅ 所有指标验证通过, 转换精度符合要求")
        else:
            failed = [r.name for r in self.results if not r.passed]
            lines.append(f"  ⚠️ 部分指标未通过: {', '.join(failed)}")
            lines.append("  建议检查这些指标的实现")

        return "\n".join(lines)

    def _add_result_lines(self, lines: List[str], result: ValidationResult):
        """添加单个结果的行"""
        status = "✅" if result.passed else "❌"
        lines.append(f"{status} {result.name}")
        lines.append(f"   数据点: {result.total_points}")
        lines.append(f"   最大误差: {result.max_error:.2e}")
        lines.append(f"   平均误差: {result.mean_error:.2e}")
        lines.append(f"   最大百分比误差: {result.max_pct_error:.6f}%")
        lines.append(f"   相关系数: {result.correlation:.10f}")
        lines.append(f"   容差阈值: {result.tolerance}")
        lines.append("")


def generate_test_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(seed)

    # 生成价格走势 (随机游走 + 趋势)
    returns = np.random.randn(n) * 0.02  # 2% 日波动
    trend = np.linspace(0, 0.5, n)  # 上涨趋势
    close = 100 * np.exp(np.cumsum(returns) + trend)

    # 生成 OHLC
    volatility = np.abs(np.random.randn(n) * 0.01) + 0.005
    high = close * (1 + volatility)
    low = close * (1 - volatility)
    open_ = np.roll(close, 1) * (1 + np.random.randn(n) * 0.005)
    open_[0] = close[0]

    # 确保 OHLC 关系正确
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(1000, 100000, n)

    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


def run_validation():
    """运行验证测试"""
    print("=" * 70)
    print("Keltner Channel 策略转换验证")
    print("=" * 70)
    print()

    # 生成测试数据
    print("1. 生成测试数据...")
    df = generate_test_data(500)
    print(f"   数据点: {len(df)}")
    print(f"   价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print()

    # 测试参数
    params = KeltnerChannelParams(
        length=20,
        bb_mult=2.0,
        kc_mult=1.5,
        use_true_range=True
    )

    print(f"2. 测试参数:")
    print(f"   周期: {params.length}")
    print(f"   BB 倍数: {params.bb_mult}")
    print(f"   KC 倍数: {params.kc_mult}")
    print(f"   使用 True Range: {params.use_true_range}")
    print()

    # 运行验证
    print("3. 运行指标验证...")
    validator = KeltnerChannelValidator(tolerance=1e-6)
    results = validator.run_full_validation(df, params)
    print()

    # 生成报告
    print("4. 验证结果:")
    print()
    report = validator.generate_report()
    print(report)

    # 返回结果
    return validator, df, params


def compare_with_tradingview_export(tv_data_path: str):
    """
    与 TradingView 导出数据对比

    使用方法:
    1. 在 TradingView 添加以下指标脚本:
       //@version=5
       indicator("KC Validation", overlay=true)
       [upper, basis, lower] = ta.bb(close, 20, 2)
       [kcUpper, kcBasis, kcLower] = ta.kc(close, 20, 1.5)
       plot(upper, "BB_Upper")
       plot(basis, "BB_Basis")
       plot(lower, "BB_Lower")
       plot(kcUpper, "KC_Upper")
       plot(kcBasis, "KC_Basis")
       plot(kcLower, "KC_Lower")

    2. 导出数据: 右键图表 -> 导出图表数据

    3. 调用此函数进行对比
    """
    print("从 TradingView 导出数据对比...")

    # 读取 TV 数据
    tv_df = pd.read_csv(tv_data_path)

    # 提取 OHLC 和指标
    close = tv_df['close'].values
    high = tv_df['high'].values
    low = tv_df['low'].values

    # 我们的计算
    params = KeltnerChannelParams(length=20, bb_mult=2.0, kc_mult=1.5)
    indicator = KeltnerChannelIndicator(params)

    our_bb_upper, our_bb_basis, our_bb_lower = indicator.calculate_bb(close)
    our_kc_upper, our_kc_basis, our_kc_lower = indicator.calculate_kc(high, low, close)

    # TradingView 导出的值
    tv_bb_upper = tv_df['BB_Upper'].values
    tv_bb_basis = tv_df['BB_Basis'].values
    tv_kc_upper = tv_df['KC_Upper'].values
    tv_kc_basis = tv_df['KC_Basis'].values

    # 验证
    validator = KeltnerChannelValidator(tolerance=0.01)  # 放宽容差
    validator.validate_indicator("BB_Upper (vs TV)", our_bb_upper, tv_bb_upper)
    validator.validate_indicator("BB_Basis (vs TV)", our_bb_basis, tv_bb_basis)
    validator.validate_indicator("KC_Upper (vs TV)", our_kc_upper, tv_kc_upper)
    validator.validate_indicator("KC_Basis (vs TV)", our_kc_basis, tv_kc_basis)

    print(validator.generate_report())


def test_signal_generation():
    """测试信号生成逻辑"""
    print()
    print("=" * 70)
    print("信号生成测试")
    print("=" * 70)
    print()

    # 生成测试数据
    df = generate_test_data(200)

    # 运行策略
    params = KeltnerChannelParams(
        length=20,
        bb_mult=2.0,
        kc_mult=1.5,
        stop_loss_pct=2.0,
        take_profit_pct=4.0
    )

    strategy = KeltnerChannelStrategy(params)
    result = strategy.run(df)

    # 统计信号
    buy_signals = (result['signal'] == 1).sum()
    sell_signals = (result['signal'] == -1).sum()
    close_signals = (result['signal'] == 2).sum()
    squeeze_bars = result['squeeze'].sum()

    print(f"数据点: {len(df)}")
    print(f"Squeeze 状态 bar 数: {squeeze_bars} ({squeeze_bars/len(df)*100:.1f}%)")
    print(f"买入信号: {buy_signals}")
    print(f"卖出信号: {sell_signals}")
    print(f"平仓信号: {close_signals}")
    print()

    # 显示信号详情
    signals = result[result['signal'] != 0][['close', 'signal', 'squeeze', 'bb_basis', 'kc_basis']]
    if len(signals) > 0:
        print("信号详情 (前10个):")
        print(signals.head(10).to_string())
    else:
        print("未产生信号 (可能数据不足或市场处于 squeeze 状态)")

    return result


if __name__ == "__main__":
    # 运行验证
    validator, df, params = run_validation()

    # 测试信号生成
    test_signal_generation()

    print()
    print("=" * 70)
    print("TradingView 导出数据对比说明")
    print("=" * 70)
    print("""
要与真实 TradingView 数据对比, 请执行以下步骤:

1. 在 TradingView Pine Editor 添加脚本:
   //@version=5
   indicator("KC Validation", overlay=true)
   [upper, basis, lower] = ta.bb(close, 20, 2)
   [kcUpper, kcBasis, kcLower] = ta.kc(close, 20, 1.5)
   plot(upper, "BB_Upper")
   plot(basis, "BB_Basis")
   plot(lower, "BB_Lower")
   plot(kcUpper, "KC_Upper")
   plot(kcBasis, "KC_Basis")
   plot(kcLower, "KC_Lower")

2. 右键图表 -> 导出图表数据 -> 保存 CSV

3. 运行对比:
   compare_with_tradingview_export('path/to/tv_export.csv')
""")
