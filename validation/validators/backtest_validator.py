"""
Backtest Validator - 对比 TradingView 和 Python 回测结果
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeRecord:
    """交易记录"""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: str  # "long" or "short"
    pnl: float
    pnl_percent: float


@dataclass
class BacktestValidationResult:
    """回测验证结果"""
    strategy_name: str
    # 交易统计
    tv_total_trades: int
    py_total_trades: int
    matching_trades: int
    trade_match_rate: float
    # 收益对比
    tv_total_pnl: float
    py_total_pnl: float
    pnl_deviation: float
    pnl_deviation_percent: float
    # 胜率对比
    tv_win_rate: float
    py_win_rate: float
    win_rate_deviation: float
    # 最大回撤对比
    tv_max_drawdown: float
    py_max_drawdown: float
    drawdown_deviation: float
    # 验证结果
    passed: bool
    details: Optional[str] = None


class BacktestValidator:
    """
    回测结果验证器 - 对比 TradingView 和 Python 的回测表现
    """

    def __init__(
        self,
        pnl_tolerance: float = 0.01,  # 1% 收益误差
        trade_match_threshold: float = 0.95,  # 95% 交易匹配率
    ):
        self.pnl_tolerance = pnl_tolerance
        self.trade_match_threshold = trade_match_threshold
        self.results: list[BacktestValidationResult] = []

    def validate_trades(
        self,
        strategy_name: str,
        tv_trades: list[TradeRecord],
        py_trades: list[TradeRecord],
        bar_tolerance: int = 1,  # 允许的K线偏差
    ) -> BacktestValidationResult:
        """
        验证交易记录是否一致

        Args:
            strategy_name: 策略名称
            tv_trades: TradingView 交易记录
            py_trades: Python 交易记录
            bar_tolerance: 允许的入场/出场K线偏差
        """
        tv_count = len(tv_trades)
        py_count = len(py_trades)

        # 匹配交易
        matched = 0
        for tv_trade in tv_trades:
            for py_trade in py_trades:
                if (
                    abs(tv_trade.entry_bar - py_trade.entry_bar) <= bar_tolerance
                    and abs(tv_trade.exit_bar - py_trade.exit_bar) <= bar_tolerance
                    and tv_trade.direction == py_trade.direction
                ):
                    matched += 1
                    break

        trade_match_rate = matched / tv_count if tv_count > 0 else 1.0

        # 计算总收益
        tv_total_pnl = sum(t.pnl_percent for t in tv_trades)
        py_total_pnl = sum(t.pnl_percent for t in py_trades)
        pnl_deviation = abs(tv_total_pnl - py_total_pnl)
        pnl_deviation_percent = pnl_deviation / abs(tv_total_pnl) if tv_total_pnl != 0 else 0

        # 计算胜率
        tv_wins = sum(1 for t in tv_trades if t.pnl > 0)
        py_wins = sum(1 for t in py_trades if t.pnl > 0)
        tv_win_rate = tv_wins / tv_count if tv_count > 0 else 0
        py_win_rate = py_wins / py_count if py_count > 0 else 0
        win_rate_deviation = abs(tv_win_rate - py_win_rate)

        # 计算最大回撤 (简化版)
        tv_max_dd = self._calculate_max_drawdown([t.pnl_percent for t in tv_trades])
        py_max_dd = self._calculate_max_drawdown([t.pnl_percent for t in py_trades])
        drawdown_deviation = abs(tv_max_dd - py_max_dd)

        # 判断是否通过
        passed = (
            trade_match_rate >= self.trade_match_threshold
            and pnl_deviation_percent <= self.pnl_tolerance
        )

        result = BacktestValidationResult(
            strategy_name=strategy_name,
            tv_total_trades=tv_count,
            py_total_trades=py_count,
            matching_trades=matched,
            trade_match_rate=trade_match_rate,
            tv_total_pnl=tv_total_pnl,
            py_total_pnl=py_total_pnl,
            pnl_deviation=pnl_deviation,
            pnl_deviation_percent=pnl_deviation_percent,
            tv_win_rate=tv_win_rate,
            py_win_rate=py_win_rate,
            win_rate_deviation=win_rate_deviation,
            tv_max_drawdown=tv_max_dd,
            py_max_drawdown=py_max_dd,
            drawdown_deviation=drawdown_deviation,
            passed=passed,
        )

        self.results.append(result)
        return result

    def validate_from_csv(
        self,
        strategy_name: str,
        tv_trades_csv: str,
        py_trades_csv: str,
    ) -> BacktestValidationResult:
        """
        从 CSV 文件验证交易记录

        CSV 格式应包含:
        entry_bar, exit_bar, entry_price, exit_price, direction, pnl, pnl_percent
        """
        tv_df = pd.read_csv(tv_trades_csv)
        py_df = pd.read_csv(py_trades_csv)

        tv_trades = [
            TradeRecord(**row) for _, row in tv_df.iterrows()
        ]
        py_trades = [
            TradeRecord(**row) for _, row in py_df.iterrows()
        ]

        return self.validate_trades(strategy_name, tv_trades, py_trades)

    def _calculate_max_drawdown(self, pnl_list: list[float]) -> float:
        """计算最大回撤"""
        if not pnl_list:
            return 0.0

        cumulative = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    def generate_report(self) -> str:
        """生成验证报告"""
        lines = [
            "=" * 60,
            "TradingView vs Python 回测验证报告",
            "=" * 60,
            "",
        ]

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"{status} {result.strategy_name}")
            lines.append("")
            lines.append("  交易统计:")
            lines.append(f"    TradingView 交易数: {result.tv_total_trades}")
            lines.append(f"    Python 交易数: {result.py_total_trades}")
            lines.append(f"    匹配交易数: {result.matching_trades}")
            lines.append(f"    匹配率: {result.trade_match_rate*100:.2f}%")
            lines.append("")
            lines.append("  收益对比:")
            lines.append(f"    TradingView 总收益: {result.tv_total_pnl:.2f}%")
            lines.append(f"    Python 总收益: {result.py_total_pnl:.2f}%")
            lines.append(f"    偏差: {result.pnl_deviation:.2f}% ({result.pnl_deviation_percent*100:.2f}%)")
            lines.append("")
            lines.append("  胜率对比:")
            lines.append(f"    TradingView 胜率: {result.tv_win_rate*100:.2f}%")
            lines.append(f"    Python 胜率: {result.py_win_rate*100:.2f}%")
            lines.append(f"    偏差: {result.win_rate_deviation*100:.2f}%")
            lines.append("")
            lines.append("  最大回撤对比:")
            lines.append(f"    TradingView: {result.tv_max_drawdown:.2f}%")
            lines.append(f"    Python: {result.py_max_drawdown:.2f}%")
            lines.append(f"    偏差: {result.drawdown_deviation:.2f}%")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"总结: {passed_count}/{total_count} 策略通过验证")
        lines.append("=" * 60)

        return "\n".join(lines)
