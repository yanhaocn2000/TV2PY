"""
完整指标验证 - 测试所有 TradingView 指标实现

运行: python validation/run_full_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from validation.validators.tv_indicators import TradingViewIndicators as ta, INDICATOR_CHECKLIST
from validation.validators.indicator_validator import IndicatorValidator


def generate_test_data(bars: int = 500) -> pd.DataFrame:
    """生成测试 OHLCV 数据"""
    np.random.seed(42)

    # 更真实的价格生成
    returns = np.random.randn(bars) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.randn(bars)) * 0.015)
    low = close * (1 - np.abs(np.random.randn(bars)) * 0.015)
    open_price = low + (high - low) * np.random.rand(bars)
    volume = np.random.randint(10000, 100000, bars).astype(float)

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class FullValidator:
    """完整指标验证器"""

    def __init__(self):
        self.df = generate_test_data()
        self.results = {}
        self.passed = 0
        self.failed = 0

    def test_indicator(self, name: str, func, *args, tolerance: float = 0.0001) -> bool:
        """测试单个指标"""
        try:
            result = func(*args)

            # 检查结果有效性
            if isinstance(result, tuple):
                for r in result:
                    if r.isna().all():
                        raise ValueError("All NaN values")
            else:
                if result.isna().all():
                    raise ValueError("All NaN values")

            self.results[name] = "✅ PASS"
            self.passed += 1
            return True
        except Exception as e:
            self.results[name] = f"❌ FAIL: {str(e)[:50]}"
            self.failed += 1
            return False

    def run_all_tests(self):
        """运行所有指标测试"""
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        volume = self.df["volume"]

        print("=" * 70)
        print("TradingView 指标完整验证")
        print("=" * 70)
        print()

        # ==================== 移动平均类 ====================
        print("📊 移动平均类指标")
        print("-" * 50)

        self.test_indicator("ta.sma(14)", ta.sma, close, 14)
        self.test_indicator("ta.ema(14)", ta.ema, close, 14)
        self.test_indicator("ta.rma(14)", ta.rma, close, 14)
        self.test_indicator("ta.wma(14)", ta.wma, close, 14)
        self.test_indicator("ta.vwma(14)", ta.vwma, close, volume, 14)
        self.test_indicator("ta.swma()", ta.swma, close)
        self.test_indicator("ta.alma(14)", ta.alma, close, 14)
        self.test_indicator("ta.hma(14)", ta.hma, close, 14)

        for name, status in list(self.results.items())[-8:]:
            print(f"  {status} {name}")
        print()

        # ==================== 动量类 ====================
        print("📈 动量类指标")
        print("-" * 50)

        self.test_indicator("ta.rsi(14)", ta.rsi, close, 14)
        self.test_indicator("ta.macd(12,26,9)", ta.macd, close, 12, 26, 9)
        self.test_indicator("ta.stoch(14,1,3)", ta.stoch, high, low, close, 14, 1, 3)
        self.test_indicator("ta.cci(20)", ta.cci, high, low, close, 20)
        self.test_indicator("ta.mom(10)", ta.mom, close, 10)
        self.test_indicator("ta.roc(10)", ta.roc, close, 10)
        self.test_indicator("ta.change(1)", ta.change, close, 1)
        self.test_indicator("ta.mfi(14)", ta.mfi, high, low, close, volume, 14)
        self.test_indicator("ta.wpr(14)", ta.willr, high, low, close, 14)

        for name, status in list(self.results.items())[-9:]:
            print(f"  {status} {name}")
        print()

        # ==================== 波动率类 ====================
        print("📉 波动率类指标")
        print("-" * 50)

        self.test_indicator("ta.tr()", ta.tr, high, low, close)
        self.test_indicator("ta.atr(14)", ta.atr, high, low, close, 14)
        self.test_indicator("ta.bb(20,2)", ta.bb, close, 20, 2.0)
        self.test_indicator("ta.kc(20,1.5)", ta.kc, high, low, close, 20, 1.5)
        self.test_indicator("ta.donchian(20)", ta.donchian, high, low, 20)

        for name, status in list(self.results.items())[-5:]:
            print(f"  {status} {name}")
        print()

        # ==================== 趋势类 ====================
        print("📊 趋势类指标")
        print("-" * 50)

        self.test_indicator("ta.adx(14)", ta.adx, high, low, close, 14)
        self.test_indicator("ta.dmi(14)", ta.dmi, high, low, close, 14)
        self.test_indicator("ta.supertrend(10,3)", ta.supertrend, high, low, close, 10, 3.0)
        self.test_indicator("ta.sar(0.02,0.02,0.2)", ta.psar, high, low, close, 0.02, 0.02, 0.2)

        for name, status in list(self.results.items())[-4:]:
            print(f"  {status} {name}")
        print()

        # ==================== 成交量类 ====================
        print("📊 成交量类指标")
        print("-" * 50)

        self.test_indicator("ta.obv()", ta.obv, close, volume)
        self.test_indicator("ta.vwap()", ta.vwap, high, low, close, volume)
        self.test_indicator("ta.accdist()", ta.ad, high, low, close, volume)
        self.test_indicator("ta.cmf(20)", ta.cmf, high, low, close, volume, 20)

        for name, status in list(self.results.items())[-4:]:
            print(f"  {status} {name}")
        print()

        # ==================== 辅助函数 ====================
        print("🔧 辅助函数")
        print("-" * 50)

        ema_fast = ta.ema(close, 12)
        ema_slow = ta.ema(close, 26)

        self.test_indicator("ta.crossover()", ta.crossover, ema_fast, ema_slow)
        self.test_indicator("ta.crossunder()", ta.crossunder, ema_fast, ema_slow)
        self.test_indicator("ta.highest(14)", ta.highest, close, 14)
        self.test_indicator("ta.lowest(14)", ta.lowest, close, 14)
        self.test_indicator("ta.highestbars(14)", ta.highestbars, close, 14)
        self.test_indicator("ta.lowestbars(14)", ta.lowestbars, close, 14)
        self.test_indicator("ta.stdev(14)", ta.stdev, close, 14)
        self.test_indicator("ta.variance(14)", ta.variance, close, 14)
        self.test_indicator("ta.correlation(14)", ta.correlation, close, volume, 14)
        self.test_indicator("ta.linreg(14)", ta.linreg, close, 14)
        self.test_indicator("ta.percentile(14,50)", ta.percentile_nearest_rank, close, 14, 50)
        self.test_indicator("ta.percentrank(14)", ta.percentrank, close, 14)
        self.test_indicator("ta.median(14)", ta.median, close, 14)
        self.test_indicator("ta.range(14)", ta.range_func, close, 14)

        for name, status in list(self.results.items())[-14:]:
            print(f"  {status} {name}")
        print()

        # ==================== 汇总 ====================
        print("=" * 70)
        total = self.passed + self.failed
        print(f"验证结果: {self.passed}/{total} 通过 ({self.passed/total*100:.1f}%)")

        if self.failed > 0:
            print(f"\n失败指标 ({self.failed}):")
            for name, status in self.results.items():
                if "FAIL" in status:
                    print(f"  {status}")

        print("=" * 70)

    def compare_with_tradingview(self, tv_csv_path: str):
        """
        与真实 TradingView 数据对比

        使用方法:
        1. 在 TradingView 中创建包含所有指标的脚本
        2. 导出图表数据到 CSV
        3. 运行此函数进行对比
        """
        print(f"\n加载 TradingView 数据: {tv_csv_path}")
        tv_df = pd.read_csv(tv_csv_path)

        validator = IndicatorValidator(tolerance=0.01)

        # 对比每个指标列
        indicator_columns = [col for col in tv_df.columns if col.startswith("tv_")]

        for col in indicator_columns:
            indicator_name = col.replace("tv_", "")
            if indicator_name in self.df.columns:
                validator.validate_series(
                    indicator_name,
                    tv_df[col],
                    self.df[indicator_name],
                )

        print(validator.generate_report())


def create_tradingview_export_template():
    """生成 TradingView 导出模板脚本"""
    template = '''
//@version=5
indicator("TV2PY Validation Export", overlay=false)

// ==================== 移动平均 ====================
sma_14 = ta.sma(close, 14)
ema_14 = ta.ema(close, 14)
rma_14 = ta.rma(close, 14)
wma_14 = ta.wma(close, 14)
vwma_14 = ta.vwma(close, 14)

// ==================== 动量指标 ====================
rsi_14 = ta.rsi(close, 14)
[macd_line, signal_line, hist] = ta.macd(close, 12, 26, 9)
[stoch_k, stoch_d] = ta.stoch(close, high, low, 14)
cci_20 = ta.cci(high, low, close, 20)
mom_10 = ta.mom(close, 10)
roc_10 = ta.roc(close, 10)
mfi_14 = ta.mfi(hlc3, 14)
wpr_14 = ta.wpr(14)

// ==================== 波动率 ====================
tr_val = ta.tr
atr_14 = ta.atr(14)
[bb_upper, bb_middle, bb_lower] = ta.bb(close, 20, 2)

// ==================== 趋势 ====================
adx_14 = ta.adx(14)
supertrend_val = ta.supertrend(3, 10)
sar_val = ta.sar(0.02, 0.02, 0.2)

// ==================== 成交量 ====================
obv_val = ta.obv
vwap_val = ta.vwap

// ==================== 输出 ====================
// 在 TradingView 中: 图表 -> 导出图表数据
plot(sma_14, "sma_14")
plot(ema_14, "ema_14")
plot(rsi_14, "rsi_14")
plot(macd_line, "macd_line")
plot(atr_14, "atr_14")
plot(adx_14, "adx_14")
'''
    return template


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TradingView 完整指标验证")
    parser.add_argument("--tv-data", type=str, help="TradingView 导出的 CSV")
    parser.add_argument("--template", action="store_true", help="生成 TradingView 导出模板")

    args = parser.parse_args()

    if args.template:
        print(create_tradingview_export_template())
    else:
        validator = FullValidator()
        validator.run_all_tests()

        if args.tv_data:
            validator.compare_with_tradingview(args.tv_data)
