"""
ICT Integration Tests

测试 ICT (Inner Circle Trader) 模块的功能。
"""

import sys
from pathlib import Path
from datetime import time, datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def generate_test_data(n: int = 200) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)

    dates = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=n,
        freq="4H"
    )

    # 生成带趋势的价格数据
    base = 3000
    trend = np.cumsum(np.random.randn(n) * 20)
    close = base + trend

    high = close + np.abs(np.random.randn(n)) * 30
    low = close - np.abs(np.random.randn(n)) * 30
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = 1000000 + np.abs(np.random.randn(n)) * 500000

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def test_smc_swing_highs_lows():
    """测试摆动高低点检测"""
    print("\n" + "=" * 60)
    print("Testing SMC Swing Highs/Lows")
    print("=" * 60)

    from integrations.ict import smc

    df = generate_test_data()
    result = smc.swing_highs_lows(df, swing_length=10)

    print(f"数据点数: {len(df)}")
    print(f"摆动高点数: {(result['HighLow'] == 1).sum()}")
    print(f"摆动低点数: {(result['HighLow'] == -1).sum()}")

    assert "HighLow" in result.columns
    assert "Level" in result.columns
    assert (result["HighLow"].dropna().isin([1, -1])).all()

    print("Swing Highs/Lows test passed!")


def test_smc_fvg():
    """测试公允价值缺口检测"""
    print("\n" + "=" * 60)
    print("Testing SMC Fair Value Gap")
    print("=" * 60)

    from integrations.ict import smc

    df = generate_test_data()
    result = smc.fvg(df)

    bullish_fvg = (result["FVG"] == 1).sum()
    bearish_fvg = (result["FVG"] == -1).sum()

    print(f"看涨 FVG: {bullish_fvg}")
    print(f"看跌 FVG: {bearish_fvg}")

    assert "FVG" in result.columns
    assert "Top" in result.columns
    assert "Bottom" in result.columns
    assert "MitigatedIndex" in result.columns

    print("FVG test passed!")


def test_smc_order_blocks():
    """测试订单块检测"""
    print("\n" + "=" * 60)
    print("Testing SMC Order Blocks")
    print("=" * 60)

    from integrations.ict import smc

    df = generate_test_data()
    swing_hl = smc.swing_highs_lows(df, swing_length=10)
    result = smc.ob(df, swing_hl)

    bullish_ob = (result["OB"] == 1).sum()
    bearish_ob = (result["OB"] == -1).sum()

    print(f"看涨订单块: {bullish_ob}")
    print(f"看跌订单块: {bearish_ob}")

    assert "OB" in result.columns
    assert "Top" in result.columns
    assert "Bottom" in result.columns
    assert "OBVolume" in result.columns

    print("Order Blocks test passed!")


def test_smc_bos_choch():
    """测试结构突破和性质改变"""
    print("\n" + "=" * 60)
    print("Testing SMC BOS/CHoCH")
    print("=" * 60)

    from integrations.ict import smc

    df = generate_test_data()
    swing_hl = smc.swing_highs_lows(df, swing_length=10)
    result = smc.bos_choch(df, swing_hl)

    bos_count = result["BOS"].dropna().shape[0]
    choch_count = result["CHOCH"].dropna().shape[0]

    print(f"BOS 信号数: {bos_count}")
    print(f"CHoCH 信号数: {choch_count}")

    assert "BOS" in result.columns
    assert "CHOCH" in result.columns
    assert "Level" in result.columns

    print("BOS/CHoCH test passed!")


def test_smc_liquidity():
    """测试流动性检测"""
    print("\n" + "=" * 60)
    print("Testing SMC Liquidity")
    print("=" * 60)

    from integrations.ict import smc

    df = generate_test_data()
    swing_hl = smc.swing_highs_lows(df, swing_length=10)
    result = smc.liquidity(df, swing_hl, range_percent=0.02)

    bullish_liq = (result["Liquidity"] == 1).sum()
    bearish_liq = (result["Liquidity"] == -1).sum()

    print(f"看涨流动性区: {bullish_liq}")
    print(f"看跌流动性区: {bearish_liq}")

    assert "Liquidity" in result.columns
    assert "Level" in result.columns
    assert "Swept" in result.columns

    print("Liquidity test passed!")


def test_smc_retracements():
    """测试回撤计算"""
    print("\n" + "=" * 60)
    print("Testing SMC Retracements")
    print("=" * 60)

    from integrations.ict import smc

    df = generate_test_data()
    swing_hl = smc.swing_highs_lows(df, swing_length=10)
    result = smc.retracements(df, swing_hl)

    print(f"回撤数据点: {len(result)}")
    print(f"最深回撤: {result['DeepestRetracement%'].max():.1f}%")

    assert "Direction" in result.columns
    assert "CurrentRetracement%" in result.columns
    assert "DeepestRetracement%" in result.columns

    print("Retracements test passed!")


def test_ict_analyzer():
    """测试 ICT 分析器"""
    print("\n" + "=" * 60)
    print("Testing ICT Analyzer")
    print("=" * 60)

    from integrations.ict import ICTAnalyzer

    df = generate_test_data()
    analyzer = ICTAnalyzer(swing_length=10)

    # 执行分析
    result = analyzer.analyze(df)

    print(f"摆动点: {len(result.swing_highs_lows.dropna())}")
    print(f"订单块: {len(result.order_blocks['OB'].dropna())}")
    print(f"FVG: {len(result.fvg['FVG'].dropna())}")

    # 获取市场偏向
    bias = analyzer.get_bias(df)
    print(f"市场偏向: {bias['bias']}")
    print(f"置信度: {bias['confidence']:.1%}")

    assert result.swing_highs_lows is not None
    assert result.fvg is not None
    assert result.order_blocks is not None

    print("ICT Analyzer test passed!")


def test_trading_sessions():
    """测试交易时段"""
    print("\n" + "=" * 60)
    print("Testing Trading Sessions")
    print("=" * 60)

    from integrations.ict import TradingSessions

    sessions = TradingSessions()

    # 测试伦敦时段
    test_times = [
        (time(8, 0), "London", True),
        (time(3, 0), "London", False),
        (time(14, 0), "New York", True),
        (time(2, 0), "Tokyo", True),
    ]

    for t, session, expected in test_times:
        result = sessions.is_in_session(session, t)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {t} in {session}: {result} (expected: {expected})")
        assert result == expected

    # 测试时段过滤
    df = generate_test_data()
    london_df = sessions.filter_session(df, "London")
    print(f"\n  伦敦时段K线数: {len(london_df)} / {len(df)}")

    print("Trading Sessions test passed!")


def test_kill_zones():
    """测试杀戮区"""
    print("\n" + "=" * 60)
    print("Testing Kill Zones")
    print("=" * 60)

    from integrations.ict import KillZones

    kz = KillZones()

    # 测试杀戮区识别
    test_times = [
        (time(2, 0), True, "Asian"),
        (time(7, 30), True, "London Open"),
        (time(12, 0), True, "New York"),
        (time(15, 0), True, "London Close"),
        (time(20, 0), False, None),
    ]

    for t, expected_active, expected_name in test_times:
        is_active, name = kz.is_in_kill_zone(t)
        status = "✓" if is_active == expected_active else "✗"
        print(f"  {status} {t}: active={is_active}, zone={name}")

    # 分析杀戮区
    df = generate_test_data()
    analysis = kz.analyze_kill_zones(df)

    print("\n  杀戮区分析:")
    for name, stats in analysis.items():
        print(f"    {name}: K线数={stats['candle_count']}, 阳线比例={stats['bullish_ratio']:.1%}")

    print("Kill Zones test passed!")


def test_signal_generator():
    """测试信号生成器"""
    print("\n" + "=" * 60)
    print("Testing ICT Signal Generator")
    print("=" * 60)

    from integrations.ict import ICTSignalGenerator

    df = generate_test_data()

    generator = ICTSignalGenerator(
        swing_length=10,
        min_rr=1.5,
        use_kill_zones=False,  # 测试时不过滤杀戮区
    )

    signals = generator.generate_signals(df, lookback=50)

    print(f"生成信号数: {len(signals)}")

    if signals:
        signal = signals[0]
        print(f"\n  示例信号:")
        print(f"    类型: {signal.signal_type.value}")
        print(f"    入场类型: {signal.entry_type.value}")
        print(f"    入场价: {signal.entry_price:.2f}")
        print(f"    止损: {signal.stop_loss:.2f}")
        print(f"    止盈: {signal.take_profit:.2f}")
        print(f"    R:R: {signal.risk_reward_ratio:.2f}")
        print(f"    置信度: {signal.confidence:.1%}")

    # 回测
    backtest = generator.backtest_signals(df)
    print(f"\n  回测结果:")
    print(f"    总交易: {backtest.get('total_trades', 0)}")
    print(f"    胜率: {backtest.get('win_rate', 0):.1%}")
    print(f"    总收益: {backtest.get('total_return', 0):.1%}")

    print("Signal Generator test passed!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("TV2PY ICT Integration Tests")
    print("=" * 60)

    tests = [
        ("SMC Swing Highs/Lows", test_smc_swing_highs_lows),
        ("SMC FVG", test_smc_fvg),
        ("SMC Order Blocks", test_smc_order_blocks),
        ("SMC BOS/CHoCH", test_smc_bos_choch),
        ("SMC Liquidity", test_smc_liquidity),
        ("SMC Retracements", test_smc_retracements),
        ("ICT Analyzer", test_ict_analyzer),
        ("Trading Sessions", test_trading_sessions),
        ("Kill Zones", test_kill_zones),
        ("Signal Generator", test_signal_generator),
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
