"""
生成 TradingView 原子级测试脚本

运行此脚本生成 Pine Script 代码，在 TradingView 中运行后导出数据进行对比
"""

PINE_SCRIPT_TEMPLATE = '''
//@version=5
indicator("TV2PY Primitive Validation", overlay=false)

// ============================================================
// Level 0: 测试数据 (使用固定值便于对比)
// ============================================================
src = close

// ============================================================
// Level 1: 历史引用
// ============================================================
ref_0 = src        // close[0]
ref_1 = src[1]     // close[1]
ref_5 = src[5]     // close[5]

// ============================================================
// Level 2: 数学函数
// ============================================================
math_abs = math.abs(src - src[1])
math_max = math.max(src, src[1])
math_min = math.min(src, src[1])
math_round = math.round(src, 2)
math_sqrt = math.sqrt(src)
math_log = math.log(src)
math_exp_small = math.exp(src / 1000)  // 避免溢出
math_sign = math.sign(src - src[1])
math_sum_14 = math.sum(src, 14)
math_avg = math.avg(src, src[1], src[2])

// ============================================================
// Level 3: 核心平滑公式验证
// ============================================================
// EMA: alpha = 2 / (length + 1)
// RMA: alpha = 1 / length

// 手动实现 alpha smooth 来验证
var float manual_ema_14 = na
alpha_ema = 2.0 / (14 + 1)
manual_ema_14 := na(manual_ema_14) ? src : alpha_ema * src + (1 - alpha_ema) * manual_ema_14

var float manual_rma_14 = na
alpha_rma = 1.0 / 14
manual_rma_14 := na(manual_rma_14) ? src : alpha_rma * src + (1 - alpha_rma) * manual_rma_14

// ============================================================
// Level 4: 基础累加器
// ============================================================
ta_cum = ta.cum(src - src[1])
ta_change_1 = ta.change(src, 1)
ta_change_5 = ta.change(src, 5)
ta_rising = ta.rising(src, 3) ? 1 : 0
ta_falling = ta.falling(src, 3) ? 1 : 0

// ============================================================
// Level 5: 条件运算
// ============================================================
cond_result = src > src[1] ? 1 : -1

// ============================================================
// Level 6: 比较和交叉
// ============================================================
ema_fast = ta.ema(src, 12)
ema_slow = ta.ema(src, 26)

ta_crossover = ta.crossover(ema_fast, ema_slow) ? 1 : 0
ta_crossunder = ta.crossunder(ema_fast, ema_slow) ? 1 : 0
ta_cross = ta.cross(ema_fast, ema_slow) ? 1 : 0

// ============================================================
// Level 7: 移动平均
// ============================================================
ta_sma_14 = ta.sma(src, 14)
ta_ema_14 = ta.ema(src, 14)
ta_rma_14 = ta.rma(src, 14)
ta_wma_14 = ta.wma(src, 14)
ta_vwma_14 = ta.vwma(src, 14)

// 验证手动 EMA 与 ta.ema 是否一致
ema_diff = math.abs(manual_ema_14 - ta_ema_14)
rma_diff = math.abs(manual_rma_14 - ta_rma_14)

// ============================================================
// 输出 - 在 TradingView 中导出图表数据
// ============================================================

// Level 1
plot(ref_0, "ref_0", display=display.data_window)
plot(ref_1, "ref_1", display=display.data_window)
plot(ref_5, "ref_5", display=display.data_window)

// Level 2
plot(math_abs, "math_abs", display=display.data_window)
plot(math_max, "math_max", display=display.data_window)
plot(math_min, "math_min", display=display.data_window)
plot(math_round, "math_round", display=display.data_window)
plot(math_sqrt, "math_sqrt", display=display.data_window)
plot(math_log, "math_log", display=display.data_window)
plot(math_sign, "math_sign", display=display.data_window)
plot(math_sum_14, "math_sum_14", display=display.data_window)
plot(math_avg, "math_avg", display=display.data_window)

// Level 3
plot(manual_ema_14, "manual_ema_14", display=display.data_window)
plot(manual_rma_14, "manual_rma_14", display=display.data_window)

// Level 4
plot(ta_cum, "ta_cum", display=display.data_window)
plot(ta_change_1, "ta_change_1", display=display.data_window)
plot(ta_change_5, "ta_change_5", display=display.data_window)
plot(ta_rising, "ta_rising", display=display.data_window)
plot(ta_falling, "ta_falling", display=display.data_window)

// Level 5
plot(cond_result, "cond_result", display=display.data_window)

// Level 6
plot(ema_fast, "ema_fast", display=display.data_window)
plot(ema_slow, "ema_slow", display=display.data_window)
plot(ta_crossover, "ta_crossover", display=display.data_window)
plot(ta_crossunder, "ta_crossunder", display=display.data_window)

// Level 7
plot(ta_sma_14, "ta_sma_14", display=display.data_window)
plot(ta_ema_14, "ta_ema_14", display=display.data_window)
plot(ta_rma_14, "ta_rma_14", display=display.data_window)
plot(ta_wma_14, "ta_wma_14", display=display.data_window)
plot(ta_vwma_14, "ta_vwma_14", display=display.data_window)

// 验证
plot(ema_diff, "ema_diff", display=display.data_window)
plot(rma_diff, "rma_diff", display=display.data_window)
'''

def generate_python_test_code():
    """生成对应的 Python 测试代码"""
    return '''
import pandas as pd
import numpy as np
from validation.primitives import *

def test_primitives(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用相同的 OHLCV 数据计算所有原子操作
    返回结果用于与 TradingView 导出对比
    """
    src = df["close"]
    volume = df["volume"]

    result = pd.DataFrame(index=df.index)

    # Level 1: 历史引用
    result["ref_0"] = series_ref(src, 0)
    result["ref_1"] = series_ref(src, 1)
    result["ref_5"] = series_ref(src, 5)

    # Level 2: 数学函数
    result["math_abs"] = math.abs(src - src.shift(1))
    result["math_max"] = math.max(src, src.shift(1))
    result["math_min"] = math.min(src, src.shift(1))
    result["math_round"] = math.round(src, 2)
    result["math_sqrt"] = math.sqrt(src)
    result["math_log"] = math.log(src)
    result["math_sign"] = math.sign(src - src.shift(1))
    result["math_sum_14"] = math.sum(src, 14)
    result["math_avg"] = (src + src.shift(1) + src.shift(2)) / 3

    # Level 3: 核心平滑
    result["manual_ema_14"] = ema(src, 14)
    result["manual_rma_14"] = rma(src, 14)

    # Level 4: 基础累加
    result["ta_cum"] = cum(src - src.shift(1))
    result["ta_change_1"] = change(src, 1)
    result["ta_change_5"] = change(src, 5)
    result["ta_rising"] = rising(src, 3).astype(int)
    result["ta_falling"] = falling(src, 3).astype(int)

    # Level 5: 条件
    result["cond_result"] = iff(src > src.shift(1), 1, -1)

    # Level 6: 交叉
    ema_fast = ema(src, 12)
    ema_slow = ema(src, 26)
    result["ema_fast"] = ema_fast
    result["ema_slow"] = ema_slow
    result["ta_crossover"] = crossover(ema_fast, ema_slow).astype(int)
    result["ta_crossunder"] = crossunder(ema_fast, ema_slow).astype(int)

    # Level 7: 移动平均
    result["ta_sma_14"] = sma(src, 14)
    result["ta_ema_14"] = ema(src, 14)
    result["ta_rma_14"] = rma(src, 14)

    return result
'''


if __name__ == "__main__":
    print("=" * 60)
    print("TradingView 原子级测试脚本")
    print("=" * 60)
    print()
    print("使用方法:")
    print("1. 复制下面的 Pine Script 代码到 TradingView")
    print("2. 应用到任意图表 (如 ETHUSDT 4H)")
    print("3. 导出图表数据 (右键 -> 导出图表数据)")
    print("4. 运行 Python 对比脚本")
    print()
    print("=" * 60)
    print("Pine Script 代码:")
    print("=" * 60)
    print(PINE_SCRIPT_TEMPLATE)
