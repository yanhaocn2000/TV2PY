"""
用固定值测试 - 不依赖数据源

在 TradingView 和 Python 中使用完全相同的数字
"""

# 固定测试数据 (10个价格)
FIXED_PRICES = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0]

PINE_SCRIPT_FIXED = '''
//@version=5
indicator("Fixed Value Test", overlay=false)

// ============================================================
// 固定测试数据 - 与 Python 完全相同
// ============================================================
var float[] prices = array.new_float()

if barstate.isfirst
    array.push(prices, 100.0)
    array.push(prices, 102.0)
    array.push(prices, 101.0)
    array.push(prices, 103.0)
    array.push(prices, 105.0)
    array.push(prices, 104.0)
    array.push(prices, 106.0)
    array.push(prices, 108.0)
    array.push(prices, 107.0)
    array.push(prices, 109.0)

// 只在前10根K线计算
bar_idx = bar_index
src = bar_idx < 10 ? array.get(prices, bar_idx) : na

// ============================================================
// EMA 计算 - 手动实现以匹配 ta.ema 逻辑
// ============================================================
length = 3
alpha = 2.0 / (length + 1)

var float ema_manual = na
if bar_idx == 0
    ema_manual := src
else if bar_idx < 10
    ema_manual := alpha * src + (1 - alpha) * ema_manual

// ta.ema 无法直接用于 array，这里用手动实现验证公式

// ============================================================
// 输出结果
// ============================================================
plot(src, "price", display=display.data_window)
plot(ema_manual, "ema_3", display=display.data_window)
plot(alpha, "alpha", display=display.data_window)

// 在标签中显示完整结果
if barstate.islast
    label.new(bar_index, 0,
        "固定数据 EMA(3) 测试\\n" +
        "价格: [100,102,101,103,105,104,106,108,107,109]\\n" +
        "alpha = " + str.tostring(alpha) + "\\n\\n" +
        "请记录每根K线的 ema_3 值\\n" +
        "与 Python 结果对比",
        style=label.style_label_left)
'''


def calculate_python_ema():
    """Python EMA 计算"""
    import pandas as pd

    prices = FIXED_PRICES
    length = 3
    alpha = 2.0 / (length + 1)

    # 方法1: 手动递归
    ema_manual = [prices[0]]
    for i in range(1, len(prices)):
        ema_val = alpha * prices[i] + (1 - alpha) * ema_manual[-1]
        ema_manual.append(ema_val)

    # 方法2: pandas
    src = pd.Series(prices)
    ema_pandas = src.ewm(alpha=alpha, adjust=False).mean()

    return {
        "prices": prices,
        "alpha": alpha,
        "ema_manual": ema_manual,
        "ema_pandas": ema_pandas.tolist(),
    }


def main():
    print("=" * 60)
    print("固定值测试 - Python vs TradingView")
    print("=" * 60)
    print()

    print("测试数据:", FIXED_PRICES)
    print("EMA 周期: 3")
    print("Alpha = 2/(3+1) = 0.5")
    print()

    result = calculate_python_ema()

    print("Python 计算结果:")
    print("-" * 40)
    print(f"{'Bar':<5} {'Price':<10} {'EMA(3)':<15}")
    print("-" * 40)
    for i, (p, e) in enumerate(zip(result["prices"], result["ema_manual"])):
        print(f"{i:<5} {p:<10} {e:<15.6f}")

    print()
    print("=" * 60)
    print("TradingView 测试步骤")
    print("=" * 60)
    print("""
1. 复制下面的 Pine Script 到 TradingView
2. 添加到任意图表
3. 查看数据窗口中的 ema_3 值
4. 与上面的 Python 结果对比

如果结果一致 → 算法对齐 ✅
如果结果不一致 → 需要调整 Python 实现
""")
    print()
    print("=" * 60)
    print("Pine Script 代码")
    print("=" * 60)
    print(PINE_SCRIPT_FIXED)


if __name__ == "__main__":
    main()
