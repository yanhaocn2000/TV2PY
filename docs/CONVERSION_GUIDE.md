# TradingView Pine Script 转换指南

## 概述

将 TradingView Pine Script 策略转换为 Python，用于 NautilusTrader 回测。

**核心原则**: 使用 PyneCore 库确保指标计算与 TradingView 对齐 (误差 < 0.001%)

---

## 转换流程

```
┌─────────────────┐
│  第1步: 分析    │  识别所有指标和函数
└────────┬────────┘
         ▼
┌─────────────────┐
│  第2步: 检查    │  确认 PyneCore 支持
└────────┬────────┘
         ▼
┌─────────────────┐
│  第3步: 转换    │  逐段翻译代码
└────────┬────────┘
         ▼
┌─────────────────┐
│  第4步: 验证    │  对比结果
└─────────────────┘
```

---

## 第1步: 分析 Pine Script

### 1.1 提取所有 ta.* 函数

从 Pine Script 中找出所有使用的技术指标:

```pinescript
// 示例 Pine Script
ema55 = ta.ema(close, 55)
rsi = ta.rsi(close, 14)
atr = ta.atr(14)
crossUp = ta.crossover(fast, slow)
```

**列出清单**:
| 函数 | 参数 |
|------|------|
| ta.ema | close, 55 |
| ta.rsi | close, 14 |
| ta.atr | 14 |
| ta.crossover | fast, slow |

### 1.2 识别自定义指标

不在 ta.* 中的计算逻辑:

```pinescript
// 自定义 TPX 指标
range = ta.highest(high, 14) - ta.lowest(low, 14)
tpx = (close - ta.lowest(low, 14)) / range * 100
```

### 1.3 识别特殊功能

| 功能 | Pine Script | 需要特殊处理 |
|------|-------------|-------------|
| 多时间框架 | `request.security()` | 是 - 需要预重采样 |
| 绘图 | `plot()`, `plotshape()` | 否 - 仅显示用 |
| 表格 | `table.new()` | 否 - 仅显示用 |
| 策略 | `strategy.entry()` | 是 - 转为信号 |

---

## 第2步: 检查 PyneCore 兼容性

### 2.1 PyneCore 支持的指标 (完整列表)

```
✅ 移动平均线: sma, ema, wma, vwma, rma, hma, alma, swma
✅ 动量指标: rsi, macd, mom, roc, cci, cmo, tsi
✅ 波动率: atr, tr, bb, bbw, kc, kcw, stdev
✅ 趋势: supertrend, dmi, sar
✅ 成交量: obv, vwap, mfi, pvt, nvi, pvi
✅ 统计: highest, lowest, change, cross, crossover, crossunder
✅ 其他: linreg, stoch, pivothigh, pivotlow, valuewhen, barssince
```

### 2.2 需要自己实现的

| 功能 | 实现方式 |
|------|---------|
| Tilson T3 | 6层 EMA 组合 |
| TPX | 用 highest/lowest 构建 |
| request.security | 预重采样 + merge_asof |

### 2.3 快速检查方法

```python
from pynecore.lib import ta

# 查看所有支持的函数
print(dir(ta))
```

---

## 第3步: 转换代码

### 3.1 转换规则对照表

| Pine Script | Python/PyneCore |
|-------------|-----------------|
| `ta.ema(close, 14)` | `ta.ema(close, 14)` |
| `ta.crossover(a, b)` | `ta.crossover(a, b)` |
| `strategy.entry("Long", strategy.long)` | `signal = 1` |
| `strategy.exit("Exit", stop=sl)` | 在回测循环中处理 |
| `request.security("", "1D", close)` | `RequestSecurity` 类 |
| `input.int("Length", 14)` | 函数参数 |
| `var float x = 0` | `Persistent[float]` |
| `x := x + 1` | `x = x + 1` |
| `na` | `NA(float)` |
| `nz(x, 0)` | `x if not isinstance(x, NA) else 0` |

### 3.2 转换示例

**Pine Script 原始代码**:
```pinescript
//@version=5
strategy("EMA Cross", overlay=true)

fast = input.int(12, "Fast")
slow = input.int(26, "Slow")

emaFast = ta.ema(close, fast)
emaSlow = ta.ema(close, slow)

longCond = ta.crossover(emaFast, emaSlow)
shortCond = ta.crossunder(emaFast, emaSlow)

if longCond
    strategy.entry("Long", strategy.long)
if shortCond
    strategy.entry("Short", strategy.short)
```

**Python 转换**:
```python
"""
@pyne
"""
from pynecore.lib import close, ta, strategy

def main(fast: int = 12, slow: int = 26):
    ema_fast = ta.ema(close, fast)
    ema_slow = ta.ema(close, slow)

    long_cond = ta.crossover(ema_fast, ema_slow)
    short_cond = ta.crossunder(ema_fast, ema_slow)

    if long_cond:
        strategy.entry("Long", strategy.long)
    if short_cond:
        strategy.entry("Short", strategy.short)
```

### 3.3 不需要逐行对比的部分

| 类型 | 原因 |
|------|------|
| `plot()` | 仅用于图表显示 |
| `plotshape()` | 仅用于图表显示 |
| `bgcolor()` | 仅用于图表显示 |
| `table.*` | 仅用于显示统计 |
| `label.*` | 仅用于图表标注 |
| `line.*` | 仅用于图表画线 |

### 3.4 必须仔细转换的部分

| 类型 | 原因 |
|------|------|
| `ta.*` 指标 | 影响信号准确性 |
| 入场条件 | 核心逻辑 |
| 出场条件 | 核心逻辑 |
| 止损止盈 | 影响回测结果 |
| `request.security` | 多时间框架逻辑 |
| 自定义函数 | 需要完整翻译 |

---

## 第4步: 验证对齐

### 4.1 指标验证

使用相同数据，对比 TradingView 和 Python 的指标值:

```python
# 生成固定测试数据
test_data = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]

# Python 计算
python_ema = tv_ema(np.array(test_data), 5)

# 与 TradingView 导出数据对比
tv_ema_values = [...]  # 从 TradingView 导出

# 检查误差
error = np.abs(python_ema - tv_ema_values) / tv_ema_values * 100
print(f"最大误差: {error.max():.6f}%")  # 应该 < 0.001%
```

### 4.2 信号验证

```python
# 对比买卖信号
python_signals = get_signals(data)
tv_signals = load_tv_signals('exported.csv')

# 信号应该完全一致
assert (python_signals == tv_signals).all()
```

### 4.3 预期误差范围

| 项目 | 预期误差 |
|------|---------|
| 指标值 | < 0.001% |
| 买卖信号 | 0 (完全一致) |
| 回测收益 | 1-2% (累积误差) |

回测收益差异来源:
- 手续费模型差异
- 滑点模型差异
- 成交价格假设差异

---

## 常见问题

### Q1: 需要每行代码都对比吗?

**不需要**。只需对比:
1. 核心指标计算
2. 入场/出场条件
3. 止损止盈逻辑

显示相关代码 (plot, table, label) 可以忽略。

### Q2: 自定义指标怎么处理?

分解为 PyneCore 支持的基础函数:

```pinescript
// Pine: 自定义 Tilson T3
e1 = ta.ema(src, length)
e2 = ta.ema(e1, length)
// ... e3-e6
t3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
```

```python
# Python: 用基础 ema 构建
e1 = ta.ema(src, length)
e2 = ta.ema(e1, length)
# ... e3-e6
t3 = c1*e6 + c2*e5 + c3*e4 + c4*e3
```

### Q3: request.security 怎么处理?

预先准备多时间框架数据:

```python
from strategies.multi_timeframe import RequestSecurity

security = RequestSecurity(df_4h)
daily_close = security.get('1D', 'close')
daily_ema = security.get_indicator('1D', tv_ema, 20)
```

### Q4: 策略入场出场怎么处理?

转换为信号，在回测循环中处理:

```python
# 信号生成
long_signal = condition1 and condition2
short_signal = condition3 and condition4

# 回测循环中
if long_signal:
    position = 1
    entry_price = close[i]
    stop_loss = entry_price - atr * 2
```

---

## 快速检查清单

转换完成后，确认以下几点:

- [ ] 所有 ta.* 函数都有对应实现
- [ ] 自定义指标已正确分解
- [ ] 入场条件完整转换
- [ ] 出场条件完整转换
- [ ] 止损止盈逻辑正确
- [ ] 多时间框架已处理 (如有)
- [ ] 使用固定数据验证指标对齐

---

## 文件结构

```
TV2PY/
├── strategies/
│   ├── tpx_slingshot.py      # PyneCore 格式策略
│   ├── tpx_nautilus_adapter.py  # 回测适配器
│   ├── stop_loss_modes.py    # 止损止盈模块
│   └── multi_timeframe.py    # 多时间框架模块
├── docs/
│   └── CONVERSION_GUIDE.md   # 本文档
└── data/
    └── ETHUSDT_4h.csv        # 测试数据
```

---

## 总结

1. **分析** - 列出所有指标和特殊功能
2. **检查** - 确认 PyneCore 支持
3. **转换** - 按对照表翻译，重点关注核心逻辑
4. **验证** - 用固定数据对比指标值

**不需要逐行对比**，只需确保核心计算逻辑一致。
