# TradingView 转换潜在问题

## 概述

根据 [TradingView 官方文档](https://www.tradingview.com/pine-script-docs/) 研究，以下是转换过程中容易忽略的关键问题。

---

## 1. 重绘问题 (Repainting)

### 什么是重绘？

历史 bar 和实时 bar 的计算行为不同，导致回测结果与实盘不一致。

参考: [Concepts / Repainting](https://www.tradingview.com/pine-script-docs/concepts/repainting/)

### 主要原因

| 原因 | 说明 | Python 影响 |
|------|------|-------------|
| **barstate 变量** | `barstate.isnew` 在历史 bar 上是 close 时为 true，实时是 open 时为 true | 需要理解逻辑意图 |
| **request.security** | HTF 数据可能包含未完成的实时 bar | 需要正确处理 |
| **varip 变量** | 保留实时 bar 内的值，历史无法模拟 | ⚠️ 无法完全复现 |

### Python 回测的优势

```python
# Python 回测只处理已完成的 bar，天然避免了部分重绘问题
# 所有 bar 都是 "历史 bar"，行为一致
for i in range(len(data)):
    # 每根 bar 只计算一次，使用收盘价
    signal = calculate_signal(data[:i+1])
```

### 需要注意的情况

如果原始 Pine Script 使用了以下功能，转换时需要特别注意：

```pinescript
// ⚠️ 这些会导致重绘
if barstate.isconfirmed  // 实时 bar 关闭时
if barstate.isnew        // 历史 vs 实时行为不同
varip float x = 0        // 实时 tick 级别状态
```

---

## 2. request.security 的正确用法

### 避免未来函数 (Lookahead Bias)

参考: [How to avoid repainting when using security()](https://www.tradingview.com/script/cyPWY96u-How-to-avoid-repainting-when-using-security-PineCoders-FAQ/)

**错误用法** (会看到未来数据):
```pinescript
// ❌ 错误: 可能获取未完成的 HTF bar 数据
dailyClose = request.security(syminfo.tickerid, "1D", close)
```

**正确用法**:
```pinescript
// ✅ 正确: 只获取已确认的 HTF bar 数据
dailyClose = request.security(syminfo.tickerid, "1D", close[1], lookahead=barmerge.lookahead_on)
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `lookahead` | `barmerge.lookahead_off` | 默认，可能重绘 |
| `lookahead` | `barmerge.lookahead_on` | 需配合 `[1]` 偏移使用 |
| `gaps` | `barmerge.gaps_off` | 默认，填充 NA |
| `gaps` | `barmerge.gaps_on` | 保留 NA 间隙 |

### Python 实现对应

```python
# 我们的实现已正确处理
pd.merge_asof(
    base_df,
    htf_data,
    on='timestamp',
    direction='backward'  # 等同于 lookahead_on + [1]
)
```

---

## 3. var vs varip 变量

参考: [Language / Variable declarations](https://www.tradingview.com/pine-script-docs/language/variable-declarations/)

### 区别

| 类型 | 历史 bar | 实时 bar | Python 等效 |
|------|---------|---------|-------------|
| 普通变量 | 每 bar 重置 | 每 tick 重置 | 普通变量 |
| `var` | 首次初始化后保持 | 每 bar 确认后保持 | `Persistent` |
| `varip` | 同 var | **每 tick 保持** | ⚠️ 无法模拟 |

### varip 的问题

```pinescript
// 这段代码在回测中无法真实模拟
varip float tickCount = 0
tickCount := tickCount + 1  // 实时每 tick +1，历史只 +1/bar
```

**结论**: 如果原始策略使用 `varip`，回测结果可能与实盘显著不同。

---

## 4. 策略执行模型

参考: [Language / Execution model](https://www.tradingview.com/pine-script-docs/language/execution-model/)

### calc_on_every_tick

```pinescript
strategy("My Strategy", calc_on_every_tick=true)
```

| 设置 | 历史 bar | 实时 bar |
|------|---------|---------|
| `false` (默认) | 每 bar 计算一次 | 每 bar 关闭时计算 |
| `true` | 每 bar 计算一次 | **每 tick 计算** |

**关键问题**: `calc_on_every_tick=true` 在历史 bar 上无法模拟 tick 级别行为。

参考: [Tip: TradingView backtest results are different when calculating on every tick](https://www.tradingcode.net/tradingview/calc-tick-backtest-difference/)

### calc_on_order_fills

```pinescript
strategy("My Strategy", calc_on_order_fills=true)
```

订单成交后立即重新计算策略，可用于：
- 快速调整仓位
- 级联订单

**Python 实现**:
```python
# 需要在成交后立即重新评估信号
if order_filled:
    recalculate_signals()
```

---

## 5. 成交价格假设

参考: [Concepts / Strategies](https://www.tradingview.com/pine-script-docs/concepts/strategies/)

### TradingView 的 Bar 内价格运动假设

TradingView 根据 Open 价格位置推断 bar 内价格运动顺序：

```
如果 Open 更接近 High:
    假设顺序: Open → High → Low → Close

如果 Open 更接近 Low:
    假设顺序: Open → Low → High → Close
```

### 对止损止盈的影响

```
Bar: O=100, H=105, L=95, C=102

情况1: Open(100) 接近 High(105)
  → 假设先涨后跌: 100 → 105 → 95 → 102
  → 如果止盈=104，止损=96
  → 结果: 先触发止盈 (104)

情况2: Open(100) 接近 Low(95)
  → 假设先跌后涨: 100 → 95 → 105 → 102
  → 如果止盈=104，止损=96
  → 结果: 先触发止损 (96)
```

### Python 实现建议

```python
def determine_fill_order(open_, high, low, close, stop_loss, take_profit, side):
    """
    模拟 TradingView 的 bar 内价格运动假设
    """
    if side == 'long':
        # 判断 open 更接近 high 还是 low
        if abs(open_ - high) < abs(open_ - low):
            # Open → High → Low → Close
            if take_profit <= high:
                return 'take_profit', take_profit
            if stop_loss >= low:
                return 'stop_loss', stop_loss
        else:
            # Open → Low → High → Close
            if stop_loss >= low:
                return 'stop_loss', stop_loss
            if take_profit <= high:
                return 'take_profit', take_profit

    return None, None
```

### Bar Magnifier (Premium 功能)

TradingView Premium 用户可以启用 Bar Magnifier，使用更细粒度的数据（通常是1分钟）来模拟 bar 内价格运动。

---

## 6. 限价单成交假设

参考: [TradingView fill limit assumption explained](https://www.tradingcode.net/tradingview/limit-fill-assumption/)

### backtest_fill_limits_assumption

```pinescript
strategy("My Strategy", backtest_fill_limits_assumption=1)
```

| 值 | 行为 |
|----|------|
| 0 (默认) | 价格触及即成交 |
| 1+ | 价格需超过限价 N 个 tick 才成交 |

**问题**: 默认设置过于乐观，假设所有限价单都能在触及价格时成交。

### Python 实现建议

```python
# 更保守的限价单成交逻辑
def check_limit_fill(limit_price, high, low, slippage_ticks=1):
    tick_size = 0.01  # 根据品种设置

    # 买入限价单: 需要价格低于限价 - slippage
    if low <= limit_price - slippage_ticks * tick_size:
        return True, limit_price

    return False, None
```

---

## 7. 数据差异

### 不同数据源的潜在差异

| 差异来源 | 说明 |
|---------|------|
| 时区 | TradingView 使用交易所时区 |
| K线闭合时间 | 部分交易所差异 |
| 价格精度 | 小数位数差异 |
| 成交量 | 不同交易所汇总方式 |

### 建议

```python
# 确保时区一致
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')

# 确保价格精度一致
df['close'] = df['close'].round(8)
```

---

## 8. 完整检查清单

转换时需要检查原始 Pine Script 是否使用了以下功能：

### ⚠️ 高风险 (可能无法准确复现)

- [ ] `varip` 变量
- [ ] `calc_on_every_tick=true`
- [ ] `barstate.isrealtime`
- [ ] `timenow` (当前时间)
- [ ] 无 `[1]` 偏移的 `request.security`

### ⚡ 中等风险 (需要仔细处理)

- [ ] `barstate.isnew` / `barstate.isconfirmed`
- [ ] `request.security` (多时间框架)
- [ ] 复杂的止损止盈逻辑
- [ ] `calc_on_order_fills=true`

### ✅ 低风险 (通常可以直接转换)

- [ ] `var` 变量 → `Persistent`
- [ ] `ta.*` 指标 → PyneCore
- [ ] 简单的入场/出场条件
- [ ] 固定止损止盈

---

## 9. 验证方法

### 9.1 指标对比

使用固定数据验证指标计算：

```python
# 生成固定测试数据
test_data = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

# 计算指标
python_result = calculate_indicator(test_data)

# 与 TradingView 导出数据对比
# 误差应 < 0.001%
```

### 9.2 信号对比

```python
# 使用相同历史数据
# 对比每根 bar 的买卖信号
# 信号应该完全一致 (同一数据源)
```

### 9.3 回测结果对比

| 指标 | 预期差异 | 原因 |
|------|---------|------|
| 总收益 | 1-5% | 成交价格假设不同 |
| 交易次数 | 0-2% | 边界条件处理 |
| 最大回撤 | 1-3% | 累积误差 |

---

## 总结

| 问题 | 严重程度 | 我们的处理状态 |
|------|---------|---------------|
| 重绘 (Repainting) | ⚠️ 中 | ✅ Python 回测天然避免 |
| request.security | ⚠️ 中 | ✅ 已实现 merge_asof |
| varip 变量 | 🔴 高 | ❌ 无法完全模拟 |
| calc_on_every_tick | 🔴 高 | ❌ 无法模拟 tick 级别 |
| 成交价格假设 | ⚠️ 中 | ⚡ 可以改进 |
| 限价单假设 | ⚠️ 中 | ⚡ 可以改进 |

### 建议

1. **避免使用** `varip` 和 `calc_on_every_tick=true` 的策略
2. **仔细检查** `request.security` 的用法
3. **理解** TradingView 的成交价格假设
4. **接受** 1-5% 的回测结果差异是正常的

---

## 参考资料

- [TradingView Pine Script Docs](https://www.tradingview.com/pine-script-docs/)
- [Execution Model](https://www.tradingview.com/pine-script-docs/language/execution-model/)
- [Repainting](https://www.tradingview.com/pine-script-docs/concepts/repainting/)
- [Strategies](https://www.tradingview.com/pine-script-docs/concepts/strategies/)
- [PineCoders FAQ](https://www.pinecoders.com/faq_and_code/)
