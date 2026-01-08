# TV2PY Hummingbot Integration

将 TradingView Top 50 指标与 [Hummingbot](https://hummingbot.org/) 交易框架整合。

## 概述 / Overview

本模块提供了将 TV2PY 指标用于 Hummingbot 自动交易策略的适配器和信号生成器。

### 特性 / Features

- ✅ 50+ TradingView 指标支持
- ✅ 预配置的策略组合 (趋势跟踪、均值回归、SMC、动量、突破)
- ✅ 自动止损止盈计算
- ✅ 仓位管理
- ✅ Hummingbot V2 Framework 兼容

## 安装 / Installation

### 1. 安装 TV2PY

```bash
git clone https://github.com/yanhaocn2000/TV2PY.git
cd TV2PY
pip install -r requirements.txt
```

### 2. 安装 Hummingbot

```bash
# 从源码安装
git clone https://github.com/hummingbot/hummingbot.git
cd hummingbot
./install
```

### 3. 配置集成

```bash
# 将 TV2PY 添加到 Python 路径
export PYTHONPATH="${PYTHONPATH}:/path/to/TV2PY"

# 或复制集成模块到 Hummingbot
cp -r TV2PY/integrations/hummingbot /path/to/hummingbot/scripts/
```

## 快速开始 / Quick Start

### 基本用法

```python
from integrations.hummingbot import TV2PYSignalGenerator, StrategyType

# 创建信号生成器
signal_gen = TV2PYSignalGenerator(
    strategy_type=StrategyType.SMART_MONEY,
    risk_per_trade=0.02,  # 2% 风险
    min_confidence=0.6,   # 60% 最小置信度
)

# 从K线数据获取信号
signal = signal_gen.get_signal(candles_df)

if signal.is_long:
    print(f"买入信号: {signal.reason}")
    print(f"止损: {signal.stop_loss}")
    print(f"止盈: {signal.take_profit}")
elif signal.is_short:
    print(f"卖出信号: {signal.reason}")
```

### 在 Hummingbot 脚本中使用

```python
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from integrations.hummingbot import TV2PYSignalGenerator, StrategyType

class MyStrategy(ScriptStrategyBase):
    def __init__(self, connectors):
        super().__init__(connectors)
        self.signal_gen = TV2PYSignalGenerator(
            strategy_type=StrategyType.MOMENTUM
        )

    def on_tick(self):
        candles_df = self.get_candles_df()
        signal = self.signal_gen.get_signal(candles_df)

        if signal.is_long and signal.confidence > 0.7:
            self.buy(...)
```

## 策略类型 / Strategy Types

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| `TREND_FOLLOWING` | SuperTrend + Squeeze + Choppiness | 趋势明显的市场 |
| `MEAN_REVERSION` | Bollinger + Volume Profile | 震荡区间市场 |
| `SMART_MONEY` | SMC + Volume Profile + Squeeze | 机构级分析 |
| `MOMENTUM` | Squeeze + SuperTrend | 动量突破 |
| `BREAKOUT` | Squeeze + Bollinger | 波动率突破 |

## 自定义指标 / Custom Indicators

```python
# 使用自定义指标组合
signal_gen = TV2PYSignalGenerator(
    strategy_type=StrategyType.CUSTOM,
    custom_indicators=[
        ("smc", {"swing_length": 10}),
        ("squeeze", {"bb_length": 20}),
        ("supertrend", {"atr_period": 10, "multiplier": 3}),
        ("atr", {"length": 14}),
    ]
)
```

### 可用指标 / Available Indicators

| 指标 | 参数 |
|------|------|
| `smc` | `swing_length`, `ob_lookback` |
| `squeeze` | `bb_length`, `bb_mult`, `kc_length`, `kc_mult` |
| `supertrend` | `atr_period`, `multiplier` |
| `volume_profile` | `row_size`, `value_area_percent` |
| `bollinger` | `length`, `mult` |
| `choppiness` | `length` |
| `atr` | `length` |

## 信号结构 / Signal Structure

```python
@dataclass
class TradingSignal:
    signal_type: SignalType    # LONG, SHORT, NEUTRAL, CLOSE_LONG, CLOSE_SHORT
    price: float               # 当前价格
    strength: float            # 信号强度 (0-1)
    confidence: float          # 信号置信度 (0-1)
    stop_loss: float           # 建议止损价
    take_profit: float         # 建议止盈价
    position_size: float       # 建议仓位比例
    reason: str                # 信号原因
    indicators: dict           # 指标详情
```

## 示例脚本 / Example Scripts

### scripts/smc_strategy_v2.py

完整的 SMC 策略示例，包含：
- 多指标组合信号
- 自动止损止盈
- 仓位管理
- 状态显示

运行测试:
```bash
python scripts/smc_strategy_v2.py
```

在 Hummingbot 中运行:
```
start --script smc_strategy_v2.py
```

## 最佳实践 / Best Practices

### 1. 多时间框架确认

```python
# 日线趋势
daily_signal = signal_gen.get_signal(daily_candles)

# 4H 入场
if daily_signal.signal_type == SignalType.LONG:
    h4_signal = signal_gen.get_signal(h4_candles)
    if h4_signal.is_long:
        # 执行交易
        pass
```

### 2. 风险管理

```python
signal_gen = TV2PYSignalGenerator(
    risk_per_trade=0.01,    # 保守: 1%
    min_confidence=0.7,     # 高置信度
)
```

### 3. 回测验证

```python
# 使用历史数据回测
for i in range(100, len(candles)):
    signal = signal_gen.get_signal(candles[:i])
    # 记录信号和结果
```

## 参考资料 / References

- [Hummingbot Documentation](https://hummingbot.org/docs/)
- [Hummingbot V2 Framework](https://hummingbot.org/strategies/)
- [TV2PY Indicators](../../strategies/)

## License

MIT License
