# TV2PY - TradingView to Python Backtesting

将 TradingView Pine Script 策略转换为 Python 进行回测的框架。

## 核心特性

- **TradingView 对齐**: 使用 [PyneCore](https://github.com/PyneSys/pynecore) 确保指标计算与 TradingView 100% 一致 (0.001% 容差)
- **高性能回测**: 基于 [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) 的事件驱动引擎
- **Pine Script 语法**: 使用与 TradingView 相似的语法编写策略
- **多资产支持**: 加密货币、外汇、股票

## 为什么选择 TV2PY?

| 问题 | 解决方案 |
|------|----------|
| TradingView 指标与 Python 计算不一致 | 使用 PyneCore，已验证对齐 |
| 需要自己实现 EMA/RSI/MACD | 直接使用 `ta.ema()`, `ta.rsi()` |
| 回测结果与 TradingView 不同 | 相同算法，相同结果 |

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 编写策略 (Pine Script 风格)

```python
"""
@pyne
"""
from pynecore.lib import script, close, ta, strategy, input

@script.strategy(title="EMA Cross")
def main(
    fast=input.int("Fast", 12),
    slow=input.int("Slow", 26),
):
    # 指标计算 - 与 TradingView 完全一致
    fast_ema = ta.ema(close, fast)
    slow_ema = ta.ema(close, slow)

    # 交易信号
    if ta.crossover(fast_ema, slow_ema):
        strategy.entry("Long", strategy.long)

    if ta.crossunder(fast_ema, slow_ema):
        strategy.close("Long")
```

### 2. 下载数据

```bash
pyne data download ccxt --symbol "BINANCE:ETH/USDT" --timeframe 4h
```

### 3. 运行回测

```bash
pyne run my_strategy.py data.ohlcv
```

## 项目结构

```
TV2PY/
├── examples/              # 策略示例
│   ├── pynecore_indicators.py  # 指标示例
│   └── rsi_strategy.py         # RSI 策略
├── validation/            # TradingView 对齐验证
│   ├── primitives/        # 原子操作验证
│   └── validators/        # 验证工具
├── src/
│   ├── strategies/        # 策略实现
│   └── utils/             # 工具函数
└── data/                  # 数据存储
```

## 可用指标 (与 TradingView 对齐)

| 类别 | 指标 |
|------|------|
| 移动平均 | `ta.sma`, `ta.ema`, `ta.rma`, `ta.wma`, `ta.vwma`, `ta.alma`, `ta.hma` |
| 动量 | `ta.rsi`, `ta.macd`, `ta.stoch`, `ta.cci`, `ta.mom`, `ta.roc`, `ta.mfi` |
| 波动率 | `ta.atr`, `ta.bb`, `ta.kc`, `ta.tr` |
| 趋势 | `ta.adx`, `ta.dmi`, `ta.supertrend`, `ta.sar` |
| 成交量 | `ta.obv`, `ta.vwap`, `ta.accdist` |
| 辅助 | `ta.crossover`, `ta.crossunder`, `ta.highest`, `ta.lowest` |

## 验证 TradingView 对齐

```bash
# 运行验证测试
python validation/run_full_validation.py

# 输出:
# ✅ PASS ta.ema(14)
# ✅ PASS ta.rsi(14)
# ✅ PASS ta.macd(12,26,9)
# ...
# 验证结果: 44/44 通过 (100.0%)
```

## 文档

- [PyneCore 文档](https://pynecore.org/docs)
- [NautilusTrader 文档](https://nautilustrader.io/docs/latest/)

## License

MIT
