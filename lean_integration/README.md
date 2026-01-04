# Lean Engine 集成指南

## 概述

将 QuantConnect Lean 引擎集成到 TV2PY，支持股票和期权的回测与实盘交易。

参考: [Lean GitHub](https://github.com/QuantConnect/Lean)

---

## 架构对比

| 功能 | NautilusTrader | Lean Engine |
|------|---------------|-------------|
| 语言 | Python/Rust | C#/Python |
| 资产类型 | 加密货币为主 | 股票/期权/期货/外汇 |
| 数据源 | 交易所直连 | QuantConnect/本地 |
| 实盘券商 | 币安/OKX | IB/Alpaca/TD |
| 期权支持 | ❌ 有限 | ✅ 完整 |

---

## 安装

### 方式1: Docker (推荐)

```bash
# 安装 Lean CLI
pip install lean

# 登录 QuantConnect (免费账户)
lean login

# 初始化工作区
lean init

# 下载股票数据
lean data download --dataset "US Equity" --ticker "AAPL" --resolution "Minute"

# 下载期权数据
lean data download --dataset "US Equity Options" \
    --data-type "Trade" \
    --option-style "American" \
    --ticker "SPY" \
    --resolution "Minute" \
    --start "20240101" \
    --end "20240601"
```

### 方式2: 本地安装

```bash
# 克隆 Lean
git clone https://github.com/QuantConnect/Lean.git
cd Lean

# 安装依赖 (需要 .NET 6.0+)
dotnet build

# Python 环境
pip install pythonnet quantconnect-stubs
```

---

## 项目结构

```
TV2PY/
├── lean_integration/
│   ├── __init__.py
│   ├── adapter.py          # TV策略 → Lean策略 适配器
│   ├── indicators.py       # 指标包装器
│   ├── options/
│   │   ├── greeks.py       # Greeks 计算
│   │   ├── strategies.py   # 期权策略
│   │   └── pricing.py      # 定价模型
│   └── brokers/
│       ├── ib.py           # Interactive Brokers
│       └── alpaca.py       # Alpaca
├── strategies/             # 已有策略
└── lean_projects/          # Lean 项目目录
    ├── tpx_equity/         # TPX 股票版
    └── covered_call/       # 期权策略示例
```

---

## 使用方式

### 1. 将 TV2PY 策略转换为 Lean 策略

```python
# lean_integration/adapter.py

from tv2py.strategies import TPXSlingShotBacktest

class LeanAdapter:
    """将 TV2PY 策略适配到 Lean"""

    def __init__(self, tv2py_strategy):
        self.strategy = tv2py_strategy

    def generate_lean_code(self) -> str:
        """生成 Lean Python 策略代码"""
        return f'''
from AlgorithmImports import *

class TPXSlingShotAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # 添加股票
        self.symbol = self.AddEquity("SPY", Resolution.Hour).Symbol

        # TPX 参数
        self.tpx_length = {self.strategy.params['tpx_length']}
        self.tpx_smooth = {self.strategy.params['tpx_smooth']}
        self.atr_mult = {self.strategy.params['atr_mult']}

        # 自定义指标
        self.tpx = TPXIndicator(self.tpx_length, self.tpx_smooth)
        self.atr = self.ATR(self.symbol, 14)
        self.ema_55 = self.EMA(self.symbol, 55)
        self.ema_200 = self.EMA(self.symbol, 200)

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        bar = data[self.symbol]
        self.tpx.Update(bar)

        if not self.tpx.IsReady:
            return

        # 趋势判断
        uptrend = bar.Close > self.ema_200.Current.Value

        # TPX 穿越
        tpx_cross_up = self.tpx.Value > 0 and self.tpx.Previous < 0

        # 入场
        if tpx_cross_up and uptrend and not self.Portfolio.Invested:
            self.SetHoldings(self.symbol, 1.0)

            # ATR 止损
            stop_price = bar.Close - self.atr.Current.Value * self.atr_mult
            self.StopMarketOrder(self.symbol, -self.Portfolio[self.symbol].Quantity, stop_price)
'''
```

### 2. 期权策略示例

```python
# lean_projects/covered_call/main.py

from AlgorithmImports import *

class CoveredCallAlgorithm(QCAlgorithm):
    """备兑看涨期权策略"""

    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)

        # 添加股票
        equity = self.AddEquity("SPY", Resolution.Minute)
        self.symbol = equity.Symbol

        # 添加期权链
        option = self.AddOption("SPY", Resolution.Minute)
        option.SetFilter(lambda u: u.Strikes(-5, 5)
                                    .Expiration(timedelta(days=25), timedelta(days=35)))

        self.contract = None

    def OnData(self, data):
        # 如果没有持仓，买入股票
        if not self.Portfolio[self.symbol].Invested:
            self.SetHoldings(self.symbol, 0.5)

        # 如果持有股票但没有期权，卖出看涨期权
        if self.Portfolio[self.symbol].Invested and self.contract is None:
            chain = data.OptionChains.get(self.symbol)
            if chain:
                # 选择 OTM 看涨期权
                calls = [x for x in chain if x.Right == OptionRight.Call
                         and x.Strike > chain.Underlying.Price]

                if calls:
                    # 选择最近到期、Delta 约 0.3 的期权
                    self.contract = sorted(calls, key=lambda x: x.Expiry)[0]
                    self.Sell(self.contract.Symbol, 1)

    def OnOrderEvent(self, orderEvent):
        # 期权到期或平仓后重置
        if orderEvent.Symbol == self.contract?.Symbol:
            if orderEvent.Status == OrderStatus.Filled:
                self.contract = None
```

### 3. 运行回测

```bash
# 运行股票策略回测
lean backtest "lean_projects/tpx_equity"

# 运行期权策略回测
lean backtest "lean_projects/covered_call" --download-data

# 查看结果
lean report "lean_projects/covered_call"
```

---

## 实盘部署

### Interactive Brokers

```python
# config.json
{
    "environment": "live-interactive",
    "live-mode": true,
    "ib-account": "YOUR_ACCOUNT",
    "ib-user-name": "YOUR_USERNAME",
    "ib-password": "YOUR_PASSWORD",
    "ib-trading-mode": "paper"  # 或 "live"
}
```

```bash
# 启动实盘
lean live "lean_projects/tpx_equity" --brokerage "Interactive Brokers"
```

### Alpaca (免佣金)

```bash
lean live "lean_projects/tpx_equity" \
    --brokerage "Alpaca" \
    --alpaca-api-key "YOUR_KEY" \
    --alpaca-api-secret "YOUR_SECRET" \
    --alpaca-environment "paper"
```

---

## TV2PY 指标 → Lean 指标映射

| TV2PY / PyneCore | Lean |
|------------------|------|
| `ta.ema(close, 14)` | `self.EMA(symbol, 14)` |
| `ta.rsi(close, 14)` | `self.RSI(symbol, 14)` |
| `ta.atr(14)` | `self.ATR(symbol, 14)` |
| `ta.macd(12, 26, 9)` | `self.MACD(symbol, 12, 26, 9)` |
| `ta.bb(20, 2)` | `self.BB(symbol, 20, 2)` |
| 自定义 TPX | 需要创建自定义指标类 |

---

## 自定义指标示例

```python
# lean_integration/indicators.py

from QuantConnect.Indicators import PythonIndicator

class TPXIndicator(PythonIndicator):
    """TPX 指标 - Lean 版本"""

    def __init__(self, length: int, smooth: int):
        super().__init__()
        self.length = length
        self.smooth = smooth
        self.Name = f"TPX({length},{smooth})"

        self.highs = []
        self.lows = []
        self.closes = []
        self.values = []

    @property
    def IsReady(self) -> bool:
        return len(self.values) >= self.smooth

    @property
    def Previous(self) -> float:
        return self.values[-2] if len(self.values) >= 2 else 0

    def Update(self, bar) -> bool:
        self.highs.append(bar.High)
        self.lows.append(bar.Low)
        self.closes.append(bar.Close)

        if len(self.closes) < self.length:
            return False

        # 计算 TPX
        highest = max(self.highs[-self.length:])
        lowest = min(self.lows[-self.length:])
        range_val = highest - lowest

        if range_val == 0:
            raw_tpx = 0
        else:
            buying = self.closes[-1] - lowest
            selling = highest - self.closes[-1]
            raw_tpx = (buying - selling) / range_val * 100

        self.values.append(raw_tpx)

        # WMA 平滑
        if len(self.values) >= self.smooth:
            weights = list(range(1, self.smooth + 1))
            wma = sum(v * w for v, w in zip(self.values[-self.smooth:], weights)) / sum(weights)
            self.Value = wma
            return True

        return False
```

---

## 期权 Greeks 计算

```python
# lean_integration/options/greeks.py

from QuantConnect import *

class OptionAnalyzer:
    """期权分析工具"""

    def __init__(self, algorithm: QCAlgorithm):
        self.algo = algorithm

    def get_greeks(self, contract) -> dict:
        """获取期权 Greeks"""
        return {
            'delta': contract.Greeks.Delta,
            'gamma': contract.Greeks.Gamma,
            'theta': contract.Greeks.Theta,
            'vega': contract.Greeks.Vega,
            'rho': contract.Greeks.Rho,
            'iv': contract.ImpliedVolatility
        }

    def filter_by_delta(self, chain, target_delta: float, tolerance: float = 0.05):
        """按 Delta 筛选期权"""
        return [c for c in chain
                if abs(c.Greeks.Delta - target_delta) < tolerance]

    def iron_condor(self, chain, width: int = 5) -> dict:
        """构建铁鹰策略"""
        underlying_price = chain.Underlying.Price

        # 找到 ATM 行权价
        atm_strike = min(chain, key=lambda x: abs(x.Strike - underlying_price)).Strike

        # 构建四条腿
        return {
            'sell_put': atm_strike - width,
            'buy_put': atm_strike - 2 * width,
            'sell_call': atm_strike + width,
            'buy_call': atm_strike + 2 * width
        }
```

---

## 对比总结

| 场景 | 推荐引擎 |
|------|---------|
| 加密货币 | NautilusTrader |
| 美股/期权 | Lean Engine |
| A股 | 需要第三方数据 |
| 高频交易 | 自建系统 |

---

## 下一步

1. 安装 Lean CLI: `pip install lean`
2. 初始化: `lean init`
3. 下载数据: `lean data download`
4. 转换 TPX 策略到 Lean 格式
5. 运行回测验证

Sources:
- [Lean GitHub](https://github.com/QuantConnect/Lean)
- [Lean CLI](https://github.com/QuantConnect/lean-cli)
- [Lean Documentation](https://www.quantconnect.com/docs/v2/lean-engine)
