# TV2PY ICT Integration

ICT (Inner Circle Trader) 方法论的完整 Python 实现。

## 致谢

Based on [smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts) by Josh Attridge (MIT License)

## 功能特性

### Smart Money Concepts (SMC)

| 功能 | 说明 |
|------|------|
| **Fair Value Gap (FVG)** | 公允价值缺口识别 |
| **Swing Highs/Lows** | 摆动高低点检测 |
| **Break of Structure (BOS)** | 结构突破 |
| **Change of Character (CHoCH)** | 性质改变 |
| **Order Blocks (OB)** | 订单块 (机构供需区) |
| **Liquidity** | 流动性区域 |
| **Previous High/Low** | 前高/前低 |
| **Retracements** | 回撤计算 |

### Trading Sessions (交易时段)

| 时段 | 时间 (UTC) |
|------|------------|
| Sydney | 21:00 - 06:00 |
| Tokyo | 00:00 - 09:00 |
| London | 07:00 - 16:00 |
| New York | 13:00 - 22:00 |

### Kill Zones (杀戮区)

| 杀戮区 | 时间 (UTC) | 说明 |
|--------|------------|------|
| Asian | 00:00 - 04:00 | 亚洲盘后段 |
| London Open | 06:00 - 09:00 | 伦敦开盘 |
| New York | 11:00 - 14:00 | 纽约开盘 |
| London Close | 14:00 - 16:00 | 伦敦收盘 |

## 快速开始

### 基本 SMC 分析

```python
from integrations.ict import smc

# 准备数据 (列名小写)
import pandas as pd
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 检测摆动高低点
swing_hl = smc.swing_highs_lows(df, swing_length=50)

# 检测公允价值缺口
fvg = smc.fvg(df)

# 检测订单块
order_blocks = smc.ob(df, swing_hl)

# 检测结构突破
bos_choch = smc.bos_choch(df, swing_hl)

# 检测流动性
liquidity = smc.liquidity(df, swing_hl)

# 检测回撤
retracements = smc.retracements(df, swing_hl)
```

### 使用 ICTAnalyzer

```python
from integrations.ict import ICTAnalyzer

# 创建分析器
analyzer = ICTAnalyzer(swing_length=50)

# 一键分析
result = analyzer.analyze(df)

# 获取结果
print(f"摆动点数量: {len(result.swing_highs_lows.dropna())}")
print(f"订单块数量: {len(result.order_blocks.dropna())}")
print(f"FVG数量: {len(result.fvg.dropna())}")

# 获取活跃订单块
bullish_ob = result.get_active_bullish_ob()
bearish_ob = result.get_active_bearish_ob()

# 获取未填补的 FVG
unfilled_fvg = result.get_unfilled_fvg()

# 获取市场偏向
bias = analyzer.get_bias(df)
print(f"市场偏向: {bias['bias']}, 置信度: {bias['confidence']:.1%}")
```

### 交易时段分析

```python
from integrations.ict import TradingSessions, KillZones

# 时段管理
sessions = TradingSessions()

# 检查是否在伦敦时段
from datetime import time
is_london = sessions.is_in_session("London", time(8, 30))

# 获取伦敦时段K线
london_candles = sessions.filter_session(df, "London")

# 获取时段高低点
london_hl = sessions.get_session_high_low(df, "London")

# 杀戮区
kz = KillZones()

# 检查是否在杀戮区
is_kz, kz_name = kz.is_in_kill_zone(time(7, 30))
print(f"在杀戮区: {is_kz}, 名称: {kz_name}")

# 分析各杀戮区表现
kz_analysis = kz.analyze_kill_zones(df)
for name, stats in kz_analysis.items():
    print(f"{name}: 平均波动 {stats['avg_range']:.2f}, 阳线比例 {stats['bullish_ratio']:.1%}")

# 获取最佳入场时间建议
optimal_times = kz.get_optimal_entry_times()
```

### 信号生成

```python
from integrations.ict import ICTSignalGenerator

# 创建信号生成器
generator = ICTSignalGenerator(
    swing_length=50,
    min_rr=2.0,           # 最小风险收益比
    use_kill_zones=True,  # 只在杀戮区生成信号
    ob_retest=True,       # 订单块回测入场
    fvg_entry=True,       # FVG 入场
)

# 生成信号
signals = generator.generate_signals(df)

for signal in signals:
    print(f"""
    信号: {signal.signal_type.value}
    入场类型: {signal.entry_type.value}
    入场价: {signal.entry_price:.2f}
    止损: {signal.stop_loss:.2f}
    止盈: {signal.take_profit:.2f}
    R:R: {signal.risk_reward_ratio:.2f}
    置信度: {signal.confidence:.1%}
    杀戮区: {signal.kill_zone}
    原因: {signal.reason}
    """)

# 回测信号
backtest = generator.backtest_signals(
    df,
    initial_capital=10000,
    risk_per_trade=0.02,
)

print(f"总交易数: {backtest['total_trades']}")
print(f"胜率: {backtest['win_rate']:.1%}")
print(f"总收益: {backtest['total_return']:.1%}")
```

## API 参考

### smc 类方法

| 方法 | 参数 | 返回 |
|------|------|------|
| `fvg(ohlc, join_consecutive)` | DataFrame, bool | FVG, Top, Bottom, MitigatedIndex |
| `swing_highs_lows(ohlc, swing_length)` | DataFrame, int | HighLow, Level |
| `bos_choch(ohlc, swing_hl, close_break)` | DataFrame, DataFrame, bool | BOS, CHOCH, Level, BrokenIndex |
| `ob(ohlc, swing_hl, close_mitigation)` | DataFrame, DataFrame, bool | OB, Top, Bottom, OBVolume, MitigatedIndex, Percentage |
| `liquidity(ohlc, swing_hl, range_percent)` | DataFrame, DataFrame, float | Liquidity, Level, End, Swept |
| `previous_high_low(ohlc, time_frame)` | DataFrame, str | PreviousHigh, PreviousLow, BrokenHigh, BrokenLow |
| `sessions(ohlc, session, ...)` | DataFrame, str, ... | Active, High, Low |
| `retracements(ohlc, swing_hl)` | DataFrame, DataFrame | Direction, CurrentRetracement%, DeepestRetracement% |

### ICTAnalyzer 类

```python
class ICTAnalyzer:
    def __init__(
        self,
        swing_length: int = 50,
        fvg_join_consecutive: bool = False,
        ob_close_mitigation: bool = False,
        liquidity_range_percent: float = 0.01,
    ):
        ...

    def analyze(self, ohlc, session=None) -> ICTAnalysisResult:
        """执行完整 ICT 分析"""

    def get_bias(self, ohlc) -> Dict:
        """获取市场偏向"""
```

### ICTSignalGenerator 类

```python
class ICTSignalGenerator:
    def __init__(
        self,
        swing_length: int = 50,
        min_rr: float = 2.0,
        use_kill_zones: bool = True,
        ob_retest: bool = True,
        fvg_entry: bool = True,
        min_confidence: float = 0.6,
    ):
        ...

    def generate_signals(self, ohlc, lookback=50) -> List[ICTSignal]:
        """生成交易信号"""

    def backtest_signals(self, ohlc, initial_capital=10000, risk_per_trade=0.02) -> Dict:
        """回测信号"""
```

## ICT 交易方法论

### 1. 确定市场偏向

使用 BOS/CHoCH 判断:
- **BOS (Break of Structure)**: 趋势延续信号
- **CHoCH (Change of Character)**: 趋势反转信号

### 2. 寻找入场点

优先级:
1. Order Block 回测
2. FVG 填补
3. 流动性扫荡后反转

### 3. 时间过滤

只在 Kill Zone 内交易:
- **London Open** (06:00-09:00): 最佳趋势交易
- **New York** (11:00-14:00): 高波动
- **London Close** (14:00-16:00): 可能反转

### 4. 目标设定

- 使用流动性区域作为目标
- 或使用 1:2 以上的风险收益比

## 示例

### 完整交易流程

```python
from integrations.ict import ICTAnalyzer, ICTSignalGenerator, KillZones
from datetime import datetime

# 1. 分析市场
analyzer = ICTAnalyzer()
result = analyzer.analyze(df)
bias = analyzer.get_bias(df)

print(f"市场偏向: {bias['bias']}")

# 2. 检查时间
kz = KillZones()
current_time = datetime.now().time()
is_kz, kz_name = kz.is_in_kill_zone(current_time)

if not is_kz:
    print("不在杀戮区，等待...")
else:
    print(f"在 {kz_name}，寻找机会...")

    # 3. 生成信号
    generator = ICTSignalGenerator(min_rr=2.0)
    signal = generator.get_current_signal(df)

    if signal:
        print(f"信号: {signal.signal_type.value}")
        print(f"入场: {signal.entry_price:.2f}")
        print(f"止损: {signal.stop_loss:.2f}")
        print(f"止盈: {signal.take_profit:.2f}")
```

## 与 TV2PY 其他模块集成

```python
# 与 ROMA 集成
from integrations.roma import ROMAOrchestrator, TradingMetaAgent
from integrations.ict import ICTAnalyzer

# 创建 ICT 增强的交易代理
class ICTTradingAgent(TradingMetaAgent):
    def __init__(self):
        super().__init__(indicators=["smc"])
        self.ict_analyzer = ICTAnalyzer()

    def process_results(self, execution_result):
        # 使用 ICT 分析增强结果
        ...
```

## 参考资料

- [ICT YouTube Channel](https://www.youtube.com/c/InnerCircleTrader)
- [smart-money-concepts GitHub](https://github.com/joshyattridge/smart-money-concepts)
