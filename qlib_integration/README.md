# Qlib Integration for TV2PY

[Qlib](https://github.com/microsoft/qlib) 是微软开源的 AI 量化投资平台，提供 40+ 预训练模型和自动化因子挖掘。

## 安装

```bash
# 推荐使用 conda
conda create -n qlib python=3.10
conda activate qlib

# 安装 Qlib
pip install pyqlib

# 初始化 Qlib (下载基础数据)
python -c "import qlib; qlib.init()"
```

## 数据准备

### 中国 A 股数据
```bash
# 下载 A 股数据 (沪深300)
python -m qlib.run.get_data qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data

# 初始化
import qlib
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
```

### 美股数据
```bash
# 下载美股数据
python -m qlib.run.get_data qlib_data_us --target_dir ~/.qlib/qlib_data/us_data

# 初始化
import qlib
qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region="us")
```

## 核心功能

### 1. 预训练模型 (40+)

| 模型 | 类型 | 特点 |
|-----|------|------|
| LightGBM | Tree | 快速, 基线模型 |
| CatBoost | Tree | 类别特征处理好 |
| XGBoost | Tree | 准确率高 |
| MLP | NN | 简单神经网络 |
| GRU | RNN | 序列建模 |
| LSTM | RNN | 长期依赖 |
| ALSTM | RNN | 注意力 LSTM |
| Transformer | Attention | 并行计算 |
| TCN | CNN | 时序卷积 |
| TabNet | Hybrid | 可解释性 |
| HIST | GNN | 图神经网络 |
| GATs | GNN | 图注意力 |

### 2. Alpha 因子挖掘

```python
from qlib_integration.alpha_factors import AlphaFactorMiner

# 自动挖掘因子
miner = AlphaFactorMiner()
factors = miner.mine_factors(
    stock_pool="CSI300",
    start_date="2020-01-01",
    end_date="2024-01-01"
)

# 评估因子
for factor in factors:
    print(f"{factor.name}: IC={factor.ic:.4f}, IR={factor.ir:.4f}")
```

### 3. 模型训练

```python
from qlib_integration.adapter import TV2PyQlibAdapter

# 将 TV2PY 策略转换为 Qlib 特征
adapter = TV2PyQlibAdapter()

# 使用 LightGBM 训练
model = adapter.train_model(
    strategy_signals=signals,  # 来自 TV2PY 策略
    model_type="lightgbm",
    train_period=("2020-01-01", "2022-12-31"),
    valid_period=("2023-01-01", "2023-06-30")
)

# 预测
predictions = model.predict(test_data)
```

### 4. 回测

```python
from qlib.backtest import backtest
from qlib.contrib.strategy import TopkDropoutStrategy

# 使用 Qlib 回测
portfolio, report = backtest(
    pred=predictions,
    strategy=TopkDropoutStrategy(
        topk=50,        # 持有前50只
        n_drop=5,       # 每期调出5只
    ),
    start_time="2023-07-01",
    end_time="2024-01-01"
)
```

## TV2PY + Qlib 集成方式

```
┌─────────────────────────────────────────────────────────────┐
│                      TV2PY 策略                              │
│  (TradingView Pine Script → Python/PyneCore)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Qlib Integration                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 信号特征化    │  │ ML 模型增强   │  │ 组合优化     │       │
│  │ TV信号→因子  │  │ 40+ 预训练   │  │ 风险控制     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 回测/实盘执行                                 │
│  NautilusTrader / Lean / Qlib Backtest                       │
└─────────────────────────────────────────────────────────────┘
```

## RD-Agent (2025新功能)

Qlib 最新集成了 [RD-Agent](https://github.com/microsoft/RD-Agent)，使用 LLM 自动化因子发现：

```python
from qlib.contrib.rd_agent import RDAgent

# LLM 驱动的自动因子挖掘
agent = RDAgent(llm_model="gpt-4")
new_factors = agent.discover_factors(
    market="cn",
    objective="momentum",
    max_iterations=100
)
```

## 使用场景

| 场景 | TV2PY 作用 | Qlib 作用 |
|-----|-----------|----------|
| 技术指标策略 | 计算指标信号 | 作为ML特征 |
| 机器学习预测 | 提供基础特征 | 训练预测模型 |
| 多因子模型 | 提供技术因子 | 因子合成优化 |
| 组合管理 | 单股票信号 | 全市场组合 |
