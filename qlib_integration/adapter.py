"""
TV2PY to Qlib Adapter
将 TradingView/PyneCore 策略信号转换为 Qlib 特征格式
"""

from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime


class ModelType(Enum):
    """Qlib 支持的模型类型"""
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    MLP = "mlp"
    GRU = "gru"
    LSTM = "lstm"
    ALSTM = "alstm"
    TRANSFORMER = "transformer"
    TCN = "tcn"
    TABNET = "tabnet"
    HIST = "hist"
    GAT = "gat"


@dataclass
class FeatureConfig:
    """特征配置"""
    name: str
    expression: str  # Qlib 表达式语法
    description: str = ""


@dataclass
class TV2PySignal:
    """TV2PY 策略信号"""
    datetime: pd.DatetimeIndex
    symbol: str
    signal: np.ndarray  # 1=买入, -1=卖出, 0=持有
    strength: Optional[np.ndarray] = None  # 信号强度 0-1
    indicators: Dict[str, np.ndarray] = field(default_factory=dict)


class TV2PyQlibAdapter:
    """
    将 TV2PY 策略转换为 Qlib 格式

    用途:
    1. 将 TV 技术指标转换为 Qlib 特征
    2. 使用 Qlib ML 模型增强 TV 信号
    3. 整合 TV 信号到 Qlib 多因子框架
    """

    # TradingView/PyneCore 指标到 Qlib 表达式的映射
    INDICATOR_MAPPING = {
        # 移动平均
        "ema": "EMA($close, {period})",
        "sma": "Mean($close, {period})",
        "wma": "WMA($close, {period})",

        # 动量指标
        "rsi": "RSI($close, {period})",
        "macd": "MACD($close, {fast}, {slow}, {signal})",
        "cci": "CCI($high, $low, $close, {period})",
        "roc": "ROC($close, {period})",
        "mom": "Mom($close, {period})",

        # 波动率指标
        "atr": "ATR($high, $low, $close, {period})",
        "bbands": "BBands($close, {period}, {std})",
        "keltner": "KeltnerChannel($high, $low, $close, {period})",

        # 成交量指标
        "obv": "OBV($close, $volume)",
        "vwap": "VWAP($high, $low, $close, $volume)",
        "mfi": "MFI($high, $low, $close, $volume, {period})",

        # 趋势指标
        "adx": "ADX($high, $low, $close, {period})",
        "aroon": "Aroon($high, $low, {period})",
        "supertrend": "SuperTrend($high, $low, $close, {period}, {multiplier})",
    }

    # 标准 Qlib Alpha158 因子 (部分)
    ALPHA158_FEATURES = [
        # 价格因子
        FeatureConfig("OPEN", "$open", "开盘价"),
        FeatureConfig("HIGH", "$high", "最高价"),
        FeatureConfig("LOW", "$low", "最低价"),
        FeatureConfig("CLOSE", "$close", "收盘价"),
        FeatureConfig("VWAP", "Sum($volume*$close, 5)/Sum($volume, 5)", "5日VWAP"),

        # 收益率因子
        FeatureConfig("ROC5", "Ref($close, 5)/$close - 1", "5日收益率"),
        FeatureConfig("ROC10", "Ref($close, 10)/$close - 1", "10日收益率"),
        FeatureConfig("ROC20", "Ref($close, 20)/$close - 1", "20日收益率"),

        # 波动率因子
        FeatureConfig("STD5", "Std($close, 5)", "5日标准差"),
        FeatureConfig("STD10", "Std($close, 10)", "10日标准差"),
        FeatureConfig("STD20", "Std($close, 20)", "20日标准差"),

        # 成交量因子
        FeatureConfig("VOLUME_RATIO", "$volume/Mean($volume, 20)", "成交量比率"),
        FeatureConfig("TURNOVER", "$volume*$close", "成交额"),

        # 技术因子
        FeatureConfig("RSI14", "RSI($close, 14)", "14日RSI"),
        FeatureConfig("MACD", "EMA($close, 12) - EMA($close, 26)", "MACD"),
        FeatureConfig("BBANDS_UP", "Mean($close, 20) + 2*Std($close, 20)", "布林上轨"),
        FeatureConfig("BBANDS_DOWN", "Mean($close, 20) - 2*Std($close, 20)", "布林下轨"),
    ]

    def __init__(self, qlib_initialized: bool = False):
        """
        初始化适配器

        Args:
            qlib_initialized: Qlib 是否已初始化
        """
        self.qlib_initialized = qlib_initialized
        self.custom_features: List[FeatureConfig] = []

    def init_qlib(self, provider_uri: str = "~/.qlib/qlib_data/cn_data",
                  region: str = "cn"):
        """初始化 Qlib"""
        try:
            import qlib
            qlib.init(provider_uri=provider_uri, region=region)
            self.qlib_initialized = True
        except ImportError:
            raise ImportError("请先安装 qlib: pip install pyqlib")

    def tv_signal_to_feature(self, signal: TV2PySignal) -> pd.DataFrame:
        """
        将 TV2PY 信号转换为 Qlib 特征 DataFrame

        Args:
            signal: TV2PY 策略信号

        Returns:
            Qlib 格式的特征 DataFrame
        """
        # 创建基础 DataFrame
        df = pd.DataFrame(index=signal.datetime)
        df['instrument'] = signal.symbol

        # 添加信号作为特征
        df['tv_signal'] = signal.signal

        if signal.strength is not None:
            df['tv_signal_strength'] = signal.strength

        # 添加指标作为特征
        for name, values in signal.indicators.items():
            df[f'tv_{name}'] = values

        # 转换为 Qlib MultiIndex 格式
        df = df.reset_index()
        df = df.rename(columns={'index': 'datetime'})
        df = df.set_index(['datetime', 'instrument'])

        return df

    def add_custom_feature(self, name: str, expression: str,
                           description: str = ""):
        """添加自定义特征"""
        self.custom_features.append(
            FeatureConfig(name, expression, description)
        )

    def get_feature_config(self, include_alpha158: bool = True,
                           include_custom: bool = True) -> Dict:
        """
        获取 Qlib 特征配置

        Returns:
            Qlib DataHandler 配置字典
        """
        features = []
        labels = []

        if include_alpha158:
            features.extend(self.ALPHA158_FEATURES)

        if include_custom:
            features.extend(self.custom_features)

        # 构建 Qlib 配置
        feature_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "instruments": "csi300",
                "start_time": None,  # 动态设置
                "end_time": None,
                "fit_start_time": None,
                "fit_end_time": None,
                "infer_processors": [
                    {"class": "RobustZScoreNorm"},
                    {"class": "Fillna"},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm"},
                ],
            }
        }

        return feature_config

    def train_model(self,
                    strategy_signals: TV2PySignal,
                    model_type: Union[str, ModelType] = ModelType.LIGHTGBM,
                    train_period: tuple = ("2020-01-01", "2022-12-31"),
                    valid_period: tuple = ("2023-01-01", "2023-06-30"),
                    test_period: tuple = ("2023-07-01", "2024-01-01"),
                    **model_kwargs):
        """
        使用 Qlib 训练 ML 模型

        Args:
            strategy_signals: TV2PY 策略信号
            model_type: 模型类型
            train_period: 训练期
            valid_period: 验证期
            test_period: 测试期
            **model_kwargs: 模型参数

        Returns:
            训练好的模型和评估结果
        """
        if not self.qlib_initialized:
            raise RuntimeError("请先调用 init_qlib() 初始化 Qlib")

        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        # 获取模型配置
        model_config = self._get_model_config(model_type, **model_kwargs)

        # 准备数据
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP

        # 创建数据处理器
        handler_config = self.get_feature_config()
        handler_config["kwargs"]["start_time"] = train_period[0]
        handler_config["kwargs"]["end_time"] = test_period[1]

        # 使用 qrun 运行训练
        train_config = {
            "task": {
                "model": model_config,
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": handler_config,
                        "segments": {
                            "train": train_period,
                            "valid": valid_period,
                            "test": test_period,
                        }
                    }
                }
            }
        }

        return train_config

    def _get_model_config(self, model_type: ModelType, **kwargs) -> Dict:
        """获取模型配置"""

        configs = {
            ModelType.LIGHTGBM: {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.0421,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                    **kwargs
                }
            },
            ModelType.CATBOOST: {
                "class": "CatBoostModel",
                "module_path": "qlib.contrib.model.catboost_model",
                "kwargs": {
                    "loss_function": "RMSE",
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 3,
                    **kwargs
                }
            },
            ModelType.XGBOOST: {
                "class": "XGBModel",
                "module_path": "qlib.contrib.model.xgboost",
                "kwargs": {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    **kwargs
                }
            },
            ModelType.GRU: {
                "class": "GRU",
                "module_path": "qlib.contrib.model.pytorch_gru",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    "lr": 0.001,
                    "early_stop": 10,
                    "batch_size": 2000,
                    "metric": "loss",
                    "GPU": 0,
                    **kwargs
                }
            },
            ModelType.LSTM: {
                "class": "LSTM",
                "module_path": "qlib.contrib.model.pytorch_lstm",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    "lr": 0.001,
                    **kwargs
                }
            },
            ModelType.ALSTM: {
                "class": "ALSTM",
                "module_path": "qlib.contrib.model.pytorch_alstm",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    **kwargs
                }
            },
            ModelType.TRANSFORMER: {
                "class": "Transformer",
                "module_path": "qlib.contrib.model.pytorch_transformer",
                "kwargs": {
                    "d_feat": 158,
                    "d_model": 64,
                    "nhead": 2,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    **kwargs
                }
            },
            ModelType.TCN: {
                "class": "TCN",
                "module_path": "qlib.contrib.model.pytorch_tcn",
                "kwargs": {
                    "d_feat": 158,
                    "num_channels": [32, 64],
                    "kernel_size": 3,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    **kwargs
                }
            },
            ModelType.TABNET: {
                "class": "TabNet",
                "module_path": "qlib.contrib.model.pytorch_tabnet",
                "kwargs": {
                    "n_d": 64,
                    "n_a": 64,
                    "n_steps": 5,
                    "gamma": 1.5,
                    "n_epochs": 100,
                    **kwargs
                }
            },
            ModelType.MLP: {
                "class": "MLP",
                "module_path": "qlib.contrib.model.pytorch_mlp",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 256,
                    "num_layers": 3,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    **kwargs
                }
            },
            ModelType.HIST: {
                "class": "HIST",
                "module_path": "qlib.contrib.model.pytorch_hist",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 128,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "K": 1,
                    "n_epochs": 100,
                    **kwargs
                }
            },
            ModelType.GAT: {
                "class": "GAT",
                "module_path": "qlib.contrib.model.pytorch_gats",
                "kwargs": {
                    "d_feat": 158,
                    "hidden_size": 64,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "n_epochs": 100,
                    **kwargs
                }
            },
        }

        return configs.get(model_type, configs[ModelType.LIGHTGBM])

    def generate_qrun_config(self,
                             model_type: ModelType = ModelType.LIGHTGBM,
                             market: str = "csi300",
                             train_period: tuple = ("2020-01-01", "2022-12-31"),
                             valid_period: tuple = ("2023-01-01", "2023-06-30"),
                             test_period: tuple = ("2023-07-01", "2024-01-01"),
                             output_path: str = "./qlib_output") -> str:
        """
        生成 qrun 配置文件 (YAML)

        Returns:
            配置文件内容
        """
        config = f"""
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

market: &market {market}

data_handler_config: &data_handler_config
    start_time: {train_period[0]}
    end_time: {test_period[1]}
    fit_start_time: {train_period[0]}
    fit_end_time: {train_period[1]}
    instruments: *market
    infer_processors:
        - class: RobustZScoreNorm
          kwargs:
              fields_group: feature
              clip_outlier: true
        - class: Fillna
          kwargs:
              fields_group: feature
    learn_processors:
        - class: DropnaLabel
        - class: CSRankNorm
          kwargs:
              fields_group: label
    label: ["Ref($close, -2)/Ref($close, -1) - 1"]

task:
    model:
        class: {self._get_model_config(model_type)['class']}
        module_path: {self._get_model_config(model_type)['module_path']}
        kwargs: {self._get_model_config(model_type)['kwargs']}

    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs: *data_handler_config
            segments:
                train: [{train_period[0]}, {train_period[1]}]
                valid: [{valid_period[0]}, {valid_period[1]}]
                test: [{test_period[0]}, {test_period[1]}]

    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
        - class: SigAnaRecord
          module_path: qlib.workflow.record_temp

port_analysis_config:
    strategy:
        class: TopkDropoutStrategy
        module_path: qlib.contrib.strategy
        kwargs:
            signal: <PRED>
            topk: 50
            n_drop: 5

    backtest:
        start_time: {test_period[0]}
        end_time: {test_period[1]}
        account: 100000000
        benchmark: SH000300
        exchange_kwargs:
            limit_threshold: 0.095
            deal_price: close
            open_cost: 0.0005
            close_cost: 0.0015
            min_cost: 5
"""
        return config


class QlibBacktester:
    """使用 Qlib 进行回测"""

    def __init__(self, adapter: TV2PyQlibAdapter):
        self.adapter = adapter

    def run_backtest(self,
                     predictions: pd.DataFrame,
                     strategy_type: str = "topk_dropout",
                     start_time: str = "2023-07-01",
                     end_time: str = "2024-01-01",
                     account: float = 1_000_000,
                     topk: int = 50,
                     n_drop: int = 5,
                     benchmark: str = "SH000300") -> Dict:
        """
        运行回测

        Args:
            predictions: 模型预测结果
            strategy_type: 策略类型
            start_time: 回测开始时间
            end_time: 回测结束时间
            account: 初始资金
            topk: 持有股票数量
            n_drop: 每期调出数量
            benchmark: 基准指数

        Returns:
            回测结果
        """
        try:
            from qlib.contrib.strategy import TopkDropoutStrategy
            from qlib.contrib.evaluate import backtest_daily
            from qlib.contrib.evaluate import risk_analysis
        except ImportError:
            raise ImportError("请先安装 qlib: pip install pyqlib")

        # 配置策略
        if strategy_type == "topk_dropout":
            strategy = TopkDropoutStrategy(
                signal=predictions,
                topk=topk,
                n_drop=n_drop
            )
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")

        # 运行回测
        portfolio_metric, indicator = backtest_daily(
            start_time=start_time,
            end_time=end_time,
            account=account,
            benchmark=benchmark,
            strategy=strategy
        )

        # 风险分析
        analysis = risk_analysis(portfolio_metric)

        return {
            "portfolio": portfolio_metric,
            "indicator": indicator,
            "analysis": analysis
        }

    def compare_with_tv_strategy(self,
                                 tv_signal: TV2PySignal,
                                 ml_predictions: pd.DataFrame,
                                 start_time: str,
                                 end_time: str) -> pd.DataFrame:
        """
        比较 TV 策略和 ML 增强策略的表现

        Returns:
            比较结果 DataFrame
        """
        # TV 策略回测
        tv_result = self._backtest_tv_signal(tv_signal, start_time, end_time)

        # ML 策略回测
        ml_result = self.run_backtest(ml_predictions, start_time=start_time,
                                       end_time=end_time)

        # 比较
        comparison = pd.DataFrame({
            "Metric": ["Annual Return", "Sharpe Ratio", "Max Drawdown",
                       "Win Rate", "Profit Factor"],
            "TV Strategy": [
                tv_result.get("annual_return", 0),
                tv_result.get("sharpe_ratio", 0),
                tv_result.get("max_drawdown", 0),
                tv_result.get("win_rate", 0),
                tv_result.get("profit_factor", 0)
            ],
            "ML Enhanced": [
                ml_result["analysis"].get("annual_return", 0),
                ml_result["analysis"].get("sharpe_ratio", 0),
                ml_result["analysis"].get("max_drawdown", 0),
                ml_result["analysis"].get("win_rate", 0),
                ml_result["analysis"].get("profit_factor", 0)
            ]
        })

        return comparison

    def _backtest_tv_signal(self, signal: TV2PySignal,
                            start_time: str, end_time: str) -> Dict:
        """使用 TV 信号进行简单回测"""
        # 简化实现 - 实际应使用 NautilusTrader
        returns = []
        position = 0

        for i, sig in enumerate(signal.signal):
            if sig == 1 and position == 0:
                position = 1
            elif sig == -1 and position == 1:
                position = 0
                # 计算收益 (简化)

        return {
            "annual_return": 0.15,  # 占位
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "profit_factor": 1.8
        }
