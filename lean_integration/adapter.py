"""
TV2PY → Lean Engine 适配器

将 TV2PY/PyneCore 策略转换为 QuantConnect Lean 格式
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import textwrap


@dataclass
class LeanStrategyConfig:
    """Lean 策略配置"""
    name: str
    symbols: List[str]
    resolution: str = "Hour"  # Minute, Hour, Daily
    start_date: str = "2020, 1, 1"
    end_date: str = "2024, 1, 1"
    cash: int = 100000
    asset_type: str = "Equity"  # Equity, Option, Future, Forex, Crypto


class TV2PyToLeanAdapter:
    """
    将 TV2PY 策略转换为 Lean 策略

    用法:
        adapter = TV2PyToLeanAdapter()
        lean_code = adapter.convert(tv2py_strategy, config)
    """

    # 指标映射表
    INDICATOR_MAP = {
        'ta.ema': 'self.EMA({symbol}, {length})',
        'ta.sma': 'self.SMA({symbol}, {length})',
        'ta.rsi': 'self.RSI({symbol}, {length})',
        'ta.atr': 'self.ATR({symbol}, {length})',
        'ta.macd': 'self.MACD({symbol}, {fast}, {slow}, {signal})',
        'ta.bb': 'self.BB({symbol}, {length}, {mult})',
        'ta.adx': 'self.ADX({symbol}, {length})',
        'ta.cci': 'self.CCI({symbol}, {length})',
        'ta.stoch': 'self.STO({symbol}, {length})',
        'ta.wma': 'self.WMA({symbol}, {length})',
        'ta.hma': 'self.HMA({symbol}, {length})',
    }

    def convert(self, strategy_params: Dict, config: LeanStrategyConfig) -> str:
        """
        将 TV2PY 策略参数转换为 Lean 策略代码

        Args:
            strategy_params: 策略参数字典
            config: Lean 配置

        Returns:
            Lean Python 策略代码
        """
        template = self._get_template(config.asset_type)
        return template.format(
            class_name=config.name.replace(" ", ""),
            start_date=config.start_date,
            end_date=config.end_date,
            cash=config.cash,
            symbol=config.symbols[0],
            resolution=config.resolution,
            params=self._format_params(strategy_params),
            indicators=self._generate_indicators(strategy_params, config.symbols[0]),
            logic=self._generate_logic(strategy_params),
        )

    def _get_template(self, asset_type: str) -> str:
        """获取策略模板"""
        if asset_type == "Option":
            return self._get_option_template()
        return self._get_equity_template()

    def _get_equity_template(self) -> str:
        """股票策略模板"""
        return textwrap.dedent('''
            from AlgorithmImports import *

            class {class_name}(QCAlgorithm):
                """
                TV2PY 策略 - Lean 版本
                自动生成，请勿手动修改
                """

                def Initialize(self):
                    # 基本设置
                    self.SetStartDate({start_date})
                    self.SetEndDate({end_date})
                    self.SetCash({cash})

                    # 添加股票
                    self.symbol = self.AddEquity("{symbol}", Resolution.{resolution}).Symbol

                    # 策略参数
                    {params}

                    # 技术指标
                    {indicators}

                    # 状态变量
                    self.previous_tpx = 0

                def OnData(self, data):
                    if not data.ContainsKey(self.symbol):
                        return

                    bar = data[self.symbol]

                    # 检查指标是否就绪
                    if not self._indicators_ready():
                        return

                    {logic}

                def _indicators_ready(self) -> bool:
                    """检查所有指标是否就绪"""
                    return all([
                        self.ema_fast.IsReady,
                        self.ema_slow.IsReady,
                        self.atr.IsReady,
                    ])

                def OnOrderEvent(self, orderEvent):
                    """订单事件处理"""
                    if orderEvent.Status == OrderStatus.Filled:
                        self.Debug(f"Order filled: {{orderEvent}}")
        ''')

    def _get_option_template(self) -> str:
        """期权策略模板"""
        return textwrap.dedent('''
            from AlgorithmImports import *

            class {class_name}(QCAlgorithm):
                """
                TV2PY 期权策略 - Lean 版本
                """

                def Initialize(self):
                    self.SetStartDate({start_date})
                    self.SetEndDate({end_date})
                    self.SetCash({cash})

                    # 添加标的股票
                    equity = self.AddEquity("{symbol}", Resolution.{resolution})
                    self.symbol = equity.Symbol

                    # 添加期权链
                    option = self.AddOption("{symbol}", Resolution.Minute)
                    option.SetFilter(self._option_filter)
                    self.option_symbol = option.Symbol

                    # 策略参数
                    {params}

                    # 技术指标
                    {indicators}

                    # 期权状态
                    self.current_contract = None

                def _option_filter(self, universe):
                    """期权筛选器"""
                    return universe.Strikes(-10, 10) \\
                                   .Expiration(timedelta(days=20), timedelta(days=40))

                def OnData(self, data):
                    if not data.ContainsKey(self.symbol):
                        return

                    # 获取期权链
                    chain = data.OptionChains.get(self.option_symbol)
                    if not chain:
                        return

                    bar = data[self.symbol]
                    underlying_price = bar.Close

                    {logic}

                def GetATMOption(self, chain, right: OptionRight):
                    """获取 ATM 期权"""
                    options = [x for x in chain if x.Right == right]
                    if not options:
                        return None
                    return min(options, key=lambda x: abs(x.Strike - chain.Underlying.Price))

                def GetOTMOption(self, chain, right: OptionRight, otm_percent: float = 0.02):
                    """获取 OTM 期权"""
                    underlying = chain.Underlying.Price
                    if right == OptionRight.Call:
                        target = underlying * (1 + otm_percent)
                        options = [x for x in chain if x.Right == right and x.Strike >= target]
                    else:
                        target = underlying * (1 - otm_percent)
                        options = [x for x in chain if x.Right == right and x.Strike <= target]

                    if not options:
                        return None
                    return min(options, key=lambda x: abs(x.Strike - target))
        ''')

    def _format_params(self, params: Dict) -> str:
        """格式化参数"""
        lines = []
        for key, value in params.items():
            if isinstance(value, str):
                lines.append(f"self.{key} = '{value}'")
            else:
                lines.append(f"self.{key} = {value}")
        return "\n                    ".join(lines)

    def _generate_indicators(self, params: Dict, symbol: str) -> str:
        """生成指标代码"""
        indicators = []

        # 常用指标
        if 'ema_fast' in params:
            indicators.append(f"self.ema_fast = self.EMA(self.symbol, {params['ema_fast']})")
        if 'ema_medium' in params:
            indicators.append(f"self.ema_medium = self.EMA(self.symbol, {params['ema_medium']})")
        if 'ema_slow' in params:
            indicators.append(f"self.ema_slow = self.EMA(self.symbol, {params['ema_slow']})")
        if 'atr_length' in params:
            indicators.append(f"self.atr = self.ATR(self.symbol, {params['atr_length']})")
        if 'rsi_length' in params:
            indicators.append(f"self.rsi = self.RSI(self.symbol, {params['rsi_length']})")

        # 自定义 TPX 指标
        if 'tpx_length' in params:
            indicators.append(f"self.tpx = TPXIndicator({params['tpx_length']}, {params.get('tpx_smooth', 5)})")
            indicators.append("self.RegisterIndicator(self.symbol, self.tpx, Resolution.Hour)")

        return "\n                    ".join(indicators) if indicators else "pass"

    def _generate_logic(self, params: Dict) -> str:
        """生成交易逻辑"""
        return textwrap.dedent('''
                    # 更新自定义指标
                    if hasattr(self, 'tpx'):
                        self.tpx.Update(bar)

                    # 趋势判断
                    uptrend = bar.Close > self.ema_slow.Current.Value
                    downtrend = bar.Close < self.ema_slow.Current.Value

                    # TPX 信号
                    tpx_value = self.tpx.Value if hasattr(self, 'tpx') else 0
                    tpx_cross_up = tpx_value > 0 and self.previous_tpx <= 0
                    tpx_cross_down = tpx_value < 0 and self.previous_tpx >= 0
                    self.previous_tpx = tpx_value

                    # 入场逻辑
                    if tpx_cross_up and uptrend and not self.Portfolio.Invested:
                        self.SetHoldings(self.symbol, 1.0)

                        # ATR 止损
                        stop_price = bar.Close - self.atr.Current.Value * self.atr_mult
                        self.StopMarketOrder(self.symbol, -self.Portfolio[self.symbol].Quantity, stop_price)

                    elif tpx_cross_down and downtrend and self.Portfolio.Invested:
                        self.Liquidate(self.symbol)
        ''')


class LeanProjectGenerator:
    """
    生成 Lean 项目结构

    用法:
        generator = LeanProjectGenerator()
        generator.create_project("tpx_equity", strategy_code)
    """

    def __init__(self, base_path: str = "./lean_projects"):
        self.base_path = base_path

    def create_project(self, name: str, main_code: str, config: Optional[Dict] = None):
        """
        创建 Lean 项目

        Args:
            name: 项目名称
            main_code: 策略代码
            config: 配置覆盖
        """
        import os

        project_path = os.path.join(self.base_path, name)
        os.makedirs(project_path, exist_ok=True)

        # 主策略文件
        with open(os.path.join(project_path, "main.py"), "w") as f:
            f.write(main_code)

        # 配置文件
        default_config = {
            "algorithm-type-name": name,
            "algorithm-language": "Python",
            "parameters": {},
            "description": f"TV2PY converted strategy: {name}"
        }
        if config:
            default_config.update(config)

        import json
        with open(os.path.join(project_path, "config.json"), "w") as f:
            json.dump(default_config, f, indent=4)

        # 自定义指标文件
        indicators_code = self._get_custom_indicators()
        with open(os.path.join(project_path, "indicators.py"), "w") as f:
            f.write(indicators_code)

        print(f"项目已创建: {project_path}")
        print(f"运行: lean backtest \"{project_path}\"")

    def _get_custom_indicators(self) -> str:
        """获取自定义指标代码"""
        return textwrap.dedent('''
            """
            自定义指标 - Lean 版本
            """
            from QuantConnect.Indicators import PythonIndicator
            from collections import deque


            class TPXIndicator(PythonIndicator):
                """
                TPX (Trading Pressure Index) 指标

                与 TradingView 对齐的实现
                """

                def __init__(self, length: int = 14, smooth: int = 5):
                    super().__init__()
                    self.length = length
                    self.smooth = smooth
                    self.Name = f"TPX({length},{smooth})"
                    self.WarmUpPeriod = length + smooth

                    self.highs = deque(maxlen=length)
                    self.lows = deque(maxlen=length)
                    self.closes = deque(maxlen=length)
                    self.raw_values = deque(maxlen=smooth)

                    self.Value = 0
                    self.Previous = 0

                @property
                def IsReady(self) -> bool:
                    return len(self.raw_values) >= self.smooth

                def Update(self, input) -> bool:
                    """更新指标"""
                    # 支持 TradeBar 或 IndicatorDataPoint
                    if hasattr(input, 'High'):
                        high = float(input.High)
                        low = float(input.Low)
                        close = float(input.Close)
                    else:
                        # IndicatorDataPoint
                        high = low = close = float(input.Value)

                    self.highs.append(high)
                    self.lows.append(low)
                    self.closes.append(close)

                    if len(self.closes) < self.length:
                        return False

                    # 计算原始 TPX
                    highest = max(self.highs)
                    lowest = min(self.lows)
                    range_val = highest - lowest

                    if range_val == 0:
                        raw_tpx = 0
                    else:
                        buying_pressure = close - lowest
                        selling_pressure = highest - close
                        raw_tpx = (buying_pressure - selling_pressure) / range_val * 100

                    self.raw_values.append(raw_tpx)

                    # WMA 平滑
                    if len(self.raw_values) >= self.smooth:
                        self.Previous = self.Value
                        weights = list(range(1, self.smooth + 1))
                        total_weight = sum(weights)
                        self.Value = sum(v * w for v, w in zip(self.raw_values, weights)) / total_weight
                        return True

                    return False


            class TilsonT3Indicator(PythonIndicator):
                """
                Tilson T3 移动平均线

                与 TradingView 对齐
                """

                def __init__(self, length: int = 8, factor: float = 0.7):
                    super().__init__()
                    self.length = length
                    self.factor = factor
                    self.Name = f"T3({length},{factor})"
                    self.WarmUpPeriod = length * 6

                    self.closes = deque(maxlen=length * 6 + 1)

                @property
                def IsReady(self) -> bool:
                    return len(self.closes) >= self.WarmUpPeriod

                def Update(self, input) -> bool:
                    close = float(input.Value) if hasattr(input, 'Value') else float(input.Close)
                    self.closes.append(close)

                    if not self.IsReady:
                        return False

                    # 计算 6 层 EMA
                    data = list(self.closes)
                    e1 = self._ema(data, self.length)
                    e2 = self._ema(e1, self.length)
                    e3 = self._ema(e2, self.length)
                    e4 = self._ema(e3, self.length)
                    e5 = self._ema(e4, self.length)
                    e6 = self._ema(e5, self.length)

                    # T3 系数
                    c1 = -(self.factor ** 3)
                    c2 = 3 * (self.factor ** 2) + 3 * (self.factor ** 3)
                    c3 = -6 * (self.factor ** 2) - 3 * self.factor - 3 * (self.factor ** 3)
                    c4 = 1 + 3 * self.factor + (self.factor ** 3) + 3 * (self.factor ** 2)

                    self.Value = c1 * e6[-1] + c2 * e5[-1] + c3 * e4[-1] + c4 * e3[-1]
                    return True

                def _ema(self, data: list, length: int) -> list:
                    """计算 EMA"""
                    alpha = 2.0 / (length + 1)
                    result = [data[0]]
                    for i in range(1, len(data)):
                        result.append(alpha * data[i] + (1 - alpha) * result[-1])
                    return result
        ''')


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    # TPX 策略参数
    tpx_params = {
        'tpx_length': 14,
        'tpx_smooth': 5,
        'ema_fast': 55,
        'ema_medium': 89,
        'ema_slow': 200,
        'atr_length': 14,
        'atr_mult': 2.0,
        'rsi_length': 14,
    }

    # 配置
    config = LeanStrategyConfig(
        name="TPX SlingShot",
        symbols=["SPY"],
        resolution="Hour",
        start_date="2020, 1, 1",
        end_date="2024, 1, 1",
        cash=100000,
        asset_type="Equity"
    )

    # 转换
    adapter = TV2PyToLeanAdapter()
    lean_code = adapter.convert(tpx_params, config)

    print("=" * 60)
    print("生成的 Lean 策略代码:")
    print("=" * 60)
    print(lean_code)

    # 生成项目
    # generator = LeanProjectGenerator()
    # generator.create_project("tpx_slingshot", lean_code)
