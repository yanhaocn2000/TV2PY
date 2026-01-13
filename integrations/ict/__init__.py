"""
TV2PY ICT (Inner Circle Trader) Integration

整合 smart-money-concepts 开源库 (https://github.com/joshyattridge/smart-money-concepts)
提供完整的 ICT 交易方法论实现。

功能:
    - Fair Value Gap (FVG) - 公允价值缺口
    - Swing Highs/Lows - 摆动高低点
    - Break of Structure (BOS) - 结构突破
    - Change of Character (CHoCH) - 性质改变
    - Order Blocks (OB) - 订单块
    - Liquidity - 流动性
    - Sessions - 交易时段 (伦敦、纽约、东京、悉尼)
    - Retracements - 回撤
    - Previous High/Low - 前高/前低

致谢:
    Based on smart-money-concepts by Josh Attridge
    https://github.com/joshyattridge/smart-money-concepts
    License: MIT
"""

from .smc import smc, ICTAnalyzer
from .sessions import TradingSessions, KillZones
from .signals import ICTSignalGenerator

__all__ = [
    "smc",
    "ICTAnalyzer",
    "TradingSessions",
    "KillZones",
    "ICTSignalGenerator",
]

__version__ = "1.0.0"
__based_on__ = "smart-money-concepts v0.0.26"
