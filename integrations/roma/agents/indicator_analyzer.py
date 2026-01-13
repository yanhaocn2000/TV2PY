"""
ROMA Indicator Analyzer Agent

指标分析代理，专注于 Pine Script 指标的分析和转换。
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

# 添加 TV2PY 路径
TV2PY_PATH = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TV2PY_PATH))

from ..orchestrator import BaseAgent
from ..core.atomizer import AtomicTask, TaskType
from ..core.executor import PlanExecutionResult


class IndicatorAnalyzerAgent(BaseAgent):
    """
    指标分析代理

    分析 Pine Script 指标代码，提取关键信息。

    能力:
        - 解析 Pine Script 语法
        - 提取指标参数
        - 识别使用的内置函数
        - 评估转换复杂度
        - 生成 Python 转换建议

    使用示例:
        agent = IndicatorAnalyzerAgent()

        result = orchestrator.execute_with_agent(
            agent=agent,
            task_description="分析 SuperTrend 指标",
            context={"pine_code": pine_script_code}
        )
    """

    # Pine Script 内置函数映射
    BUILTIN_FUNCTIONS = {
        # 数学函数
        "math.abs": "abs",
        "math.sqrt": "np.sqrt",
        "math.pow": "np.power",
        "math.log": "np.log",
        "math.exp": "np.exp",
        "math.max": "np.maximum",
        "math.min": "np.minimum",
        "math.round": "np.round",
        "math.floor": "np.floor",
        "math.ceil": "np.ceil",
        "math.sign": "np.sign",

        # 技术分析函数
        "ta.sma": "talib.SMA",
        "ta.ema": "talib.EMA",
        "ta.wma": "talib.WMA",
        "ta.rsi": "talib.RSI",
        "ta.macd": "talib.MACD",
        "ta.bbands": "talib.BBANDS",
        "ta.atr": "talib.ATR",
        "ta.adx": "talib.ADX",
        "ta.cci": "talib.CCI",
        "ta.stoch": "talib.STOCH",
        "ta.obv": "talib.OBV",
        "ta.mfi": "talib.MFI",

        # 统计函数
        "ta.stdev": "np.std",
        "ta.variance": "np.var",
        "ta.correlation": "np.corrcoef",
        "ta.linreg": "stats.linregress",

        # 价格数据
        "ta.highest": "rolling_max",
        "ta.lowest": "rolling_min",
        "ta.pivothigh": "pivot_high",
        "ta.pivotlow": "pivot_low",
        "ta.crossover": "crossover",
        "ta.crossunder": "crossunder",
        "ta.change": "np.diff",
        "ta.tr": "true_range",
    }

    def __init__(self):
        super().__init__(name="IndicatorAnalyzerAgent")

    def prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """准备分析上下文"""
        prepared = context.copy()

        # 确保有 Pine Script 代码
        if "pine_code" not in prepared:
            prepared["pine_code"] = ""

        return prepared

    def customize_tasks(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Optional[List[AtomicTask]]:
        """自定义指标分析任务"""
        tasks = []
        task_counter = 0

        def make_id(prefix: str) -> str:
            nonlocal task_counter
            task_counter += 1
            return f"{prefix}_{task_counter:04d}"

        pine_code = context.get("pine_code", "")

        if pine_code:
            # Pine Script 解析任务
            parse_task = AtomicTask(
                id=make_id("pine_parse"),
                task_type=TaskType.PINE_PARSE,
                name="Parse Pine Script",
                description="解析 Pine Script 代码结构",
                inputs={"pine_code": pine_code},
                expected_outputs=["ast", "parsed_elements"],
                priority=100,
            )
            tasks.append(parse_task)

            # 转换任务
            convert_task = AtomicTask(
                id=make_id("pine_convert"),
                task_type=TaskType.PINE_CONVERT,
                name="Convert to Python",
                description="转换为 Python 代码",
                inputs={"ast_ref": parse_task.id},
                dependencies=[parse_task.id],
                expected_outputs=["python_code", "conversion_notes"],
                priority=90,
            )
            tasks.append(convert_task)

        # 代码生成任务
        gen_task = AtomicTask(
            id=make_id("code_gen"),
            task_type=TaskType.CODE_GENERATE,
            name="Generate Final Code",
            description="生成最终可用代码",
            inputs={"template": context.get("template", "indicator")},
            dependencies=[tasks[-1].id] if tasks else [],
            expected_outputs=["final_code", "test_code"],
            priority=80,
        )
        tasks.append(gen_task)

        return tasks if tasks else None

    def process_results(
        self,
        execution_result: PlanExecutionResult
    ) -> Dict[str, Any]:
        """处理分析结果"""
        output = {
            "agent": self.name,
            "success_rate": execution_result.success_rate,
        }

        # 收集各阶段结果
        for task_id, result in execution_result.results.items():
            if not result.success:
                continue

            for key, value in result.output.items():
                if key in ["ast", "parsed_elements"]:
                    output["parsed"] = value
                elif key == "python_code":
                    output["python_code"] = value
                elif key == "conversion_notes":
                    output["notes"] = value
                elif key == "final_code":
                    output["final_code"] = value

        return output

    def analyze_pine_code(self, code: str) -> Dict[str, Any]:
        """
        分析 Pine Script 代码

        提取:
        - 指标名称
        - 输入参数
        - 使用的函数
        - 复杂度评估
        """
        analysis = {
            "name": "",
            "version": "",
            "inputs": [],
            "functions_used": [],
            "variables": [],
            "plots": [],
            "complexity": "low",
            "conversion_difficulty": 1,
        }

        lines = code.split("\n")

        for line in lines:
            line = line.strip()

            # 版本检测
            if line.startswith("//@version"):
                analysis["version"] = line.split("=")[-1].strip()

            # 指标名称
            if "indicator(" in line:
                match = re.search(r'indicator\s*\(\s*["\']([^"\']+)["\']', line)
                if match:
                    analysis["name"] = match.group(1)

            # 输入参数
            if "input." in line or "input(" in line:
                match = re.search(r'(\w+)\s*=\s*input', line)
                if match:
                    param_name = match.group(1)
                    analysis["inputs"].append({
                        "name": param_name,
                        "line": line,
                    })

            # 函数使用
            for func in self.BUILTIN_FUNCTIONS.keys():
                if func in line:
                    if func not in analysis["functions_used"]:
                        analysis["functions_used"].append(func)

            # 绘图
            if any(p in line for p in ["plot(", "plotshape(", "plotchar(", "bgcolor(", "fill("]):
                analysis["plots"].append(line)

        # 评估复杂度
        func_count = len(analysis["functions_used"])
        input_count = len(analysis["inputs"])

        if func_count > 10 or input_count > 8:
            analysis["complexity"] = "high"
            analysis["conversion_difficulty"] = 3
        elif func_count > 5 or input_count > 4:
            analysis["complexity"] = "medium"
            analysis["conversion_difficulty"] = 2
        else:
            analysis["complexity"] = "low"
            analysis["conversion_difficulty"] = 1

        return analysis

    def suggest_python_mapping(self, functions: List[str]) -> Dict[str, str]:
        """建议 Python 函数映射"""
        mapping = {}
        for func in functions:
            if func in self.BUILTIN_FUNCTIONS:
                mapping[func] = self.BUILTIN_FUNCTIONS[func]
            else:
                mapping[func] = f"# TODO: Implement {func}"
        return mapping


class PineScriptConverter(IndicatorAnalyzerAgent):
    """
    Pine Script 转换器

    专注于将 Pine Script 代码转换为 Python。
    """

    def __init__(self):
        super().__init__()
        self.name = "PineScriptConverter"

    def convert(self, pine_code: str) -> Dict[str, Any]:
        """
        转换 Pine Script 到 Python

        Returns:
            Dict with:
                - python_code: 转换后的代码
                - analysis: 代码分析
                - notes: 转换注意事项
        """
        # 分析代码
        analysis = self.analyze_pine_code(pine_code)

        # 生成 Python 代码框架
        python_code = self._generate_python_template(analysis)

        # 生成注意事项
        notes = self._generate_conversion_notes(analysis)

        return {
            "python_code": python_code,
            "analysis": analysis,
            "notes": notes,
        }

    def _generate_python_template(self, analysis: Dict[str, Any]) -> str:
        """生成 Python 代码模板"""
        name = analysis.get("name", "CustomIndicator")
        class_name = "".join(word.capitalize() for word in name.split())

        inputs = analysis.get("inputs", [])
        functions = analysis.get("functions_used", [])

        # 生成参数类
        params_code = ""
        if inputs:
            params_code = f"""
@dataclass
class {class_name}Params:
    \"\"\"指标参数\"\"\"
"""
            for inp in inputs:
                param_name = inp["name"]
                params_code += f"    {param_name}: float = 14.0  # TODO: Set default\n"

        # 生成主类
        template = f'''"""
{name} Indicator
Converted from Pine Script by ROMA
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

{params_code}

class {class_name}:
    """
    {name} 指标

    Pine Script 版本: {analysis.get("version", "unknown")}

    Parameters:
        params: 指标参数
    """

    def __init__(self, params: {class_name}Params = None):
        self.params = params or {class_name}Params()

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> dict:
        """
        计算指标

        Args:
            high: 最高价数组
            low: 最低价数组
            close: 收盘价数组

        Returns:
            dict: 指标值
        """
        # TODO: Implement calculation logic
        # Functions to implement: {functions}

        result = {{
            "value": np.zeros_like(close),
        }}

        return result


# 测试代码
if __name__ == "__main__":
    import numpy as np

    # 生成测试数据
    n = 100
    close = np.random.randn(n).cumsum() + 100
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))

    # 计算指标
    indicator = {class_name}()
    result = indicator.calculate(high, low, close)
    print(f"Indicator result: {{result}}")
'''
        return template

    def _generate_conversion_notes(self, analysis: Dict[str, Any]) -> List[str]:
        """生成转换注意事项"""
        notes = []

        # 复杂度提示
        if analysis["complexity"] == "high":
            notes.append("该指标复杂度较高，建议分步骤转换")

        # 函数映射提示
        unmapped = []
        for func in analysis.get("functions_used", []):
            if func not in self.BUILTIN_FUNCTIONS:
                unmapped.append(func)

        if unmapped:
            notes.append(f"以下函数需要自定义实现: {unmapped}")

        # 绘图提示
        if analysis.get("plots"):
            notes.append(f"原指标包含 {len(analysis['plots'])} 个绘图，需要转换为 matplotlib 或其他可视化")

        return notes
