"""
工具定义和工具执行器模块

定义可用的工具，并提供工具执行逻辑。
"""

import json
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema格式的参数定义


# 可用工具列表
AVAILABLE_TOOLS = {
    "calculator": Tool(
        name="calculator",
        description="执行数学计算，支持基本运算（加、减、乘、除、幂等）",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '2 + 3 * 4' 或 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    ),
    "weather_query": Tool(
        name="weather_query",
        description="查询指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询天气的城市名称"
                },
                "date": {
                    "type": "string",
                    "description": "查询日期，格式为 YYYY-MM-DD，可选，默认为今天"
                }
            },
            "required": ["city"]
        }
    ),
    "search": Tool(
        name="search",
        description="在网络上搜索信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询关键词"
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回的最大结果数量，默认5",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    ),
    "get_current_time": Tool(
        name="get_current_time",
        description="获取当前时间",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，例如 'UTC', 'Asia/Shanghai'，可选"
                }
            },
            "required": []
        }
    ),
}


def get_tools_system_prompt() -> str:
    """生成工具描述的系统提示"""
    tools_desc = []
    for tool_name, tool in AVAILABLE_TOOLS.items():
        tools_desc.append(
            f"工具名称: {tool.name}\n"
            f"描述: {tool.description}\n"
            f"参数: {json.dumps(tool.parameters, ensure_ascii=False, indent=2)}\n"
        )
    
    return f"""你是一个可以调用工具的AI助手。可用的工具如下：

{chr(10).join(tools_desc)}

当需要调用工具时，请使用以下格式：
<tool_call>
{{
    "name": "工具名称",
    "arguments": {{
        "参数名": "参数值"
    }}
}}
</tool_call>

工具调用后，你将收到工具的执行结果。请根据结果继续回答用户的问题。"""


class ToolExecutor:
    """工具执行器"""
    
    def __init__(self):
        self.tools = AVAILABLE_TOOLS
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具调用
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            执行结果字典，包含 'success', 'result', 'error' 字段
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "result": None,
                "error": f"未知工具: {tool_name}"
            }
        
        try:
            if tool_name == "calculator":
                result = self._execute_calculator(arguments)
            elif tool_name == "weather_query":
                result = self._execute_weather_query(arguments)
            elif tool_name == "search":
                result = self._execute_search(arguments)
            elif tool_name == "get_current_time":
                result = self._execute_get_current_time(arguments)
            else:
                result = {"success": False, "error": f"工具 {tool_name} 未实现"}
            
            return result
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"执行工具时出错: {str(e)}"
            }
    
    def _execute_calculator(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行计算器工具"""
        expression = arguments.get("expression", "")
        if not expression:
            return {"success": False, "error": "缺少表达式参数"}
        
        try:
            # 简单的安全计算（仅支持基本运算）
            # 在实际应用中，应该使用更安全的表达式求值方法
            # 这里为了演示，我们支持一些基本函数
            safe_dict = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
            }
            
            # 替换一些常用函数名
            expr = expression.replace("sqrt", "sqrt").replace("^", "**")
            
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            return {
                "success": True,
                "result": str(result),
                "expression": expression
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"计算错误: {str(e)}"
            }
    
    def _execute_weather_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行天气查询工具（模拟）"""
        city = arguments.get("city", "")
        date = arguments.get("date", "今天")
        
        if not city:
            return {"success": False, "error": "缺少城市参数"}
        
        # 模拟天气数据
        weather_data = {
            "北京": {"temperature": "15°C", "condition": "晴朗", "humidity": "45%"},
            "上海": {"temperature": "18°C", "condition": "多云", "humidity": "60%"},
            "广州": {"temperature": "25°C", "condition": "小雨", "humidity": "75%"},
            "深圳": {"temperature": "26°C", "condition": "晴朗", "humidity": "55%"},
        }
        
        default_weather = {"temperature": "20°C", "condition": "晴朗", "humidity": "50%"}
        weather = weather_data.get(city, default_weather)
        
        return {
            "success": True,
            "result": f"{city}在{date}的天气：温度{weather['temperature']}，"
                      f"天气{weather['condition']}，湿度{weather['humidity']}",
            "city": city,
            "date": date,
            "details": weather
        }
    
    def _execute_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行搜索工具（模拟）"""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        
        if not query:
            return {"success": False, "error": "缺少查询参数"}
        
        # 模拟搜索结果
        results = [
            f"关于 '{query}' 的搜索结果 {i+1}: 这是一个相关的搜索结果..."
            for i in range(min(max_results, 5))
        ]
        
        return {
            "success": True,
            "result": f"找到 {len(results)} 条相关结果",
            "query": query,
            "results": results
        }
    
    def _execute_get_current_time(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """获取当前时间"""
        from datetime import datetime
        import pytz
        
        timezone_str = arguments.get("timezone", "UTC")
        try:
            if timezone_str == "UTC":
                tz = pytz.UTC
            else:
                tz = pytz.timezone(timezone_str)
            
            current_time = datetime.now(tz)
            return {
                "success": True,
                "result": current_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "timezone": timezone_str,
                "timestamp": current_time.timestamp()
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"时区错误: {str(e)}"
            }

