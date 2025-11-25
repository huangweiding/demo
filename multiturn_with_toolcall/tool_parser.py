"""
工具调用解析器

从模型输出中检测和解析工具调用请求。
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple


class ToolCallParser:
    """工具调用解析器"""
    
    # 工具调用格式的正则表达式
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
        re.DOTALL
    )
    
    def __init__(self):
        pass
    
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中解析工具调用
        
        Args:
            text: 包含工具调用的文本
            
        Returns:
            工具调用列表，每个元素包含 'name' 和 'arguments'
        """
        tool_calls = []
        matches = self.TOOL_CALL_PATTERN.findall(text)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append({
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"]
                    })
            except json.JSONDecodeError as e:
                # 如果JSON解析失败，尝试修复常见的格式问题
                try:
                    # 尝试修复单引号等问题
                    fixed_match = match.replace("'", '"')
                    tool_call = json.loads(fixed_match)
                    if "name" in tool_call and "arguments" in tool_call:
                        tool_calls.append({
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"]
                        })
                except:
                    continue
        
        return tool_calls
    
    def extract_tool_call_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        从文本中提取工具调用，并返回清理后的文本和工具调用列表
        
        Args:
            text: 包含工具调用的文本
            
        Returns:
            (清理后的文本, 工具调用列表)
        """
        tool_calls = self.parse_tool_calls(text)
        
        # 移除工具调用标记，保留其他文本
        cleaned_text = self.TOOL_CALL_PATTERN.sub('', text).strip()
        
        return cleaned_text, tool_calls
    
    def has_tool_calls(self, text: str) -> bool:
        """检查文本中是否包含工具调用"""
        return bool(self.TOOL_CALL_PATTERN.search(text))
    
    def format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """
        格式化工具执行结果为文本，用于返回给模型
        
        Args:
            tool_name: 工具名称
            result: 工具执行结果
            
        Returns:
            格式化的文本
        """
        if result.get("success", False):
            result_text = result.get("result", "")
            return f"<tool_result>\n工具: {tool_name}\n结果: {result_text}\n</tool_result>"
        else:
            error = result.get("error", "未知错误")
            return f"<tool_result>\n工具: {tool_name}\n错误: {error}\n</tool_result>"
    
    def format_tool_results(self, tool_calls: List[Dict[str, Any]], 
                           results: List[Dict[str, Any]]) -> str:
        """
        格式化多个工具调用结果为文本
        
        Args:
            tool_calls: 工具调用列表
            results: 对应的执行结果列表
            
        Returns:
            格式化的文本
        """
        formatted_results = []
        for tool_call, result in zip(tool_calls, results):
            formatted_results.append(
                self.format_tool_result(tool_call["name"], result)
            )
        
        return "\n\n".join(formatted_results)

