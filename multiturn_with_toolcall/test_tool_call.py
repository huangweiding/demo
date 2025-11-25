"""
简单的工具调用测试脚本

用于验证工具调用解析和执行功能是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import ToolExecutor, AVAILABLE_TOOLS
from tool_parser import ToolCallParser


def test_tool_parser():
    """测试工具调用解析器"""
    print("=" * 50)
    print("测试工具调用解析器")
    print("=" * 50)
    
    parser = ToolCallParser()
    
    # 测试用例1：正确的工具调用格式
    text1 = """我需要计算一下。
<tool_call>
{
    "name": "calculator",
    "arguments": {
        "expression": "25 * 37 + 128"
    }
}
</tool_call>
"""
    
    print("\n测试用例1：正确的工具调用格式")
    print(f"输入文本:\n{text1}")
    tool_calls = parser.parse_tool_calls(text1)
    print(f"解析结果: {tool_calls}")
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "calculator"
    print("✓ 测试通过")
    
    # 测试用例2：多个工具调用
    text2 = """<tool_call>
{"name": "weather_query", "arguments": {"city": "北京"}}
</tool_call>
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>
"""
    
    print("\n测试用例2：多个工具调用")
    tool_calls = parser.parse_tool_calls(text2)
    print(f"解析结果: {tool_calls}")
    assert len(tool_calls) == 2
    print("✓ 测试通过")
    
    # 测试用例3：无工具调用
    text3 = "这是一个普通的回答，没有工具调用。"
    tool_calls = parser.parse_tool_calls(text3)
    assert len(tool_calls) == 0
    print("\n测试用例3：无工具调用 - ✓ 测试通过")


def test_tool_executor():
    """测试工具执行器"""
    print("\n" + "=" * 50)
    print("测试工具执行器")
    print("=" * 50)
    
    executor = ToolExecutor()
    
    # 测试计算器
    print("\n测试计算器工具")
    result = executor.execute("calculator", {"expression": "25 * 37 + 128"})
    print(f"结果: {result}")
    assert result["success"] == True
    assert result["result"] == "1053"
    print("✓ 测试通过")
    
    # 测试天气查询
    print("\n测试天气查询工具")
    result = executor.execute("weather_query", {"city": "北京"})
    print(f"结果: {result}")
    assert result["success"] == True
    assert "北京" in result["result"]
    print("✓ 测试通过")
    
    # 测试搜索工具
    print("\n测试搜索工具")
    result = executor.execute("search", {"query": "机器学习", "max_results": 3})
    print(f"结果: {result}")
    assert result["success"] == True
    print("✓ 测试通过")
    
    # 测试时间工具
    print("\n测试时间工具")
    result = executor.execute("get_current_time", {"timezone": "UTC"})
    print(f"结果: {result}")
    assert result["success"] == True
    print("✓ 测试通过")
    
    # 测试错误情况
    print("\n测试错误情况：未知工具")
    result = executor.execute("unknown_tool", {})
    print(f"结果: {result}")
    assert result["success"] == False
    print("✓ 测试通过")


def test_integration():
    """集成测试：解析 + 执行"""
    print("\n" + "=" * 50)
    print("集成测试：解析 + 执行")
    print("=" * 50)
    
    parser = ToolCallParser()
    executor = ToolExecutor()
    
    # 模拟模型输出
    model_output = """让我帮你计算一下。
<tool_call>
{
    "name": "calculator",
    "arguments": {
        "expression": "sqrt(144) + 5 * 3"
    }
}
</tool_call>
"""
    
    print(f"\n模型输出:\n{model_output}")
    
    # 解析工具调用
    tool_calls = parser.parse_tool_calls(model_output)
    print(f"\n解析到的工具调用: {tool_calls}")
    
    # 执行工具
    results = []
    for tool_call in tool_calls:
        result = executor.execute(tool_call["name"], tool_call["arguments"])
        results.append(result)
        print(f"\n工具执行结果: {result}")
    
    # 格式化结果
    formatted_results = parser.format_tool_results(tool_calls, results)
    print(f"\n格式化后的结果:\n{formatted_results}")
    
    print("\n✓ 集成测试通过")


if __name__ == "__main__":
    try:
        test_tool_parser()
        test_tool_executor()
        test_integration()
        print("\n" + "=" * 50)
        print("所有测试通过！✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

