"""
多轮对话处理器

处理包含工具调用的多轮对话，包括：
1. 检测工具调用请求
2. 执行工具
3. 将工具结果返回给模型
4. 继续对话
"""

from typing import List, Dict, Any, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools import ToolExecutor, get_tools_system_prompt
from tool_parser import ToolCallParser


class MultiTurnHandler:
    """多轮对话处理器"""
    
    def __init__(self, max_turns: int = 5, max_tool_calls_per_turn: int = 3):
        """
        Args:
            max_turns: 最大对话轮数
            max_tool_calls_per_turn: 每轮最大工具调用次数
        """
        self.tool_executor = ToolExecutor()
        self.tool_parser = ToolCallParser()
        self.max_turns = max_turns
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
    
    def process_conversation(self, 
                            initial_prompt: str,
                            model_generate_fn,
                            tokenizer) -> Dict[str, Any]:
        """
        处理多轮对话，包括工具调用
        
        Args:
            initial_prompt: 初始用户提示
            model_generate_fn: 模型生成函数，接受 (prompt, tokenizer) 返回生成文本
            tokenizer: 分词器
            
        Returns:
            对话结果字典，包含：
            - conversation_history: 完整的对话历史
            - final_response: 最终响应
            - tool_calls_made: 所有工具调用
            - tool_results: 所有工具执行结果
            - num_turns: 对话轮数
        """
        conversation_history = []
        tool_calls_made = []
        tool_results = []
        
        # 添加系统提示（包含工具描述）
        system_prompt = get_tools_system_prompt()
        current_prompt = f"{system_prompt}\n\n<|user|> {initial_prompt}\n<|assistant|>"
        
        turn_count = 0
        
        while turn_count < self.max_turns:
            turn_count += 1
            
            # 生成模型响应
            response = model_generate_fn(current_prompt, tokenizer)
            
            # 检测工具调用
            cleaned_response, tool_calls = self.tool_parser.extract_tool_call_text(response)
            
            # 记录对话历史
            conversation_history.append({
                "turn": turn_count,
                "prompt": current_prompt,
                "response": response,
                "cleaned_response": cleaned_response,
                "tool_calls": tool_calls
            })
            
            # 如果没有工具调用，对话结束
            if not tool_calls:
                return {
                    "conversation_history": conversation_history,
                    "final_response": cleaned_response or response,
                    "tool_calls_made": tool_calls_made,
                    "tool_results": tool_results,
                    "num_turns": turn_count,
                    "completed": True
                }
            
            # 执行工具调用（限制每轮工具调用数量）
            tool_calls = tool_calls[:self.max_tool_calls_per_turn]
            executed_results = []
            
            for tool_call in tool_calls:
                tool_calls_made.append(tool_call)
                
                # 执行工具
                result = self.tool_executor.execute(
                    tool_call["name"],
                    tool_call["arguments"]
                )
                
                executed_results.append(result)
                tool_results.append({
                    "tool_call": tool_call,
                    "result": result
                })
            
            # 格式化工具结果
            tool_results_text = self.tool_parser.format_tool_results(
                tool_calls, executed_results
            )
            
            # 构建下一轮的提示（包含工具结果）
            # 保留之前的对话上下文
            current_prompt += f" {response}\n\n{tool_results_text}\n\n<|assistant|>"
        
        # 达到最大轮数，返回当前结果
        return {
            "conversation_history": conversation_history,
            "final_response": conversation_history[-1]["cleaned_response"] if conversation_history else "",
            "tool_calls_made": tool_calls_made,
            "tool_results": tool_results,
            "num_turns": turn_count,
            "completed": turn_count < self.max_turns
        }
    
    def build_prompt_from_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """
        从对话历史构建提示文本
        
        Args:
            conversation: 对话历史，格式为 [{"role": "user|assistant", "content": "..."}, ...]
            
        Returns:
            格式化的提示文本
        """
        system_prompt = get_tools_system_prompt()
        parts = [system_prompt]
        
        for turn in conversation:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")
            
            if role == "user":
                parts.append(f"<|user|> {content}")
            elif role == "assistant":
                parts.append(f"<|assistant|> {content}")
            elif role == "tool_result":
                # 工具结果
                parts.append(f"<tool_result>\n{content}\n</tool_result>")
        
        # 如果最后一条不是用户消息，添加assistant标记
        if not conversation or conversation[-1].get("role", "").lower() != "user":
            parts.append("<|assistant|>")
        
        return "\n".join(parts)
    
    def simulate_conversation_for_training(self,
                                         conversation: List[Dict[str, str]],
                                         model_generate_fn,
                                         tokenizer) -> Dict[str, Any]:
        """
        为训练目的模拟对话（从给定的对话历史开始）
        
        Args:
            conversation: 初始对话历史
            model_generate_fn: 模型生成函数
            tokenizer: 分词器
            
        Returns:
            对话结果字典
        """
        # 构建初始提示
        initial_prompt = self.build_prompt_from_conversation(conversation)
        
        # 处理对话
        return self.process_conversation(
            initial_prompt="",  # 已经在build_prompt_from_conversation中包含了
            model_generate_fn=lambda prompt, tok: model_generate_fn(
                initial_prompt if not prompt else prompt, tok
            ),
            tokenizer=tokenizer
        )

