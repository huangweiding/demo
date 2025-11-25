"""
奖励函数模块

评估包含工具调用的多轮对话的质量，包括：
1. 工具调用的正确性
2. 工具使用的必要性
3. 最终回答的质量
4. 多轮对话的连贯性
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tool_parser import ToolCallParser


@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 工具调用相关权重
    w_tool_call_correctness: float = 0.30  # 工具调用格式和参数正确性
    w_tool_necessity: float = 0.20  # 工具使用的必要性
    w_tool_result_usage: float = 0.20  # 工具结果的使用情况
    w_final_answer_quality: float = 0.20  # 最终回答质量
    w_conversation_coherence: float = 0.10  # 对话连贯性
    
    # 奖励/惩罚系数
    tool_call_format_bonus: float = 0.5  # 正确格式的工具调用奖励
    tool_call_error_penalty: float = -0.3  # 工具调用错误惩罚
    unnecessary_tool_penalty: float = -0.2  # 不必要的工具调用惩罚
    successful_tool_result_usage_bonus: float = 0.3  # 成功使用工具结果奖励


class ToolCallRewardFunction:
    """工具调用奖励函数"""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.tool_parser = ToolCallParser()
    
    def compute_reward(self,
                      conversation_history: List[Dict[str, Any]],
                      tool_calls_made: List[Dict[str, Any]],
                      tool_results: List[Dict[str, Any]],
                      final_response: str,
                      user_query: str) -> float:
        """
        计算整体奖励
        
        Args:
            conversation_history: 对话历史
            tool_calls_made: 所有工具调用
            tool_results: 所有工具执行结果
            final_response: 最终响应
            user_query: 用户原始查询
            
        Returns:
            奖励分数 (0-1之间)
        """
        scores = {}
        
        # 1. 工具调用正确性
        scores['tool_call_correctness'] = self._score_tool_call_correctness(
            tool_calls_made, tool_results
        )
        
        # 2. 工具使用必要性
        scores['tool_necessity'] = self._score_tool_necessity(
            user_query, tool_calls_made, final_response
        )
        
        # 3. 工具结果使用情况
        scores['tool_result_usage'] = self._score_tool_result_usage(
            conversation_history, tool_results, final_response
        )
        
        # 4. 最终回答质量
        scores['final_answer_quality'] = self._score_final_answer_quality(
            user_query, final_response, tool_results
        )
        
        # 5. 对话连贯性
        scores['conversation_coherence'] = self._score_conversation_coherence(
            conversation_history
        )
        
        # 加权求和
        total_reward = (
            self.config.w_tool_call_correctness * scores['tool_call_correctness'] +
            self.config.w_tool_necessity * scores['tool_necessity'] +
            self.config.w_tool_result_usage * scores['tool_result_usage'] +
            self.config.w_final_answer_quality * scores['final_answer_quality'] +
            self.config.w_conversation_coherence * scores['conversation_coherence']
        )
        
        # 确保奖励在合理范围内
        return max(0.0, min(1.0, total_reward))
    
    def _score_tool_call_correctness(self,
                                     tool_calls: List[Dict[str, Any]],
                                     tool_results: List[Dict[str, Any]]) -> float:
        """评估工具调用的正确性"""
        if not tool_calls:
            return 0.5  # 没有工具调用时给予中性分数
        
        correct_count = 0
        total_count = len(tool_calls)
        
        for i, tool_call in enumerate(tool_calls):
            is_correct = False
            
            # 检查工具调用是否有对应的结果
            if i < len(tool_results):
                result = tool_results[i]["result"]
                # 如果工具执行成功，说明调用格式和参数正确
                if result.get("success", False):
                    is_correct = True
                    correct_count += 1
                else:
                    # 检查错误类型
                    error = result.get("error", "")
                    # 如果是参数错误，给予部分分数
                    if "参数" in error or "缺少" in error:
                        correct_count += 0.3
                    elif "未知工具" in error:
                        correct_count += 0.1  # 格式正确但工具不存在
        
        return correct_count / total_count if total_count > 0 else 0.0
    
    def _score_tool_necessity(self,
                             user_query: str,
                             tool_calls: List[Dict[str, Any]],
                             final_response: str) -> float:
        """评估工具使用的必要性"""
        # 检查用户查询是否需要工具
        query_lower = user_query.lower()
        needs_tool = any(keyword in query_lower for keyword in [
            "计算", "算", "天气", "查询", "搜索", "查找", "时间", "现在几点",
            "calculate", "weather", "search", "time", "what time"
        ])
        
        has_tool_calls = len(tool_calls) > 0
        
        if needs_tool and has_tool_calls:
            return 1.0  # 需要工具且使用了工具
        elif not needs_tool and not has_tool_calls:
            return 1.0  # 不需要工具且没有使用工具
        elif needs_tool and not has_tool_calls:
            return 0.3  # 需要工具但没有使用
        else:  # not needs_tool and has_tool_calls
            # 检查工具调用是否真的不需要
            # 如果最终回答质量很好，可能工具调用是合理的
            if len(final_response) > 20:  # 有实质性的回答
                return 0.7  # 可能工具调用有帮助
            else:
                return 0.4  # 不必要的工具调用
    
    def _score_tool_result_usage(self,
                                conversation_history: List[Dict[str, Any]],
                                tool_results: List[Dict[str, Any]],
                                final_response: str) -> float:
        """评估工具结果的使用情况"""
        if not tool_results:
            return 0.5  # 没有工具结果时给予中性分数
        
        # 检查最终回答中是否引用了工具结果
        usage_score = 0.0
        total_tools = len(tool_results)
        
        for tool_result in tool_results:
            result = tool_result["result"]
            if result.get("success", False):
                result_text = str(result.get("result", ""))
                # 检查最终回答是否包含工具结果的关键信息
                if result_text and len(result_text) > 0:
                    # 提取关键信息（简单方法：取前几个词）
                    key_words = result_text.split()[:3]
                    if any(word in final_response.lower() for word in key_words if len(word) > 2):
                        usage_score += 1.0
                    elif len(final_response) > 50:  # 有较长的回答，可能使用了工具结果
                        usage_score += 0.7
                    else:
                        usage_score += 0.3
            else:
                # 工具执行失败，检查是否在回答中处理了错误
                error = result.get("error", "")
                if error and error in final_response.lower():
                    usage_score += 0.5  # 提到了错误，给予部分分数
        
        return usage_score / total_tools if total_tools > 0 else 0.0
    
    def _score_final_answer_quality(self,
                                    user_query: str,
                                    final_response: str,
                                    tool_results: List[Dict[str, Any]]) -> float:
        """评估最终回答的质量"""
        if not final_response or len(final_response.strip()) < 5:
            return 0.1  # 回答太短
        
        score = 0.5  # 基础分数
        
        # 检查回答长度（适中最好）
        response_length = len(final_response.strip())
        if 20 <= response_length <= 500:
            score += 0.2
        elif response_length < 20:
            score -= 0.2
        
        # 检查是否回答了用户的问题（简单的关键词匹配）
        query_words = set(re.findall(r'\w+', user_query.lower()))
        response_words = set(re.findall(r'\w+', final_response.lower()))
        overlap = len(query_words & response_words)
        if overlap > 0:
            score += min(0.3, overlap / max(len(query_words), 1) * 0.3)
        
        # 如果有工具结果，检查回答是否与工具结果相关
        if tool_results:
            for tool_result in tool_results:
                result = tool_result["result"]
                if result.get("success", False):
                    result_text = str(result.get("result", ""))
                    # 检查回答中是否包含结果的关键信息
                    if result_text and any(word in final_response.lower() 
                                          for word in result_text.split()[:5] 
                                          if len(word) > 3):
                        score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _score_conversation_coherence(self,
                                     conversation_history: List[Dict[str, Any]]) -> float:
        """评估对话的连贯性"""
        if len(conversation_history) < 2:
            return 0.8  # 单轮对话，给予较高分数
        
        # 检查对话是否有逻辑连续性
        coherence_score = 0.5
        
        # 每轮对话都有响应（不是空响应）
        valid_responses = sum(1 for turn in conversation_history 
                            if turn.get("cleaned_response", "").strip())
        coherence_score += (valid_responses / len(conversation_history)) * 0.3
        
        # 检查工具调用和后续响应的连贯性
        for i, turn in enumerate(conversation_history):
            if turn.get("tool_calls") and i < len(conversation_history) - 1:
                # 有工具调用，检查下一轮是否使用了结果
                next_turn = conversation_history[i + 1]
                if next_turn.get("cleaned_response", "").strip():
                    coherence_score += 0.1
        
        return min(1.0, coherence_score)


def create_reward_function(config: RewardConfig = None):
    """
    创建奖励函数（用于GRPO训练）
    
    Args:
        config: 奖励配置
        
    Returns:
        奖励函数，接受 (prompt, response) 参数
    """
    reward_fn = ToolCallRewardFunction(config)
    
    def reward_function(samples: List[Dict[str, str]]) -> List[float]:
        """
        GRPO训练所需的奖励函数格式
        
        Args:
            samples: 样本列表，每个样本包含 'prompt' 和 'response'
                   对于多轮工具调用，response应该包含完整的对话历史
            
        Returns:
            奖励分数列表
        """
        rewards = []
        
        for sample in samples:
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            
            # 解析response中的对话历史和工具调用信息
            # 这里假设response是一个JSON字符串，包含完整的对话信息
            # 在实际应用中，可能需要根据实际的数据格式调整
            
            # 简化版本：从response中提取信息
            # 如果response是字典格式（包含conversation_history等）
            if isinstance(response, dict):
                conversation_history = response.get("conversation_history", [])
                tool_calls_made = response.get("tool_calls_made", [])
                tool_results = response.get("tool_results", [])
                final_response = response.get("final_response", "")
            else:
                # 如果response是字符串，尝试提取工具调用信息
                tool_parser = ToolCallParser()
                tool_calls = tool_parser.parse_tool_calls(str(response))
                conversation_history = [{"response": response}]
                tool_calls_made = tool_calls
                tool_results = []
                final_response = response
            
            # 提取用户查询（从prompt中）
            user_query = prompt.split("<|user|>")[-1].split("<|assistant|>")[0].strip() if "<|user|>" in prompt else prompt
            
            reward = reward_fn.compute_reward(
                conversation_history=conversation_history,
                tool_calls_made=tool_calls_made,
                tool_results=tool_results,
                final_response=final_response,
                user_query=user_query
            )
            
            rewards.append(reward)
        
        return rewards
    
    return reward_function

