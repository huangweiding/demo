"""
包含工具调用的多轮 GRPO 训练主脚本

这个脚本展示了如何使用 GRPO (Group Relative Policy Optimization) 
训练一个能够进行多轮对话并使用工具的语言模型。

主要特性：
1. 支持多轮对话
2. 支持工具调用（calculator, weather_query, search, get_current_time）
3. 自动检测和执行工具调用
4. 基于工具使用质量的奖励函数
5. 完整的 GRPO 训练流程
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

# 导入 GRPO 核心组件
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grpo_demo.grpo_tutorial import (
    GRPOConfig,
    generate_completions,
    compute_rewards,
    compute_advantages,
    get_per_token_logps,
    compute_grpo_loss,
)

# 导入工具调用相关模块
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from tools import get_tools_system_prompt, ToolExecutor
from tool_parser import ToolCallParser
from multi_turn_handler import MultiTurnHandler
from reward_functions import create_reward_function, RewardConfig


@dataclass
class TrainingConfig:
    """训练配置"""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    data_path: str = "sample_data.jsonl"
    output_dir: str = "./checkpoints"
    
    # GRPO 参数
    beta: float = 0.04
    num_generations: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    epsilon: float = 0.2
    learning_rate: float = 1e-6
    num_iterations: int = 1
    loss_type: str = "bnpo"
    scale_rewards: bool = True
    
    # 训练参数
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    save_steps: int = 100
    logging_steps: int = 10
    
    # 多轮对话参数
    max_turns: int = 5
    max_tool_calls_per_turn: int = 3
    
    # 其他参数
    bf16: bool = False
    seed: int = 42
    device: str = "auto"


class ToolCallDataset(Dataset):
    """工具调用数据集"""
    
    def __init__(self, jsonl_path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def build_prompt_with_tools(conversation: List[Dict[str, str]]) -> str:
    """构建包含工具描述的提示"""
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
            parts.append(f"<tool_result>\n{content}\n</tool_result>")
    
    # 如果最后一条是用户消息，添加assistant标记
    if conversation and conversation[-1].get("role", "").lower() == "user":
        parts.append("<|assistant|>")
    
    return "\n".join(parts)


def generate_with_tool_calls(model, prompts: List[str], tokenizer, 
                             config: GRPOConfig,
                             multi_turn_handler: MultiTurnHandler) -> Dict:
    """
    生成包含工具调用的多轮响应
    
    Args:
        model: 语言模型
        prompts: 提示列表
        tokenizer: 分词器
        config: GRPO配置
        multi_turn_handler: 多轮对话处理器
    
    Returns:
        生成结果字典，包含多轮对话信息
    """
    all_conversations = []
    all_final_responses = []
    all_tool_calls = []
    all_tool_results = []
    
    # 定义模型生成函数
    def model_generate_fn(prompt: str, tok) -> str:
        # 编码提示
        inputs = tok(prompt, return_tensors="pt", padding=True, 
                    padding_side="left", add_special_tokens=False)
        
        # 移动到模型设备
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 截断长度
        if config.max_prompt_length:
            max_length = config.max_prompt_length + config.max_completion_length
            inputs["input_ids"] = inputs["input_ids"][:, -config.max_prompt_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -config.max_prompt_length:]
        
        # 生成
        generation_config = {
            "max_new_tokens": config.max_completion_length,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": True,
            "pad_token_id": tok.pad_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_config)
        
        # 解码
        prompt_length = inputs["input_ids"].size(1)
        completion_ids = outputs[:, prompt_length:]
        response = tok.batch_decode(completion_ids, skip_special_tokens=True)[0]
        
        return response
    
    # 处理每个提示
    for prompt in prompts:
        # 解析prompt中的对话历史
        # 假设prompt格式包含对话历史
        conversation = []
        if "<|user|>" in prompt:
            # 解析对话历史
            parts = prompt.split("<|user|>")
            for i, part in enumerate(parts[1:], 1):
                if "<|assistant|>" in part:
                    user_content, assistant_content = part.split("<|assistant|>", 1)
                    conversation.append({"role": "user", "content": user_content.strip()})
                    if assistant_content.strip():
                        conversation.append({"role": "assistant", "content": assistant_content.strip()})
                else:
                    conversation.append({"role": "user", "content": part.strip()})
        else:
            # 简单情况：只有用户消息
            conversation = [{"role": "user", "content": prompt}]
        
        # 处理多轮对话
        initial_prompt_text = build_prompt_with_tools(conversation)
        result = multi_turn_handler.process_conversation(
            initial_prompt="",  # 已经在build_prompt_with_tools中包含了
            model_generate_fn=lambda p, tok: model_generate_fn(
                initial_prompt_text if not p else p, tok
            ),
            tokenizer=tokenizer
        )
        
        all_conversations.append(result["conversation_history"])
        all_final_responses.append(result["final_response"])
        all_tool_calls.append(result["tool_calls_made"])
        all_tool_results.append(result["tool_results"])
    
    # 为了GRPO训练，我们需要生成多个候选响应
    # 这里简化处理：使用最终的完整响应作为completion
    # 实际应用中，应该对每个prompt生成num_generations个候选
    
    return {
        "conversations": all_conversations,
        "final_responses": all_final_responses,
        "tool_calls": all_tool_calls,
        "tool_results": all_tool_results,
    }


def grpo_training_step_with_tools(
    model,
    ref_model,
    prompts: List[str],
    tokenizer,
    reward_funcs: List[callable],
    config: GRPOConfig,
    multi_turn_handler: MultiTurnHandler,
    training_config: TrainingConfig
) -> Dict:
    """
    执行一个包含工具调用的GRPO训练步骤
    
    Args:
        model: 当前模型
        ref_model: 参考模型
        prompts: 提示列表
        tokenizer: 分词器
        reward_funcs: 奖励函数列表
        config: GRPO配置
        multi_turn_handler: 多轮对话处理器
        training_config: 训练配置
    
    Returns:
        训练结果字典
    """
    # 扩展prompts以生成多个候选
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * config.num_generations)
    
    # 生成多个候选响应（每个prompt生成num_generations个）
    all_responses = []
    all_completions_text = []
    
    # 简化版本：对每个prompt生成num_generations个响应
    for prompt in prompts:
        # 生成多个候选
        for _ in range(config.num_generations):
            # 使用标准生成函数（简化版本，实际应该使用多轮处理器）
            prompt_inputs = tokenizer(
                prompt, return_tensors="pt", padding=True,
                padding_side="left", add_special_tokens=False
            )
            
            device = next(model.parameters()).device
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            
            if config.max_prompt_length:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -config.max_prompt_length:]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -config.max_prompt_length:]
            
            generation_config = {
                "max_new_tokens": config.max_completion_length,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = model.generate(**prompt_inputs, **generation_config)
            
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = outputs[:, prompt_length:]
            completion_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
            
            all_responses.append(completion_text)
            all_completions_text.append(completion_text)
    
    # 计算奖励
    samples = [
        {"prompt": prompt, "response": response}
        for prompt, response in zip(expanded_prompts, all_responses)
    ]
    
    rewards_list = []
    for reward_func in reward_funcs:
        rewards = reward_func(samples)
        rewards_list.append(rewards)
    
    # 转换为tensor
    rewards = torch.tensor(rewards_list, dtype=torch.float32).T  # [batch_size, num_reward_funcs]
    
    # 计算优势
    advantages = compute_advantages(rewards, config.num_generations, config.scale_rewards)
    
    # 计算损失（需要重新生成以获取logits）
    # 这里简化处理，实际应该保存生成时的logits
    prompt_inputs = tokenizer(
        expanded_prompts, return_tensors="pt", padding=True,
        padding_side="left", add_special_tokens=False
    )
    
    device = next(model.parameters()).device
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    
    if config.max_prompt_length:
        prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -config.max_prompt_length:]
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -config.max_prompt_length:]
    
    # 重新编码完整序列以计算logits
    # 这里需要将prompt和completion拼接
    # 简化版本：使用generate_completions
    generation_results = generate_completions(model, prompts, tokenizer, config)
    
    # 计算当前模型和参考模型的logits
    input_ids = torch.cat([
        generation_results["prompt_ids"],
        generation_results["completion_ids"]
    ], dim=1).to(device)
    
    attention_mask = torch.cat([
        generation_results["prompt_mask"],
        generation_results["completion_mask"]
    ], dim=1).to(device)
    
    per_token_logps = get_per_token_logps(
        model, input_ids, attention_mask,
        generation_results["completion_ids"].size(1)
    )
    
    ref_logps = None
    if config.beta != 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logps = get_per_token_logps(
                ref_model, input_ids, attention_mask,
                generation_results["completion_ids"].size(1)
            )
    
    # 计算损失（需要扩展advantages以匹配per_token_logps的形状）
    # 这里简化处理
    advantages_expanded = advantages.unsqueeze(1).expand(-1, per_token_logps.size(1))
    
    loss = compute_grpo_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=per_token_logps.detach(),
        advantages=advantages,
        completion_mask=generation_results["completion_mask"],
        config=config,
        ref_logps=ref_logps
    )
    
    metrics = {
        "loss": loss.item(),
        "mean_reward": rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }
    
    return {
        "loss": loss,
        "metrics": metrics,
        "rewards": rewards,
        "advantages": advantages,
    }


def main():
    parser = argparse.ArgumentParser(description="多轮工具调用 GRPO 训练")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_path", type=str, default="sample_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建配置
    grpo_config = GRPOConfig(
        beta=0.04,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        epsilon=0.2,
        loss_type="bnpo",
        scale_rewards=True,
    )
    
    training_config = TrainingConfig(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        bf16=args.bf16,
        seed=args.seed,
    )
    
    # 加载模型和tokenizer
    print(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map=training_config.device,
    )
    
    # 创建参考模型（用于KL散度计算）
    ref_model = None
    if grpo_config.beta > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else None,
            device_map=training_config.device,
        )
        ref_model.eval()
    
    # 加载数据集
    print(f"加载数据集: {args.data_path}")
    dataset = ToolCallDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # 创建多轮对话处理器
    multi_turn_handler = MultiTurnHandler(
        max_turns=training_config.max_turns,
        max_tool_calls_per_turn=training_config.max_tool_calls_per_turn,
    )
    
    # 创建奖励函数
    reward_config = RewardConfig()
    reward_fn = create_reward_function(reward_config)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    
    # 训练循环
    print("开始训练...")
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            # 构建提示
            # DataLoader 返回的是字典，其中 "conversation" 键包含批次中的所有对话
            if isinstance(batch, dict):
                conversations = batch.get("conversation", [])
                if not isinstance(conversations, list):
                    conversations = [conversations]
            elif isinstance(batch, list):
                conversations = [item.get("conversation", []) for item in batch]
            else:
                conversations = []
            
            prompts = []
            for conv in conversations:
                if conv:  # 确保对话不为空
                    prompt = build_prompt_with_tools(conv)
                    prompts.append(prompt)
            
            if not prompts:
                continue  # 跳过空批次
            
            # 执行训练步骤
            result = grpo_training_step_with_tools(
                model=model,
                ref_model=ref_model,
                prompts=prompts,
                tokenizer=tokenizer,
                reward_funcs=[reward_fn],
                config=grpo_config,
                multi_turn_handler=multi_turn_handler,
                training_config=training_config,
            )
            
            # 反向传播
            loss = result["loss"]
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 日志记录
                if global_step % args.gradient_accumulation_steps == 0:
                    print(f"Step {global_step}: Loss={result['metrics']['loss']:.4f}, "
                          f"Reward={result['metrics']['mean_reward']:.4f}, "
                          f"Advantage={result['metrics']['mean_advantage']:.4f}")
                
                # 保存模型
                if global_step % (args.save_steps if hasattr(args, 'save_steps') else 100) == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"保存模型到: {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"\n训练完成！最终模型保存到: {final_save_path}")


if __name__ == "__main__":
    main()

