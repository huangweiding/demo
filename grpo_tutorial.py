"""
GRPO (Group Relative Policy Optimization) 详细实现演示

本文件详细展示了GRPO算法的核心实现，包括：
1. 算法原理和数学公式
2. 核心组件实现
3. 训练流程详解
4. 关键参数说明

GRPO是一种基于PPO的强化学习算法，专门用于语言模型的微调。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np

# ============================================================================
# 1. GRPO算法核心数学原理
# ============================================================================

"""
GRPO算法的核心思想：
1. 对每个prompt生成多个completion (G个)
2. 计算每个completion的reward
3. 在组内计算相对优势 (Group Relative Advantage)
4. 使用裁剪的surrogate objective进行优化

数学公式：
优势函数: A_{i,t} = (r_i - mean(r)) / std(r)
损失函数: L = -min(coef_1 * A, clip(coef_2, 1-ε, 1+ε) * A) + β * KL(π_θ || π_ref)
其中 coef_1 = π_θ(a|s) / π_old(a|s), coef_2 = π_θ(a|s) / π_old(a|s)
"""

# ============================================================================
# 2. 核心配置类
# ============================================================================

@dataclass
class GRPOConfig:
    """GRPO训练的核心配置参数"""
    
    # 模型和参考模型参数
    beta: float = 0.04  # KL散度系数，控制与参考模型的偏离程度
    disable_dropout: bool = False  # 是否禁用dropout，提高训练稳定性
    
    # 数据预处理参数
    max_prompt_length: int = 512  # prompt最大长度
    num_generations: int = 8  # 每个prompt生成的completion数量
    max_completion_length: int = 256  # completion最大长度
    shuffle_dataset: bool = True  # 是否打乱数据集
    
    # 生成参数
    temperature: float = 0.9  # 采样温度，控制生成的随机性
    top_p: float = 1.0  # 核采样参数
    top_k: int = 50  # top-k采样参数
    min_p: Optional[float] = None  # 最小概率阈值
    repetition_penalty: float = 1.0  # 重复惩罚系数
    
    # 训练参数
    learning_rate: float = 1e-6  # 学习率
    num_iterations: int = 1  # 每批次的迭代次数 (μ参数)
    epsilon: float = 0.2  # 裁剪参数的下界
    epsilon_high: Optional[float] = None  # 裁剪参数的上界
    delta: Optional[float] = None  # 双面GRPO的上界裁剪参数
    
    # 奖励函数参数
    scale_rewards: bool = True  # 是否对奖励进行标准化
    reward_weights: Optional[List[float]] = None  # 奖励函数权重
    
    # 损失函数类型
    loss_type: str = "bnpo"  # 损失函数类型: "grpo", "bnpo", "dr_grpo"
    mask_truncated_completions: bool = False  # 是否屏蔽截断的completion
    
    # 参考模型同步参数
    sync_ref_model: bool = False  # 是否同步参考模型
    ref_model_mixup_alpha: float = 0.6  # 参考模型混合参数
    ref_model_sync_steps: int = 512  # 参考模型同步步数

# ============================================================================
# 3. 核心工具函数
# ============================================================================

def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """计算张量的标准差，忽略NaN值"""
    # 移除NaN值
    tensor_clean = tensor[~torch.isnan(tensor)]
    if len(tensor_clean) == 0:
        return torch.tensor(torch.nan, device=tensor.device)
    return torch.std(tensor_clean)

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """计算张量的最小值，忽略NaN值"""
    tensor_clean = tensor[~torch.isnan(tensor)]
    if len(tensor_clean) == 0:
        return torch.tensor(torch.nan, device=tensor.device)
    return torch.min(tensor_clean)

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """计算张量的最大值，忽略NaN值"""
    tensor_clean = tensor[~torch.isnan(tensor)]
    if len(tensor_clean) == 0:
        return torch.tensor(torch.nan, device=tensor.device)
    return torch.max(tensor_clean)

def get_per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                       logits_to_keep: int) -> torch.Tensor:
    """
    计算每个token的对数概率
    
    Args:
        model: 语言模型
        input_ids: 输入token ID
        attention_mask: 注意力掩码
        logits_to_keep: 需要计算logits的token数量
    
    Returns:
        per_token_logps: 每个token的对数概率 [batch_size, seq_len]
    """
    # 获取模型输出
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # 只取completion部分的logits
    logits = logits[:, -logits_to_keep:]
    input_ids = input_ids[:, -logits_to_keep:]
    
    # 计算log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 获取对应token的概率
    # log_probs: [batch_size, seq_len, vocab_size]
    # input_ids unsqueeze(-1): [batch_size, seq_len, 1]
    per_token_logps = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
    
    return per_token_logps

# ============================================================================
# 4. 奖励函数计算
# ============================================================================

def compute_rewards(prompts: List[str], completions: List[str], 
                   reward_funcs: List[callable]) -> torch.Tensor:
    """
    计算奖励值
    
    Args:
        prompts: 提示列表
        completions: 完成列表
        reward_funcs: 奖励函数列表
    
    Returns:
        rewards: 奖励张量 [batch_size, num_reward_funcs]
    """
    batch_size = len(prompts)
    num_reward_funcs = len(reward_funcs)
    rewards = torch.zeros(batch_size, num_reward_funcs)
    
    for i, reward_func in enumerate(reward_funcs):
        # 调用奖励函数
        reward_values = reward_func(prompts=prompts, completions=completions)
        
        # 处理None值
        reward_values = [r if r is not None else torch.nan for r in reward_values]
        rewards[:, i] = torch.tensor(reward_values, dtype=torch.float32)
    
    return rewards

def compute_advantages(rewards: torch.Tensor, num_generations: int, 
                     scale_rewards: bool = True) -> torch.Tensor:
    """
    计算相对优势 (Group Relative Advantage)
    
    Args:
        rewards: 奖励张量 [batch_size, num_reward_funcs]
        num_generations: 每个prompt的生成数量
        scale_rewards: 是否标准化奖励
    
    Returns:
        advantages: 优势张量 [batch_size]
    """
    # 将奖励重塑为 [num_prompts, num_generations, num_reward_funcs]
    num_prompts = rewards.size(0) // num_generations
    rewards_reshaped = rewards.view(num_prompts, num_generations, -1)
    
    # 计算每个prompt组内的均值和标准差
    mean_rewards = rewards_reshaped.mean(dim=1)  # [num_prompts, num_reward_funcs]
    std_rewards = rewards_reshaped.std(dim=1)    # [num_prompts, num_reward_funcs]
    
    # 计算优势: (reward - mean) / std
    advantages = rewards - mean_rewards.repeat_interleave(num_generations, dim=0)
    
    if scale_rewards:
        # 标准化优势
        std_rewards_expanded = std_rewards.repeat_interleave(num_generations, dim=0)
        advantages = advantages / (std_rewards_expanded + 1e-4)
    
    # 如果有多个奖励函数，取平均
    if advantages.size(-1) > 1:
        advantages = advantages.mean(dim=-1)
    else:
        advantages = advantages.squeeze(-1)
    
    return advantages

# ============================================================================
# 5. 核心损失计算
# ============================================================================

def compute_kl_divergence(ref_logps: torch.Tensor, current_logps: torch.Tensor) -> torch.Tensor:
    """
    计算KL散度: KL(π_ref || π_current)
    
    Args:
        ref_logps: 参考模型的对数概率
        current_logps: 当前模型的对数概率
    
    Returns:
        kl_div: KL散度
    """
    # KL散度近似: exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1
    kl_div = torch.exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1
    return kl_div

def compute_grpo_loss(per_token_logps: torch.Tensor, 
                     old_per_token_logps: torch.Tensor,
                     advantages: torch.Tensor,
                     completion_mask: torch.Tensor,
                     config: GRPOConfig,
                     ref_logps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    计算GRPO损失
    
    Args:
        per_token_logps: 当前模型的对数概率 [batch_size, seq_len]
        old_per_token_logps: 旧模型的对数概率 [batch_size, seq_len]
        advantages: 优势值 [batch_size]
        completion_mask: completion掩码 [batch_size, seq_len]
        config: GRPO配置
        ref_logps: 参考模型的对数概率 (可选)
    
    Returns:
        loss: GRPO损失
    """
    # 计算概率比率
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    
    # 设置裁剪参数
    epsilon_low = config.epsilon
    epsilon_high = config.epsilon_high if config.epsilon_high is not None else config.epsilon
    
    # 计算裁剪后的概率比率
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    
    # 计算第一项损失
    if config.delta is not None:
        # 双面GRPO: 使用delta作为上界
        per_token_loss1 = torch.clamp(coef_1, max=config.delta) * advantages.unsqueeze(1)
    else:
        # 标准GRPO: 只应用下界裁剪
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    
    # 计算第二项损失 (裁剪版本)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    
    # 取最小值
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    
    # 添加KL散度项
    if config.beta != 0.0 and ref_logps is not None:
        kl_div = compute_kl_divergence(ref_logps, per_token_logps)
        per_token_loss = per_token_loss + config.beta * kl_div
    
    # 根据损失类型进行归一化
    if config.loss_type == "grpo":
        # GRPO: 按序列长度归一化
        loss = ((per_token_loss * completion_mask).sum(-1) / 
                completion_mask.sum(-1).clamp(min=1.0)).mean()
    
    elif config.loss_type == "bnpo":
        # BNPO: 按批次中活跃token数量归一化
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    
    elif config.loss_type == "dr_grpo":
        # Dr. GRPO: 使用全局常数归一化
        loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * config.max_completion_length)
    
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")
    
    return loss

# ============================================================================
# 6. 生成和评分流程
# ============================================================================

def generate_completions(model, prompts: List[str], tokenizer, config: GRPOConfig) -> Dict:
    """
    生成completion并计算相关指标
    
    Args:
        model: 语言模型
        prompts: 提示列表
        tokenizer: 分词器
        config: GRPO配置
    
    Returns:
        generation_results: 生成结果字典
    """
    # 对prompt进行编码
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, 
                             padding_side="left", add_special_tokens=False)
    
    # 截断prompt长度
    if config.max_prompt_length is not None:
        prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -config.max_prompt_length:]
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -config.max_prompt_length:]
    
    # 生成completion
    generation_config = {
        "max_new_tokens": config.max_completion_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "repetition_penalty": config.repetition_penalty,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        outputs = model.generate(**prompt_inputs, **generation_config)
    
    # 分离prompt和completion
    prompt_length = prompt_inputs["input_ids"].size(1)
    prompt_ids = outputs[:, :prompt_length]
    completion_ids = outputs[:, prompt_length:]
    
    # 处理EOS token
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    
    # 创建completion掩码
    sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    # 解码completion文本
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_inputs["attention_mask"],
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completions_text": completions_text,
        "is_eos": is_eos,
        "eos_idx": eos_idx
    }

def score_completions(prompts: List[str], completions: List[str], 
                     reward_funcs: List[callable], config: GRPOConfig) -> Dict:
    """
    对completion进行评分
    
    Args:
        prompts: 提示列表
        completions: 完成列表
        reward_funcs: 奖励函数列表
        config: GRPO配置
    
    Returns:
        scoring_results: 评分结果字典
    """
    # 计算奖励
    rewards = compute_rewards(prompts, completions, reward_funcs)
    
    # 计算优势
    advantages = compute_advantages(rewards, config.num_generations, config.scale_rewards)
    
    # 计算统计指标
    mean_rewards = rewards.mean(dim=0)
    std_rewards = rewards.std(dim=0)
    
    return {
        "rewards": rewards,
        "advantages": advantages,
        "mean_rewards": mean_rewards,
        "std_rewards": std_rewards
    }

# ============================================================================
# 7. 完整训练步骤
# ============================================================================

def grpo_training_step(model, ref_model, prompts: List[str], tokenizer, 
                      reward_funcs: List[callable], config: GRPOConfig) -> Dict:
    """
    执行一个GRPO训练步骤
    
    Args:
        model: 当前模型
        ref_model: 参考模型
        prompts: 提示列表
        tokenizer: 分词器
        reward_funcs: 奖励函数列表
        config: GRPO配置
    
    Returns:
        training_results: 训练结果字典
    """
    # 步骤1: 生成completion
    generation_results = generate_completions(model, prompts, tokenizer, config)
    
    # 步骤2: 评分completion
    scoring_results = score_completions(
        prompts, generation_results["completions_text"], reward_funcs, config
    )
    
    # 步骤3: 计算当前模型的对数概率
    input_ids = torch.cat([generation_results["prompt_ids"], 
                          generation_results["completion_ids"]], dim=1)
    attention_mask = torch.cat([generation_results["prompt_mask"], 
                              generation_results["completion_mask"]], dim=1)
    
    per_token_logps = get_per_token_logps(
        model, input_ids, attention_mask, generation_results["completion_ids"].size(1)
    )
    
    # 步骤4: 计算参考模型的对数概率 (如果启用)
    ref_logps = None
    if config.beta != 0.0 and ref_model is not None:
        with torch.no_grad():
            ref_logps = get_per_token_logps(
                ref_model, input_ids, attention_mask, generation_results["completion_ids"].size(1)
            )
    
    # 步骤5: 计算损失
    loss = compute_grpo_loss(
        per_token_logps=per_token_logps,
        old_per_token_logps=per_token_logps.detach(),  # 简化版本，实际应该保存旧值
        advantages=scoring_results["advantages"],
        completion_mask=generation_results["completion_mask"],
        config=config,
        ref_logps=ref_logps
    )
    
    # 步骤6: 计算统计指标
    metrics = {
        "loss": loss.item(),
        "mean_reward": scoring_results["mean_rewards"].mean().item(),
        "mean_advantage": scoring_results["advantages"].mean().item(),
        "completion_length": generation_results["completion_mask"].sum().item() / len(prompts),
        "eos_ratio": generation_results["is_eos"].any(dim=1).float().mean().item()
    }
    
    return {
        "loss": loss,
        "metrics": metrics,
        "generation_results": generation_results,
        "scoring_results": scoring_results
    }

# ============================================================================
# 8. 示例奖励函数
# ============================================================================

def length_reward(prompts: List[str], completions: List[str]) -> List[float]:
    """基于长度的奖励函数"""
    return [len(completion) for completion in completions]

def uniqueness_reward(prompts: List[str], completions: List[str]) -> List[float]:
    """基于独特字符数量的奖励函数"""
    return [len(set(completion)) for completion in completions]

def sentiment_reward(prompts: List[str], completions: List[str]) -> List[float]:
    """基于情感倾向的奖励函数 (简化版本)"""
    positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "horrible", "disgusting"]
    
    rewards = []
    for completion in completions:
        completion_lower = completion.lower()
        positive_count = sum(1 for word in positive_words if word in completion_lower)
        negative_count = sum(1 for word in negative_words if word in completion_lower)
        # 奖励正面情感，惩罚负面情感
        reward = positive_count - negative_count
        rewards.append(reward)
    
    return rewards

# ============================================================================
# 9. 使用示例
# ============================================================================

def demo_grpo_usage():
    """GRPO使用演示"""
    
    # 创建配置
    config = GRPOConfig(
        beta=0.04,
        num_generations=4,
        max_completion_length=64,
        temperature=0.9,
        epsilon=0.2,
        loss_type="bnpo",
        scale_rewards=True
    )
    
    # 示例数据
    prompts = [
        "The weather today is",
        "I love to",
        "The best movie I've seen is",
        "My favorite food is"
    ]
    
    # 奖励函数
    reward_funcs = [length_reward, uniqueness_reward, sentiment_reward]
    
    print("=== GRPO算法演示 ===")
    print(f"配置参数:")
    print(f"  - beta (KL系数): {config.beta}")
    print(f"  - num_generations (每prompt生成数): {config.num_generations}")
    print(f"  - epsilon (裁剪参数): {config.epsilon}")
    print(f"  - loss_type (损失类型): {config.loss_type}")
    print(f"  - scale_rewards (奖励标准化): {config.scale_rewards}")
    
    print(f"\n输入prompts:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
    
    print(f"\n奖励函数:")
    for i, func in enumerate(reward_funcs):
        print(f"  {i+1}. {func.__name__}")
    
    # 模拟训练步骤
    print(f"\n模拟训练步骤...")
    
    # 这里只是演示，实际需要真实的模型和tokenizer
    print("注意: 这是一个演示版本，实际使用需要:")
    print("1. 真实的语言模型 (如GPT-2, LLaMA等)")
    print("2. 对应的tokenizer")
    print("3. 适当的奖励函数")
    print("4. 完整的训练循环")

if __name__ == "__main__":
    demo_grpo_usage()
