from typing import List, Optional
import torch
from dataclasses import dataclass

from grpo_demo.grpo_tutorial import compute_grpo_loss
# grpo test

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

def generate_completion(model, prompts, tokenizer, config):
    prompt_inputs = tokenizer(prompts, return_types="pt", padding=True, padding_side="left", add_special_tokens=False)

    if config.max_prompt_length is not None:
        prompt_inputs = prompt_inputs["input_ids"][:, :config.max_prompt_length]
        attention_mask = prompt_inputs["attention_mask"][:, :config.max_prompt_length]
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

    prompt_length = prompt_inputs["input_ids"].size(1)
    prompt_ids = outputs[:, :prompt_length]
    completion_ids = outputs[:, prompt_length:]

    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]

    sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    completions_text = tokenizer.batch_code(completion_ids, skip_special_tokens=True)

    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_inputs["attention_mask"],
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "completions_text": completions_text,
        "is_eos": is_eos,
        "eos_idx": eos_idx
    }


def compute_rewards(prompt, completions, reward_funcs):
    batch_size = len(prompt)
    rewards = torch.zeros((batch_size, len(reward_funcs)))

    for i, reward_func in enumerate(reward_funcs):
        reward_values = reward_func(prompt, completions)

        reward_values = [r if r is not None else torch.nan for r in reward_values]

        rewards[:, i] = torch.tensor(reward_values, dtype=torch.float32)

    return rewards


def compute_advantages(rewards, num_generations, scale_rewards: bool=True):
    num_prompts = rewards.size(0) //num_generations

    rewards_reshaped = rewards.view(num_prompts, num_generations, -1)

    mean_rewards = rewards_reshaped.mean(dim=1)
    std_rewards = rewards_reshaped.std(dim=1)


    advantages = rewards - torch.repeat_interleave(mean_rewards, num_generations, -1)

    if scale_rewards:
        std_rewards_reshaped = torch.repeat_interleave(std_rewards, num_generations, -1)
        advantages = advantages/ (std_rewards+1e-6)

    if advantages.size(-1) > 1:
        advantages = advantages.mean(-1)
    else:
        advantages = advantages.squeeze(-1)

    return advantages

def score_completions(prompt, completions, reward_func, config):
    rewards = compute_rewards(prompt, completions, reward_func)

    advantages = compute_advantages(rewards, config.num_generations, config.scale_rewards)

    mean_rewards = rewards.mean(dim=0)
    std_rewards = rewards.std(dim=0)


    return {"rewards": rewards, "advantages": advantages, "mean_rewards": mean_rewards, "std_rewards": std_rewards}


def compute_logp_per_token(model, input_ids, attention_mask, logits_to_keep):

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits

    logits = logits[:, -logits_to_keep:]
    input_ids = input_ids[:, -logits_to_keep:]


    log_prob = torch.nn.functional.log_softmax(logits)


    # log_prob  [batch_size, seq_length, logp]
    # input_ids [batch_size, seq_length, 1]

    per_token_logps = torch.gather(log_prob, -1, input_ids.unsqueeze(-1)).squeeze(-1)


    return per_token_logps

def compute_grpo_loss(per_token_logps, old_per_token_logps, advantages, completion_mask, ref_probps):

    coef_1 = torch.exp(per_token_logps - old_per_token_logps)

    coef_2 = torch.clamp(coef_1, min=1+)


def grpo_trainer(model, ref_model, prompts: List[str], tokenizer, reward_func: List[callable], config: GRPOConfig):

    completion_result = generate_completion(model, prompts, tokenizer, reward_func)

    rewards = score_completions(prompts, completion_result["completion_result"], reward_func, config)

    input_ids = torch.cat([completion_result["prompt_ids"], completion_result["completion_ids"]], dim=1)
    attention_mask = torch.cat([completion_result["prompt_mask"], completion_result["completion_mask"]], dim=1)


    per_token_logps = compute_logp_per_token(model, input_ids, attention_mask, completion_result["completion_ids"].size(1))

    if ref_model is not None:
        ref_per_token_logps = compute_logp_per_token(ref_model, input_ids, attention_mask, completion_result["completion_ids"].size(1))

    loss = compute_grpo_loss()

if __name__ == "__main__":
    config = GRPOConfig(
        beta=0.04,
        num_generations=4,
        max_completion_length=64,
        temperature=0.9,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards=True
    )




