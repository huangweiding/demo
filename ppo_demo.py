"""
PPO (Proximal Policy Optimization) 详细实现演示

本文件详细展示了PPO算法的核心实现，基于TRL库中的PPO训练器。
包括：
1. 算法原理和数学公式
2. 核心组件实现
3. 训练流程详解
4. 关键参数说明

PPO是一种重要的强化学习算法，广泛应用于语言模型的强化学习微调。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math
import time

# ============================================================================
# 1. PPO算法核心数学原理
# ============================================================================

"""
PPO算法的核心思想：
1. 使用当前策略生成轨迹
2. 计算优势函数 (Advantage Function)
3. 使用裁剪的surrogate objective进行优化
4. 包含价值函数学习和KL散度约束

数学公式：
优势函数: A(s,a) = Q(s,a) - V(s)
策略损失: L_policy = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
价值损失: L_value = (V(s) - R)^2
总损失: L = L_policy + c1 * L_value + c2 * KL(π_old || π_new)
其中 r_t = π_new(a|s) / π_old(a|s)
"""

# ============================================================================
# 2. 核心配置类
# ============================================================================

@dataclass
class PPOConfig:
    """PPO训练的核心配置参数"""
    
    # 训练参数
    learning_rate: float = 1e-5  # 学习率
    num_ppo_epochs: int = 4  # PPO训练轮数
    num_mini_batches: int = 4  # 小批次数量
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    num_total_batches: int = 100  # 总训练批次
    
    # 生成参数
    response_length: int = 128  # 响应长度
    temperature: float = 1.0  # 采样温度
    batch_size: int = 8  # 批次大小
    local_batch_size: int = 8  # 本地批次大小
    local_rollout_forward_batch_size: int = 4  # 前向传播批次大小
    
    # PPO核心参数
    cliprange: float = 0.2  # 策略裁剪范围
    vf_coef: float = 0.1  # 价值函数系数
    cliprange_value: float = 0.2  # 价值函数裁剪范围
    gamma: float = 1.0  # 折扣因子
    lam: float = 0.95  # GAE参数
    
    # KL散度参数
    kl_coef: float = 0.05  # KL散度系数
    kl_estimator: str = "k1"  # KL散度估计器类型
    
    # 奖励处理
    whiten_rewards: bool = False  # 是否白化奖励
    
    # 日志参数
    logging_steps: int = 10  # 日志记录步数
    eval_steps: int = 100  # 评估步数
    save_steps: int = 500  # 保存步数
    
    # 生成采样参数
    num_sample_generations: int = 0  # 采样生成数量
    sample_generations_freq: int = 10  # 采样生成频率

# ============================================================================
# 3. 核心工具函数
# ============================================================================

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算掩码张量的均值"""
    return (tensor * mask).sum() / mask.sum().clamp(min=1.0)

def masked_whiten(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对掩码张量进行白化处理"""
    mean = masked_mean(tensor, mask)
    var = masked_mean((tensor - mean) ** 2, mask)
    return (tensor - mean) / (var + 1e-8).sqrt()

def selective_log_softmax(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    计算选择性log_softmax
    
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size] 或 [batch_size, vocab_size]
        tokens: 目标token ID [batch_size, seq_len] 或 [batch_size]
    
    Returns:
        log_probs: 对应的对数概率 [batch_size, seq_len] 或 [batch_size]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 确保tokens的维度正确
    if tokens.dim() == 1:
        # tokens是 [batch_size]
        return torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
    else:
        # tokens是 [batch_size, seq_len]
        return torch.gather(log_probs, -1, tokens.unsqueeze(-1)).squeeze(-1)

def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算GAE (Generalized Advantage Estimation) 优势函数
    
    Args:
        rewards: 奖励序列 [batch_size, seq_len]
        values: 价值函数输出 [batch_size, seq_len]
        gamma: 折扣因子
        lam: GAE参数
    
    Returns:
        advantages: 优势函数 [batch_size, seq_len]
        returns: 回报 [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # 从后往前计算GAE
    last_gae_lam = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = last_gae_lam = delta + gamma * lam * last_gae_lam
    
    returns = advantages + values
    return advantages, returns

def compute_kl_divergence(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    estimator: str = "k1"
) -> torch.Tensor:
    """
    计算KL散度
    
    Args:
        old_log_probs: 旧策略的对数概率
        new_log_probs: 新策略的对数概率
        estimator: 估计器类型 ("k1" 或 "k3")
    
    Returns:
        kl_div: KL散度
    """
    if estimator == "k1":
        # K1估计器: 直接计算KL散度
        kl_div = (torch.exp(new_log_probs - old_log_probs) - 
                  (new_log_probs - old_log_probs) - 1)
    elif estimator == "k3":
        # K3估计器: 使用更稳定的估计
        ratio = torch.exp(new_log_probs - old_log_probs)
        kl_div = (ratio - 1) - (new_log_probs - old_log_probs)
    else:
        raise ValueError(f"Unknown KL estimator: {estimator}")
    
    return kl_div

# ============================================================================
# 4. 策略和价值函数模型
# ============================================================================

class PolicyModel(nn.Module):
    """策略模型 (Actor)"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 创建注意力掩码
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Transformer编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        logits = self.lm_head(x)
        return logits

class ValueModel(nn.Module):
    """价值函数模型 (Critic)"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1
        )
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            values: 价值函数输出 [batch_size, seq_len]
        """
        x = self.embedding(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Transformer编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        values = self.value_head(x).squeeze(-1)
        return values

class RewardModel(nn.Module):
    """奖励模型"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1
        )
        self.reward_head = nn.Linear(hidden_size, 1)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        
        Returns:
            rewards: 奖励输出 [batch_size, seq_len]
        """
        x = self.embedding(input_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Transformer编码
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        x = self.transformer(x, src_key_padding_mask=~attention_mask.bool())
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        rewards = self.reward_head(x).squeeze(-1)
        return rewards

# ============================================================================
# 5. 生成和采样函数
# ============================================================================

def generate_responses(
    policy_model: PolicyModel,
    queries: torch.Tensor,
    max_length: int,
    temperature: float = 1.0,
    batch_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用策略模型生成响应
    
    Args:
        policy_model: 策略模型
        queries: 查询序列 [batch_size, query_len]
        max_length: 最大生成长度
        temperature: 采样温度
        batch_size: 批次大小
    
    Returns:
        responses: 生成的响应 [batch_size, response_len]
        log_probs: 对应的对数概率 [batch_size, response_len]
    """
    device = queries.device
    batch_size, query_len = queries.shape
    
    # 初始化响应
    responses = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
    log_probs = torch.zeros(batch_size, max_length, device=device)
    
    # 逐token生成
    for t in range(max_length):
        # 构建输入序列
        if t == 0:
            input_ids = queries
        else:
            input_ids = torch.cat([queries, responses[:, :t]], dim=1)
        
        # 获取logits
        with torch.no_grad():
            logits = policy_model(input_ids)
            next_logits = logits[:, -1, :]  # 取最后一个token的logits
        
        # 应用温度
        next_logits = next_logits / temperature
        
        # 采样下一个token
        probs = F.softmax(next_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 记录响应和概率
        responses[:, t] = next_tokens
        
        # 计算对数概率
        log_probs_batch = F.log_softmax(next_logits, dim=-1)
        log_probs[:, t] = torch.gather(log_probs_batch, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
    
    return responses, log_probs

# ============================================================================
# 6. PPO核心损失计算
# ============================================================================

def compute_ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
    padding_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    计算PPO损失
    
    Args:
        old_log_probs: 旧策略的对数概率 [batch_size, seq_len]
        new_log_probs: 新策略的对数概率 [batch_size, seq_len]
        advantages: 优势函数 [batch_size, seq_len]
        values: 价值函数输出 [batch_size, seq_len]
        returns: 回报 [batch_size, seq_len]
        config: PPO配置
        padding_mask: 填充掩码 [batch_size, seq_len]
    
    Returns:
        policy_loss: 策略损失
        value_loss: 价值损失
        total_loss: 总损失
        metrics: 指标字典
    """
    # 计算概率比率
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 计算策略损失 (PPO裁剪目标)
    policy_losses1 = -advantages * ratio
    policy_losses2 = -advantages * torch.clamp(
        ratio, 1 - config.cliprange, 1 + config.cliprange
    )
    policy_loss = torch.max(policy_losses1, policy_losses2)
    
    # 应用掩码
    if padding_mask is not None:
        policy_loss = masked_mean(policy_loss, ~padding_mask)
    else:
        policy_loss = policy_loss.mean()
    
    # 计算价值损失
    value_losses1 = torch.square(values - returns)
    value_losses2 = torch.square(
        torch.clamp(
            values,
            returns - config.cliprange_value,
            returns + config.cliprange_value
        ) - returns
    )
    value_loss = torch.max(value_losses1, value_losses2)
    
    # 应用掩码
    if padding_mask is not None:
        value_loss = 0.5 * masked_mean(value_loss, ~padding_mask)
    else:
        value_loss = 0.5 * value_loss.mean()
    
    # 计算KL散度
    kl_div = compute_kl_divergence(old_log_probs, new_log_probs, config.kl_estimator)
    if padding_mask is not None:
        kl_loss = masked_mean(kl_div, ~padding_mask)
    else:
        kl_loss = kl_div.mean()
    
    # 总损失
    total_loss = policy_loss + config.vf_coef * value_loss + config.kl_coef * kl_loss
    
    # 计算指标
    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_var": ratio.var().item(),
        "advantages_mean": advantages.mean().item(),
        "advantages_std": advantages.std().item(),
    }
    
    return policy_loss, value_loss, total_loss, metrics

# ============================================================================
# 7. PPO训练器
# ============================================================================

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        policy_model: PolicyModel,
        value_model: ValueModel,
        reward_model: RewardModel,
        config: PPOConfig,
        device: str = "cpu"
    ):
        self.policy_model = policy_model.to(device)
        self.value_model = value_model.to(device)
        self.reward_model = reward_model.to(device)
        self.config = config
        self.device = device
        
        # 创建参考模型 (用于KL散度计算)
        self.ref_policy_model = PolicyModel(
            policy_model.embedding.num_embeddings,
            policy_model.embedding.embedding_dim
        ).to(device)
        self.ref_policy_model.load_state_dict(policy_model.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_model.parameters()),
            lr=config.learning_rate
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=config.num_total_batches
        )
        
        # 训练状态
        self.global_step = 0
        self.episode = 0
        
    def generate_batch(self, queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        生成一个批次的数据
        
        Args:
            queries: 查询序列 [batch_size, query_len]
        
        Returns:
            batch_data: 批次数据字典
        """
        # 生成响应
        responses, log_probs = generate_responses(
            self.policy_model,
            queries,
            self.config.response_length,
            self.config.temperature,
            self.config.local_rollout_forward_batch_size
        )
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            ref_log_probs = self.compute_ref_log_probs(queries, responses)
        
        # 计算价值函数
        with torch.no_grad():
            values = self.value_model(torch.cat([queries, responses], dim=1))
            values = values[:, queries.size(1):]  # 只取响应部分
        
        # 计算奖励
        with torch.no_grad():
            rewards = self.reward_model(torch.cat([queries, responses], dim=1))
            rewards = rewards[:, queries.size(1):]  # 只取响应部分
        
        # 计算优势函数
        advantages, returns = compute_gae_advantages(
            rewards, values, self.config.gamma, self.config.lam
        )
        
        # 白化优势函数
        if self.config.whiten_rewards:
            advantages = masked_whiten(advantages, torch.ones_like(advantages))
        
        return {
            "queries": queries,
            "responses": responses,
            "log_probs": log_probs,
            "ref_log_probs": ref_log_probs,
            "values": values,
            "rewards": rewards,
            "advantages": advantages,
            "returns": returns,
        }
    
    def compute_ref_log_probs(self, queries: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """计算参考模型的对数概率"""
        input_ids = torch.cat([queries, responses], dim=1)
        with torch.no_grad():
            logits = self.ref_policy_model(input_ids)
            logits = logits[:, queries.size(1):, :]  # 只取响应部分
            log_probs = selective_log_softmax(logits, responses)
        return log_probs
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            batch_data: 批次数据
        
        Returns:
            metrics: 训练指标
        """
        # 获取数据
        queries = batch_data["queries"]
        responses = batch_data["responses"]
        old_log_probs = batch_data["log_probs"]
        ref_log_probs = batch_data["ref_log_probs"]
        old_values = batch_data["values"]
        advantages = batch_data["advantages"]
        returns = batch_data["returns"]
        
        # 创建填充掩码
        padding_mask = (responses == 0)  # 假设0是pad_token
        
        # 多轮PPO训练
        policy_losses = []
        value_losses = []
        total_losses = []
        
        for epoch in range(self.config.num_ppo_epochs):
            # 重新计算当前策略的对数概率
            input_ids = torch.cat([queries, responses], dim=1)
            logits = self.policy_model(input_ids)
            logits = logits[:, queries.size(1):, :]  # 只取响应部分
            new_log_probs = selective_log_softmax(logits, responses)
            
            # 重新计算价值函数
            new_values = self.value_model(input_ids)
            new_values = new_values[:, queries.size(1):]  # 只取响应部分
            
            # 计算PPO损失
            policy_loss, value_loss, total_loss, metrics = compute_ppo_loss(
                old_log_probs,
                new_log_probs,
                advantages,
                new_values,
                returns,
                self.config,
                padding_mask
            )
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            total_losses.append(total_loss)
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均指标
        avg_metrics = {
            "policy_loss_avg": torch.stack(policy_losses).mean().item(),
            "value_loss_avg": torch.stack(value_losses).mean().item(),
            "total_loss_avg": torch.stack(total_losses).mean().item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
        
        return avg_metrics
    
    def train(self, train_queries: List[torch.Tensor]):
        """训练循环"""
        print("开始PPO训练...")
        start_time = time.time()
        
        for update in range(1, self.config.num_total_batches + 1):
            # 生成批次数据 - 确保批次大小正确
            batch_size = min(self.config.batch_size, len(train_queries))
            batch_queries = torch.stack(train_queries[:batch_size])
            batch_data = self.generate_batch(batch_queries)
            
            # 训练步骤
            metrics = self.train_step(batch_data)
            
            # 更新状态
            self.global_step += 1
            self.episode += batch_size
            
            # 记录日志
            if update % self.config.logging_steps == 0:
                elapsed_time = time.time() - start_time
                eps = int(self.episode / elapsed_time) if elapsed_time > 0 else 0
                
                print(f"Step {update}/{self.config.num_total_batches}")
                print(f"Episode: {self.episode}, EPS: {eps}")
                print(f"Policy Loss: {metrics['policy_loss_avg']:.4f}")
                print(f"Value Loss: {metrics['value_loss_avg']:.4f}")
                print(f"Total Loss: {metrics['total_loss_avg']:.4f}")
                print(f"Learning Rate: {metrics['learning_rate']:.6f}")
                print("-" * 50)
            
            # 保存模型
            if update % self.config.save_steps == 0:
                self.save_model(f"ppo_model_step_{update}.pt")
        
        print("PPO训练完成!")
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            "policy_model": self.policy_model.state_dict(),
            "value_model": self.value_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
        }, path)
        print(f"模型已保存到: {path}")

# ============================================================================
# 8. 示例使用
# ============================================================================

def demo_ppo_usage():
    """PPO使用演示"""
    
    # 创建配置
    config = PPOConfig(
        learning_rate=1e-5,
        num_ppo_epochs=4,
        response_length=64,
        batch_size=4,
        cliprange=0.2,
        vf_coef=0.1,
        kl_coef=0.05,
        gamma=1.0,
        lam=0.95
    )
    
    # 创建模型
    vocab_size = 1000
    hidden_size = 256
    
    policy_model = PolicyModel(vocab_size, hidden_size)
    value_model = ValueModel(vocab_size, hidden_size)
    reward_model = RewardModel(vocab_size, hidden_size)
    
    # 创建训练器
    trainer = PPOTrainer(policy_model, value_model, reward_model, config, device="cpu")
    
    # 示例查询 - 确保所有查询具有相同长度
    query_length = 10
    queries = [
        torch.randint(0, vocab_size, (query_length,)),  # 查询1
        torch.randint(0, vocab_size, (query_length,)),  # 查询2
        torch.randint(0, vocab_size, (query_length,)),  # 查询3
        torch.randint(0, vocab_size, (query_length,)),  # 查询4
    ]
    
    print("=== PPO算法演示 ===")
    print(f"配置参数:")
    print(f"  - learning_rate (学习率): {config.learning_rate}")
    print(f"  - num_ppo_epochs (PPO轮数): {config.num_ppo_epochs}")
    print(f"  - cliprange (裁剪范围): {config.cliprange}")
    print(f"  - vf_coef (价值函数系数): {config.vf_coef}")
    print(f"  - kl_coef (KL散度系数): {config.kl_coef}")
    print(f"  - gamma (折扣因子): {config.gamma}")
    print(f"  - lam (GAE参数): {config.lam}")
    
    print(f"\n模型架构:")
    print(f"  - Policy Model: {policy_model}")
    print(f"  - Value Model: {value_model}")
    print(f"  - Reward Model: {reward_model}")
    
    print(f"\n开始训练...")
    
    # 开始训练
    trainer.train(queries)

if __name__ == "__main__":
    demo_ppo_usage()
