"""multi_rollout/main.py
========================

本文件整理了目前在多步（multi-step）rollout过程中常见的几种损失函数，
并给出了每种损失的核心计算步骤。这里选取了三个在实际大模型对齐或
强化学习训练中最常用的策略梯度类损失：

1. REINFORCE / Monte-Carlo Policy Gradient
2. PPO (Proximal Policy Optimization) 裁剪损失
3. GRPO (Generative Rollout Policy Optimization)

为了保持代码的可读性，每个损失都包含：
    * 输入张量的含义
    * 核心数学公式
    * 对应的 PyTorch 实现步骤

此外在 `__main__` 部分提供一个随机生成的 rollout 示例，用于演示三种
损失的调用方式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# 数据结构：用于存放多步 rollout 的中间结果
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    """封装一批多步 rollout 数据。

    Attributes:
        log_probs_old: 旧策略在每个时间步的 log π_old(a_t | s_t)
        rewards: 每个时间步获得的即时奖励 r_t
        dones: episode 是否在该时间步结束（1 表示结束，0 表示未结束）
        values: 旧价值网络对每个时间步的估计 V_old(s_t)
        log_probs_new: 可选，当前策略的 log π_new(a_t | s_t)，
            在 REINFORCE 中可以复用旧 log prob，在 PPO/GRPO 中需要新策略
    """

    log_probs_old: torch.Tensor  # [batch, T]
    rewards: torch.Tensor  # [batch, T]
    dones: torch.Tensor  # [batch, T]
    values: torch.Tensor  # [batch, T]
    log_probs_new: Optional[torch.Tensor] = None  # [batch, T]


# ---------------------------------------------------------------------------
# 公共辅助函数：计算多步 return 与优势函数
# ---------------------------------------------------------------------------


def compute_returns_and_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用广义优势估计（GAE）计算多步 return 与 advantage。

    公式：
        δ_t = r_t + γ (1 - done_t) V(s_{t+1}) - V(s_t)
        A_t = δ_t + γλ (1 - done_t) A_{t+1}
        R_t = A_t + V(s_t)

    Args:
        rewards: 即时奖励，[batch, T]
        values: 价值估计，[batch, T]
        dones: 终止标记，[batch, T]
        gamma: 折扣因子
        gae_lambda: GAE 的 λ 参数

    Returns:
        returns: 折扣回报 R_t
        advantages: 优势 A_t
    """

    batch_size, horizon = rewards.size()
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    next_advantage = torch.zeros(batch_size, device=rewards.device)
    next_value = torch.zeros(batch_size, device=rewards.device)

    # 逆序遍历时间步，逐步回推优势
    for t in reversed(range(horizon)):
        # mask 用于在 episode 结束时截断后续折扣
        mask = 1.0 - dones[:, t]

        # TD 残差 δ_t，作为优势估计的基础
        delta = rewards[:, t] + gamma * mask * next_value - values[:, t]

        # GAE 递推公式：当前优势 = δ_t + γλ * 下一个优势
        next_advantage = delta + gamma * gae_lambda * mask * next_advantage
        advantages[:, t] = next_advantage

        # 折扣回报 R_t = A_t + V(s_t)
        returns[:, t] = advantages[:, t] + values[:, t]

        # 为下一个时间步准备 value(s_t)
        next_value = values[:, t]

    return returns, advantages


# ---------------------------------------------------------------------------
# 1. REINFORCE / Monte-Carlo Policy Gradient
# ---------------------------------------------------------------------------


def reinforce_loss(
    rollout: RolloutBatch,
    gamma: float = 0.99,
    normalize_advantage: bool = True,
) -> torch.Tensor:
    """计算 REINFORCE 损失。

    REINFORCE 核心公式：
        L = -E[ log π(a_t | s_t) * (G_t - b_t) ]

    其中 G_t 为折扣回报，这里使用 GAE 得到的优势作为 (G_t - b_t)。
    """

    # REINFORCE 常用优势替代折扣回报，提高数值稳定性
    _, advantages = compute_returns_and_gae(
        rollout.rewards, rollout.values, rollout.dones, gamma=gamma
    )

    if normalize_advantage:
        # 对优势做标准化，缓解样本方差过大的问题
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # REINFORCE 损失：对 logπ 加权求和取负号
    loss = -(rollout.log_probs_old * advantages.detach()).mean()
    return loss


# ---------------------------------------------------------------------------
# 2. PPO (Clipped Objective)
# ---------------------------------------------------------------------------


def ppo_clip_loss(
    rollout: RolloutBatch,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
) -> torch.Tensor:
    """计算 PPO 裁剪目标的总损失。

    1) 使用 GAE 计算优势 A_t
    2) 比较新旧策略概率比 r_t = exp(logπ_new - logπ_old)
    3) 策略损失：
        L^{CLIP}_t = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    4) 价值损失：0.5 * (V_new - R_t)^2
    5) 可选熵正则：-β * H[π_new]
    """

    if rollout.log_probs_new is None:
        raise ValueError("PPO 需要提供当前策略的 log_probs_new")

    # GAE 返回折扣回报与优势，其中优势用于策略损失、回报用于价值损失
    returns, advantages = compute_returns_and_gae(
        rollout.rewards, rollout.values, rollout.dones, gamma=gamma, gae_lambda=gae_lambda
    )

    # 对优势进行归一化，提高梯度稳定性
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 新旧策略的 log 概率差值 -> 概率比 r_t
    log_ratio = rollout.log_probs_new - rollout.log_probs_old
    ratio = torch.exp(log_ratio)

    # 未裁剪与裁剪后的策略损失（乘以负号表示最大化目标）
    unclipped = -ratio * advantages.detach()
    clipped = -torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages.detach()
    policy_loss = torch.max(unclipped, clipped).mean()

    # 价值损失：新价值预测与折扣回报之间的 MSE
    value_loss = 0.5 * (returns.detach() - rollout.values).pow(2).mean()

    # 熵正则：鼓励策略多样性（取负号转为损失）
    entropy = -rollout.log_probs_new.exp() * rollout.log_probs_new
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total_loss


# ---------------------------------------------------------------------------
# 3. GRPO (Generative Rollout Policy Optimization)
# ---------------------------------------------------------------------------


def grpo_loss(
    per_token_logps_new: torch.Tensor,
    per_token_logps_old: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    epsilon_high: Optional[float] = None,
    beta: float = 0.0,
    ref_logps: Optional[torch.Tensor] = None,
    loss_type: str = "grpo",
) -> torch.Tensor:
    """按 GRPO 论文/实践实现的损失。

    Args:
        per_token_logps_new: 当前策略的逐 token log π
        per_token_logps_old: rollout 时的逐 token log π
        completion_mask: 有效 token 掩码（用于可变长度序列）
        advantages: 每条序列的优势（广播到 token）
        epsilon: 下界裁剪幅度
        epsilon_high: 上界裁剪幅度（None 时等于 epsilon）
        beta: KL 正则系数
        ref_logps: 参考策略的 log π（做 KL 约束时需要）
        loss_type: grpo / bnpo / dr_grpo
    """

    if epsilon_high is None:
        epsilon_high = epsilon

    # 计算逐 token 的概率比率以及裁剪值
    ratio = torch.exp(per_token_logps_new - per_token_logps_old)
    ratio_clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon_high)

    # 将序列优势扩展到 token 维度，逐 token 复用同一优势
    advantages_expanded = advantages.unsqueeze(1)

    # 裁剪前后的逐 token 策略损失
    loss_unclipped = -ratio * advantages_expanded
    loss_clipped = -ratio_clipped * advantages_expanded
    per_token_loss = torch.min(loss_unclipped, loss_clipped)

    if beta != 0.0 and ref_logps is not None:
        # KL 正则项：约束当前策略偏离参考策略的幅度
        kl_div = torch.exp(ref_logps - per_token_logps_new) - (
            ref_logps - per_token_logps_new
        ) - 1
        per_token_loss = per_token_loss + beta * kl_div

    if loss_type == "grpo":
        # GRPO：按每条序列的有效 token 数平均
        denom = completion_mask.sum(dim=1).clamp(min=1.0)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()
    elif loss_type == "bnpo":
        # BNPO：按全 batch 的有效 token 归一化
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    elif loss_type == "dr_grpo":
        # Dr.GRPO：使用固定的分母（batch_size * max_len）
        batch, _ = per_token_loss.shape
        loss = (per_token_loss * completion_mask).sum() / (batch * completion_mask.size(1))
    else:
        raise ValueError(f"未知的 GRPO 损失类型: {loss_type}")

    return loss


# ---------------------------------------------------------------------------
# 示例：随机 rollout 数据演示三种损失的调用流程
# ---------------------------------------------------------------------------


def build_dummy_rollout(
    batch_size: int = 3,
    horizon: int = 5,
) -> RolloutBatch:
    # 固定随机种子，便于复现实验
    torch.manual_seed(0)

    # 随机生成 rollout 的基础数据
    rewards = torch.randn(batch_size, horizon)
    dones = torch.zeros(batch_size, horizon)
    dones[:, -1] = 1.0  # 每个 episode 在最后一步结束

    log_probs_old = torch.randn(batch_size, horizon)
    values = torch.randn(batch_size, horizon)
    log_probs_new = log_probs_old + 0.1 * torch.randn(batch_size, horizon)

    return RolloutBatch(
        log_probs_old=log_probs_old,
        rewards=rewards,
        dones=dones,
        values=values,
        log_probs_new=log_probs_new,
    )


def main():
    # 生成示例 rollout，用于展示三种损失的计算接口
    rollout = build_dummy_rollout()

    loss_reinforce = reinforce_loss(rollout)
    print(f"REINFORCE 损失: {loss_reinforce.item():.4f}")

    # PPO 需要新策略的 log prob（这里用旧 log prob 的扰动模拟）
    loss_ppo = ppo_clip_loss(rollout)
    print(f"PPO 损失: {loss_ppo.item():.4f}")

    # GRPO 通常以序列优势为输入，这里取每条序列的平均优势作演示
    returns, advantages = compute_returns_and_gae(
        rollout.rewards, rollout.values, rollout.dones
    )
    sequence_advantage = advantages.mean(dim=1)

    # 全 1 的 completion_mask 表示序列长度固定；实际训练中应根据 PAD/EOS 构造
    completion_mask = torch.ones_like(rollout.log_probs_old)
    grpo = grpo_loss(
        per_token_logps_new=rollout.log_probs_new,
        per_token_logps_old=rollout.log_probs_old,
        completion_mask=completion_mask,
        advantages=sequence_advantage.detach(),
    )
    print(f"GRPO 损失: {grpo.item():.4f}")


if __name__ == "__main__":
    main()
