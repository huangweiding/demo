#!/usr/bin/env python3
"""
简化版DeepSeek V3 MLA (Multi-head Latent Attention) 实现演示

这个文件用最少的代码展示了MLA的核心机制：
1. LoRA低秩分解
2. 混合位置编码 (RoPE + NOPE)
3. KV缓存压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SimpleRMSNorm(nn.Module):
    """简化的RMS归一化"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """简化的旋转位置编码"""
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SimpleMLA(nn.Module):
    """
    简化版MLA实现
    
    核心创新：
    1. LoRA分解：将线性投影分解为低秩矩阵乘积
    2. 混合位置编码：RoPE + NOPE
    3. KV缓存压缩：通过LoRA减少内存占用
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        q_lora_rank: int = 1536,      # 查询LoRA秩
        kv_lora_rank: int = 512,       # KV LoRA秩
        qk_rope_head_dim: int = 64,   # RoPE维度
        qk_nope_head_dim: int = 128,  # NOPE维度
        v_head_dim: int = 128,        # 值向量维度
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
        
        # === MLA核心：LoRA分解的查询投影 ===
        # 传统方式：hidden_size → num_heads * qk_head_dim
        # MLA方式：hidden_size → q_lora_rank → num_heads * qk_head_dim
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)      # 压缩
        self.q_a_layernorm = SimpleRMSNorm(q_lora_rank)                       # 归一化
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)  # 扩展
        
        # === MLA核心：LoRA分解的KV投影 ===
        # 传统方式：hidden_size → num_heads * (qk_head_dim + v_head_dim)
        # MLA方式：hidden_size → kv_lora_rank → num_heads * (qk_nope_head_dim + v_head_dim)
        self.kv_a_proj = nn.Linear(
            hidden_size, 
            kv_lora_rank + qk_rope_head_dim,  # 压缩KV + 直接RoPE
            bias=False
        )
        self.kv_a_layernorm = SimpleRMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, 
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False
        )
        
        # 输出投影
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        
        # 注意力缩放因子
        self.scaling = self.qk_head_dim ** (-0.5)
        
        # RoPE位置编码
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
    def _get_rotary_embeddings(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成RoPE位置编码"""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.qk_rope_head_dim, 2, device=device).float() / self.qk_rope_head_dim))
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        MLA前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # === 步骤1：LoRA查询投影 ===
        # hidden_states → q_lora_rank → num_heads * qk_head_dim
        q_compressed = self.q_a_proj(hidden_states)           # [B, L, q_lora_rank]
        q_compressed = self.q_a_layernorm(q_compressed)       # 归一化
        q_states = self.q_b_proj(q_compressed)               # [B, L, num_heads * qk_head_dim]
        
        # 重塑为多头格式
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        q_states = q_states.transpose(1, 2)  # [B, num_heads, L, qk_head_dim]
        
        # 分离RoPE和NOPE部分
        q_nope, q_rope = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # === 步骤2：LoRA KV投影 ===
        kv_compressed = self.kv_a_proj(hidden_states)        # [B, L, kv_lora_rank + qk_rope_head_dim]
        
        # 分离压缩KV和直接RoPE
        k_compressed, k_rope = torch.split(kv_compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 压缩KV的LoRA处理
        k_compressed = self.kv_a_layernorm(k_compressed)     # 归一化
        k_states = self.kv_b_proj(k_compressed)              # [B, L, num_heads * (qk_nope_head_dim + v_head_dim)]
        
        # 重塑为多头格式
        k_states = k_states.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_states = k_states.transpose(1, 2)  # [B, num_heads, L, qk_nope_head_dim + v_head_dim]
        
        # 分离键和值
        k_nope, value_states = torch.split(k_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # === 步骤3：RoPE位置编码 ===
        # 处理RoPE部分
        k_rope = k_rope.view(batch_size, 1, seq_len, self.qk_rope_head_dim)  # [B, 1, L, qk_rope_head_dim]
        
        # 生成位置编码
        cos, sin = self._get_rotary_embeddings(seq_len, hidden_states.device)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, qk_rope_head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, L, qk_rope_head_dim]
        
        # 应用RoPE
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        
        # 扩展k_rope以匹配k_nope的形状
        # origin k_rope dimension is [B, 1, L, qk_rope_head_dim]
        k_rope = k_rope.expand(batch_size, self.num_heads, seq_len, self.qk_rope_head_dim)
        
        # === 步骤4：合并位置编码 ===
        query_states = torch.cat([q_nope, q_rope], dim=-1)  # [B, num_heads, L, qk_head_dim]
        key_states = torch.cat([k_nope, k_rope], dim=-1)     # [B, num_heads, L, qk_head_dim]
        
        # === 步骤5：注意力计算 ===
        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, value_states)  # [B, num_heads, L, v_head_dim]
        
        # === 步骤6：输出投影 ===
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, num_heads, v_head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attn_output)  # [B, L, hidden_size]
        
        return output


def demo_mla():
    """演示MLA的使用"""
    print("=== DeepSeek V3 MLA 简化演示 ===\n")
    
    # 创建MLA模型
    mla = SimpleMLA(
        hidden_size=1024,      # 简化参数
        num_heads=8,
        q_lora_rank=256,       # LoRA压缩比：1024 → 256
        kv_lora_rank=128,      # LoRA压缩比：1024 → 128
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
    )
    
    # 创建输入
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 1024)
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"参数数量: {sum(p.numel() for p in mla.parameters()):,}")
    
    # 前向传播
    with torch.no_grad():
        output = mla(hidden_states)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    
    # 计算内存节省
    traditional_params = 1024 * (8 * 96) * 3  # Q, K, V投影
    mla_params = (
        1024 * 256 + 256 * (8 * 96) +  # Q LoRA
        1024 * (128 + 32) + 128 * (8 * 128)  # KV LoRA
    )
    
    print(f"\n=== 内存效率对比 ===")
    print(f"传统注意力参数: {traditional_params:,}")
    print(f"MLA注意力参数: {mla_params:,}")
    print(f"参数减少: {(1 - mla_params/traditional_params)*100:.1f}%")
    
    print(f"\n=== MLA核心优势 ===")
    print("1. LoRA低秩分解：减少参数和计算量")
    print("2. KV缓存压缩：显著减少内存占用")
    print("3. 混合位置编码：优化RoPE计算")
    print("4. 保持性能：通过精心设计的缩放保持表达能力")


if __name__ == "__main__":
    demo_mla()
